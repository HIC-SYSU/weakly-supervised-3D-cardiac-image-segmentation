"""147 例济南 2016.10 CT 上运行 UNet 分支推理并计算 Dice。

用法：
    python one_encoder_40140/entire_image_test.py

默认值：(可以直接改DEFAULT_IMAGE_DIR——DEFAULT_OUTPUT_DIR几个参数)
    图像输入    --image-dir /data/zhongjuntao/ALL_Data/2016.10            (这个有147例)
    GT         --gt-dir /data/zhongjuntao/ALL_Data/label_jinan/2016.10_147已完成
    权重       --checkpoint result/train400_epoch500/model_best_mean_dice.pth
    输出       --output-dir result/test_147_train400_bestepoch160

输入：图像与 GT 同名 .nii.gz 一一配对,GT 标签 ∈ {0,1,2,3}。
输出：每例 {case}_pred.nii.gz(还原到原图空间的 uint8 预测)
      以及 dice.json(逐例 class_1/2/3/mean_dice + 汇总包括最好的那例和平均dice)。

其他参数用法：
    单例      python one_encoder_40140/entire_image_test.py --case aixinghui
    前 N 例   python one_encoder_40140/entire_image_test.py --limit 5
    覆盖重跑  python one_encoder_40140/entire_image_test.py --overwrite
"""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from Model.model_copy import DPT


DEFAULT_IMAGE_DIR = Path("/data/zhongjuntao/ALL_Data/2016.10")
DEFAULT_GT_DIR = Path("/data/zhongjuntao/ALL_Data/label_jinan/2016.10_147已完成")
DEFAULT_CHECKPOINT = Path("/data/zhongjuntao/Model_Data/duibishiyan/one_encoder_40140/result/ckpt/model_best_mean_dice.pth")
DEFAULT_OUTPUT_DIR = Path("/data/zhongjuntao/Model_Data/duibishiyan/one_encoder_40140/result/test_147_train400_bestepoch160")
UNET_SIZE = (256, 256, 128)
FOREGROUND_CLASSES = (1, 2, 3)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Dice for test_147")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gpu", default="0",
        help="Physical GPU number; CUDA maps it to logical cuda:0 inside this process",
    )
    parser.add_argument(
        "--case", default=None,
        help="Only process this filename or stem, for example aixinghui",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N cases")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def discover_cases(image_dir, gt_dir, selected_case=None, limit=None):
    images = {p.name: p for p in image_dir.glob("*.nii.gz")}
    ground_truths = {p.name: p for p in gt_dir.glob("*.nii.gz")}
    missing_gt = sorted(images.keys() - ground_truths.keys())
    if missing_gt:
        raise ValueError(f"缺少同名 GT（前5个）：{missing_gt[:5]}")

    names = sorted(images.keys() & ground_truths.keys())
    if selected_case:
        filename = selected_case if selected_case.endswith(".nii.gz") else selected_case + ".nii.gz"
        if filename not in images or filename not in ground_truths:
            raise FileNotFoundError(f"找不到同名 image/GT：{filename}")
        names = [filename]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than 0")
        names = names[:limit]
    if not names:
        raise ValueError("没有发现可测试的同名 NIfTI")
    return [(images[name], ground_truths[name]) for name in names]


def prepare_image(image_path, device):
    """Canonicalize image to RAS, resize and normalize exactly as validation."""
    original = nib.load(str(image_path))
    canonical = nib.as_closest_canonical(original)
    image = np.asarray(canonical.dataobj, dtype=np.float32)
    image = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(device)
    image = F.interpolate(
        image, size=UNET_SIZE, mode="trilinear", align_corners=False,
    )
    image = ((image.clamp(-400, 1000) + 400.0) / 1400.0).contiguous()
    return image, original, canonical


def restore_prediction(prediction, original, canonical):
    """Resize a RAS prediction and transform it back to the original voxel order."""
    prediction = torch.from_numpy(prediction.astype(np.float32))[None, None]
    prediction = F.interpolate(
        prediction, size=canonical.shape, mode="nearest",
    )[0, 0].numpy().astype(np.uint8)

    canonical_orientation = nib.orientations.io_orientation(canonical.affine)
    original_orientation = nib.orientations.io_orientation(original.affine)
    transform = nib.orientations.ornt_transform(
        canonical_orientation, original_orientation,
    )
    restored = nib.orientations.apply_orientation(prediction, transform)
    if restored.shape != original.shape:
        raise ValueError(
            f"恢复后的预测尺寸 {restored.shape} 与原图 {original.shape} 不一致"
        )
    return np.ascontiguousarray(restored)


def calculate_dice(prediction, ground_truth):
    values = set(np.unique(ground_truth).tolist())
    allowed = {0, 1, 2, 3}
    if not values <= allowed:
        raise ValueError(f"GT 出现未知标签：{sorted(values)}")

    result = {}
    scores = []
    for class_id in FOREGROUND_CLASSES:
        pred_mask = prediction == class_id
        gt_mask = ground_truth == class_id
        denominator = int(pred_mask.sum() + gt_mask.sum())
        if denominator == 0:
            score = None
        else:
            intersection = int(np.logical_and(pred_mask, gt_mask).sum())
            score = 2.0 * intersection / denominator
            scores.append(score)
        result[f"class_{class_id}"] = score
    result["mean_dice"] = float(np.mean(scores)) if scores else None
    return result


def save_prediction(prediction, original, output_path):
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(prediction, original.affine, header=header)
    output.set_qform(original.get_qform(), int(original.header["qform_code"]))
    output.set_sform(original.get_sform(), int(original.header["sform_code"]))
    nib.save(output, str(output_path))


def load_model(checkpoint, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"权重不存在：{checkpoint}")
    model = DPT(args=SimpleNamespace()).to(device)
    checkpoint_data = torch.load(str(checkpoint), map_location=device)
    state = (
        checkpoint_data["model"]
        if isinstance(checkpoint_data, dict) and "model" in checkpoint_data
        else checkpoint_data
    )
    model.load_state_dict(state)
    model.eval()
    epoch = checkpoint_data.get("epoch") if isinstance(checkpoint_data, dict) else None
    print(f"已加载权重：{checkpoint}（epoch={epoch}）")
    return model


def update_json(output_dir, cases):
    dice_keys = ("class_1", "class_2", "class_3", "mean_dice")
    compact_cases = {
        name: {key: entry.get(key) for key in dice_keys}
        for name, entry in sorted(cases.items())
    }
    valid_cases = {
        name: entry
        for name, entry in compact_cases.items()
        if entry["mean_dice"] is not None
    }
    if valid_cases:
        best_name, best_entry = max(
            valid_cases.items(), key=lambda item: item[1]["mean_dice"],
        )
        summary = {"best_case": best_name, **best_entry}
        # 所有有效病例的逐类均值 Dice（class_1/2/3 及 mean_dice 各自取平均）
        all_cases_mean = {}
        for key in dice_keys:
            values = [
                entry[key]
                for entry in valid_cases.values()
                if entry.get(key) is not None
            ]
            all_cases_mean[key] = float(np.mean(values)) if values else None
        summary["all_cases_mean_dice"] = all_cases_mean
    else:
        summary = {"best_case": None, **{key: None for key in dice_keys}}

    content = {"cases": compact_cases, "summary": summary}
    json_path = output_dir / "dice.json"
    temporary_path = output_dir / "dice.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, json_path)
    return json_path


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法执行该 3D 模型推理")

    cases = discover_cases(args.image_dir, args.gt_dir, args.case, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dice.json"
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as file:
            results = json.load(file).get("cases", {})
    else:
        results = {}

    device = torch.device("cuda:0")
    model = load_model(args.checkpoint, device)
    print(f"本次处理 {len(cases)} 例；物理 GPU={args.gpu}，进程内设备={device}")

    for index, (image_path, gt_path) in enumerate(cases, start=1):
        name = image_path.name[:-len(".nii.gz")]
        prediction_path = args.output_dir / f"{name}_pred.nii.gz"
        started = time.time()

        if prediction_path.is_file() and not args.overwrite:
            original = nib.load(str(image_path))
            prediction = np.asanyarray(nib.load(str(prediction_path)).dataobj).astype(np.uint8)
            print(f"[{index}/{len(cases)}] 复用已有预测：{prediction_path.name}")
        else:
            image, original, canonical = prepare_image(image_path, device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=not args.no_amp):
                features = model.backbone.encoder(image)
                logits = model.backbone.final_convolution(model.backbone.decoder(features))
                prediction_ras = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            prediction = restore_prediction(prediction_ras, original, canonical)
            save_prediction(prediction, original, prediction_path)

        ground_truth = np.asanyarray(nib.load(str(gt_path)).dataobj).astype(np.uint8)
        if ground_truth.shape != prediction.shape:
            raise ValueError(
                f"{name}: GT {ground_truth.shape} 与预测 {prediction.shape} 尺寸不一致"
            )
        dice = calculate_dice(prediction, ground_truth)
        results[name] = dice
        written_json = update_json(args.output_dir, results)
        print(
            f"[{index}/{len(cases)}] {name}: "
            f"Dice(c1={dice['class_1']:.4f}, c2={dice['class_2']:.4f}, "
            f"c3={dice['class_3']:.4f}, mean={dice['mean_dice']:.4f})；"
            f"耗时 {time.time() - started:.1f}s"
        )

    print(f"预测目录：{args.output_dir}")
    print(f"Dice 文件：{written_json}")


if __name__ == "__main__":
    main()
