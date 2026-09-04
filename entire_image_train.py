"""
Train DPT (UNet segmentation + point/MLP reconstruction branch) on 430 CT cases.

Usage:
    nohup python one_encoder_40140/entire_image_train.py \
      --gpu 1 --epochs 500 --sample-num 2048 --save-every 50 &

Parameters that may need to be changed:
    --image-dir / --label-dir / --save-dir / --gpu / --epochs /
    --sample-num / --save-every / --val-count / --val-every
    (Keep all other parameters at their default values.)

    python one_encoder_40140/entire_image_train.py

Default settings:
    (You can directly modify DEFAULT_IMAGE_DIR / DEFAULT_LABEL_DIR.)

    Images:
        --image-dir /data/zhongjuntao/Model_Data/Model/SAM-Med3D-main/visualization/images

    Labels:
        --label-dir /data/zhongjuntao/Model_Data/Model/SAM-Med3D-main/visualization/label_sam_430_255
        (430 cases, paired with images by identical filenames)

    Save directory:
        --save-dir result/train400_epoch500
        (The directory and log file will be created automatically.)

    Dataset split:
        train = 400 / val = 30
        (--val-count 30, seed 0)

        Validation is performed every --val-every epochs.

        --val-count specifies the number of cases reserved from the
        full dataset as the validation set.

Input:
    Images and labels are paired one-to-one using identical .nii.gz filenames.
    Label values are in {0, 1, 2, 3, 255}.
    Label 255 is ignored and excluded from loss computation.

Output:
    Files saved under --save-dir:

    model_best_mean_dice.pth
        Model weights with the highest mean Dice score on the validation set.

    model_epoch_{N}.pth
        Model checkpoint saved every --save-every epochs.

    train_nohup.log
        Training log automatically written by the script.
        No additional `>` redirection is required in the command.

Common commands:

    Train for 500 epochs:
        python one_encoder_40140/entire_image_train.py \
          --gpu 1 --epochs 500 --sample-num 2048 --save-every 50

    Resume training:
        python one_encoder_40140/entire_image_train.py \
          --gpu 1 \
          --checkpoint result/train400_epoch500/model_epoch_100.pth

    Validate/check data only:
        python one_encoder_40140/entire_image_train.py --validate-only

    Smoke test (run only a few steps):
        python one_encoder_40140/entire_image_train.py --max-steps 2
"""

import argparse
import os
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
)
from monai.utils import set_determinism
from tqdm import tqdm

from Model.model_copy import DPT


DEFAULT_IMAGE_DIR = Path("/data/zhongjuntao/Model_Data/Model/SAM-Med3D-main/visualization/images")
DEFAULT_LABEL_DIR = Path("/data/zhongjuntao/Model_Data/Model/SAM-Med3D-main/visualization/label_sam_430_255")
SPATIAL_SIZE = (512, 512, 256)
UNET_SIZE = (256, 256, 128)
NUM_CLASSES = 4

def parse_args():
    parser = argparse.ArgumentParser(description="Train DPT using whole CT volumes")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parent / "result" / "train400_epoch500")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gpu", default="0", help="CUDA device number")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--sample-num", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--val-count", type=int, default=30, help="Number of cases randomly reserved for validation")
    parser.add_argument("--val-every", type=int, default=10, help="Run validation every N epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit steps per epoch; useful for a smoke test")
    parser.add_argument("--validate-only", action="store_true", help="Only check image/label pairing and NIfTI metadata")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_file_list(image_dir, label_dir):
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    images = {p.name: p for p in image_dir.glob("*.nii.gz") if p.exists()}
    labels = {p.name: p for p in label_dir.glob("*.nii.gz") if p.exists()}
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise ValueError(
            f"Unpaired files: missing labels={missing_labels[:5]}, "
            f"missing images={missing_images[:5]}"
        )
    if not images:
        raise ValueError("No paired .nii.gz files were found")
    return [
        {"image": str(images[name]), "label": str(labels[name])}
        for name in sorted(images)
    ]


def validate_files(files):
    mismatches = []
    for item in files:
        image = nib.load(item["image"])
        label = nib.load(item["label"])
        if image.shape != label.shape:
            mismatches.append((Path(item["image"]).name, image.shape, label.shape))
    if mismatches:
        raise ValueError(f"Image/label shape mismatch: {mismatches[:5]}")

    first_label = np.asanyarray(nib.load(files[0]["label"]).dataobj)
    values = set(np.unique(first_label).tolist())
    allowed = set(range(NUM_CLASSES)) | {255}
    if not values <= allowed:
        raise ValueError(
            f"Unexpected labels in {Path(files[0]['label']).name}: {sorted(values)}; "
            f"expected 0..{NUM_CLASSES - 1} or 255"
        )
    print(
        f"数据检查通过：{len(files)} 对 NIfTI；首个标签值={sorted(values)}；"
        f"网络输入尺寸={UNET_SIZE}。"
    )


def get_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(
                keys=["image", "label"], spatial_size=SPATIAL_SIZE,
                mode=("trilinear", "nearest"),
            ),
            CopyItemsd(
                keys=["image", "label"], names=["unet_image", "unet_label"],
            ),
            Resized(
                keys=["unet_image", "unet_label"], spatial_size=UNET_SIZE,
                mode=("trilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image", "unet_image"], a_min=-400, a_max=1000,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            EnsureTyped(keys=["image", "label", "unet_image", "unet_label"]),
        ]
    )


def get_validation_transforms():
    """Only prepare the low-resolution UNet inputs to keep validation inexpensive."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Resized(
                keys=["image", "label"], spatial_size=UNET_SIZE,
                mode=("trilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-400, a_max=1000,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def split_train_validation(files, val_count, seed):
    if val_count <= 0:
        raise ValueError("--val-count must be greater than 0")
    if val_count >= len(files):
        raise ValueError(
            f"--val-count ({val_count}) must be smaller than the dataset size ({len(files)})"
        )
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    return shuffled[val_count:], shuffled[:val_count]


def sample_voxels(labels, sample_num, ignore_label=255):
    """Sample equal upper-bounded counts from labeled and ignored voxels."""
    labeled_items = []
    ignored_items = []
    for label in labels[:, 0]:
        labeled_items.append(torch.nonzero(label != ignore_label, as_tuple=False))
        ignored_items.append(torch.nonzero(label == ignore_label, as_tuple=False))

    labeled_n = min([sample_num] + [len(x) for x in labeled_items])
    ignored_n = min([sample_num] + [len(x) for x in ignored_items])
    if labeled_n == 0:
        raise ValueError("A batch has no labeled voxels (all voxels are 255)")

    def choose(items, count):
        if count == 0:
            return labels.new_empty((labels.shape[0], 0, 3), dtype=torch.long)
        result = []
        for coords in items:
            ids = torch.randperm(len(coords), device=coords.device)[:count]
            result.append(coords[ids])
        return torch.stack(result)

    labeled_coords = choose(labeled_items, labeled_n)
    ignored_coords = choose(ignored_items, ignored_n)
    targets = []
    for batch_index, coords in enumerate(labeled_coords):
        targets.append(labels[batch_index, 0, coords[:, 0], coords[:, 1], coords[:, 2]])
    return labeled_coords, ignored_coords, torch.stack(targets).long()


def normalize_coords(coords):
    scale = coords.new_tensor(SPATIAL_SIZE, dtype=torch.float32) - 1
    return coords.float() * (2.0 / scale) - 1.0


def load_checkpoint(model, checkpoint, device):
    if checkpoint is None:
        print("未指定 checkpoint，使用 PyTorch 默认随机初始化。")
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    print(f"已加载 checkpoint：{checkpoint}")


@torch.no_grad()
def run_validation(model, loader, ce_loss, device, use_amp):
    """Validate only the UNet branch and report foreground Dice scores."""
    model.eval()
    loss_total = 0.0
    intersections = torch.zeros(NUM_CLASSES - 1, device=device)
    denominators = torch.zeros(NUM_CLASSES - 1, device=device)

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).long()[:, 0]
        with torch.cuda.amp.autocast(enabled=use_amp):
            features = model.backbone.encoder(image)
            logits = model.backbone.final_convolution(model.backbone.decoder(features))
            loss_total += ce_loss(logits, label).item()

        prediction = logits.argmax(dim=1)
        valid_mask = label != 255
        for index, class_id in enumerate(range(1, NUM_CLASSES)):
            pred_class = (prediction == class_id) & valid_mask
            target_class = (label == class_id) & valid_mask
            intersections[index] += 2.0 * (pred_class & target_class).sum()
            denominators[index] += pred_class.sum() + target_class.sum()

    dice = torch.where(
        denominators > 0,
        intersections / denominators.clamp_min(1),
        torch.full_like(denominators, float("nan")),
    )
    finite_dice = dice[torch.isfinite(dice)]
    mean_dice = finite_dice.mean().item() if len(finite_dice) else float("nan")
    dice_text = ", ".join(
        f"class_{class_id}={score:.4f}"
        for class_id, score in zip(range(1, NUM_CLASSES), dice.tolist())
    )
    print(
        f"Validation: loss={loss_total / len(loader):.4f}, "
        f"mean_dice={mean_dice:.4f}, {dice_text}"
    )
    model.train()
    return mean_dice


def save_checkpoint(path, model, optimizer, epoch, args):
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "epoch": epoch, "args": vars(args)},
        path,
    )


def train(args, train_files, val_files):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this 3D model requires a CUDA GPU")

    dataset = CacheDataset(
        data=train_files, transform=get_transforms(), cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_dataset = CacheDataset(
        data=val_files, transform=get_validation_transforms(),
        cache_rate=0.0, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print(
        f"数据划分：train={len(train_files)}，validation={len(val_files)}；"
        f"每次验证使用全部 {len(val_files)} 例。"
    )

    device = torch.device("cuda:0")
    model = DPT(args=args).to(device)
    load_checkpoint(model, args.checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    mse_loss = torch.nn.MSELoss()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    center = (torch.tensor(SPATIAL_SIZE, device=device, dtype=torch.float32) / 2).unsqueeze(0)
    model.train()
    best_mean_dice = -1.0

    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress):
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            unet_image = batch["unet_image"].to(device, non_blocking=True)
            unet_label = batch["unet_label"].to(device, non_blocking=True).long()[:, 0]

            labeled_coords, ignored_coords, point_target = sample_voxels(label, args.sample_num)
            all_coords = torch.cat([labeled_coords, ignored_coords], dim=1)
            normalized_coords = normalize_coords(all_coords)
            reconstruct_target = []
            for batch_index, coords in enumerate(all_coords):
                reconstruct_target.append(
                    image[batch_index, 0, coords[:, 0], coords[:, 1], coords[:, 2]]
                )
            reconstruct_target = torch.stack(reconstruct_target)
            batch_center = center.expand(image.shape[0], -1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                unet_out, point_out, reconstruct_out = model(
                    unet_image, normalized_coords, batch_center, SPATIAL_SIZE,
                )
                point_out = point_out[:, : labeled_coords.shape[1]].permute(0, 2, 1)
                reconstruct_out = reconstruct_out.squeeze(-1)
                loss_unet = ce_loss(unet_out, unet_label)
                loss_point = ce_loss(point_out, point_target)
                loss_reconstruct = mse_loss(reconstruct_out, reconstruct_target)
                loss = loss_unet + loss_point + 10.0 * loss_reconstruct

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.set_postfix(
                loss=f"{loss.item():.4f}", unet=f"{loss_unet.item():.4f}",
                point=f"{loss_point.item():.4f}", mse=f"{loss_reconstruct.item():.4f}",
            )
            if args.max_steps is not None and step + 1 >= args.max_steps:
                break

        current_epoch = epoch + 1
        if current_epoch % args.val_every == 0 or current_epoch == args.epochs:
            mean_dice = run_validation(
                model, val_loader, ce_loss, device, use_amp=not args.no_amp,
            )
            if np.isfinite(mean_dice) and mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_output = args.save_dir / "model_best_mean_dice.pth"
                save_checkpoint(
                    best_output, model, optimizer, current_epoch, args,
                )
                print(
                    f"Validation mean Dice 提升到 {best_mean_dice:.4f}；"
                    f"已保存：{best_output}"
                )

        if current_epoch % args.save_every == 0 or current_epoch == args.epochs:
            output = args.save_dir / f"model_epoch_{epoch + 1}.pth"
            save_checkpoint(output, model, optimizer, current_epoch, args)
            print(f"已保存：{output}")


def setup_logging(save_dir):
    """自动创建保存目录，并把输出写入 save_dir/train_nohup.log。

    这样直接运行脚本即可，无需在 shell 里手动 `> 日志 2>&1`；
    目录不存在时由脚本自动生成。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_nohup.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file


def main():
    args = parse_args()
    setup_logging(args.save_dir)
    if args.sample_num <= 0:
        raise ValueError("--sample-num must be greater than 0")
    if args.val_every <= 0:
        raise ValueError("--val-every must be greater than 0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_determinism(seed=args.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")

    files = build_file_list(args.image_dir, args.label_dir)
    validate_files(files)
    if not args.validate_only:
        train_files, val_files = split_train_validation(
            files, args.val_count, args.seed,
        )
        train(args, train_files, val_files)


if __name__ == "__main__":
    main()
