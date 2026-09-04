# 使用说明

DPT 模型：**UNet 分支**（256×256×128）做 CT 分割 + **点/MLP 重建分支**（512×512×256），分割目标为左心室心肌/血池/右心室。
环境：`Python 3.8。

## 核心文件

| 文件 | 作用 |
| `entire_image_train.py` | 训练（详细用法见脚本） |
| `entire_image_test.py` |测试集推理 + Dice（详细用法见脚本） |
| `Model/model_copy.py` | 模型定义 |


## 结果

- `result/ckpt/model_best_mean_dice.pth`：
验证集 mean Dice **0.9343**（class_1=0.8755 / class_2=0.9608 / class_3=0.9665）
- `result/test/`：最终保留了相关结果的可视化（`images/`、`label_GT/`、`label_pred/` + `dice.json`）

## Citation
@article{lin2026adversarial,
  title={Adversarial-consistency enhanced implicit segmentation field for weakly supervised 3D cardiac image segmentation},
  author={Lin, Weiyuan and Zhong, Juntao and Gao, Zhifan and Chen, Jinfeng and Zhao, Jichao and Wu, Weiwen and Xu, Chenchu and Shi, Changzheng and Zhang, Zhihui and Liu, Xiujian},
  journal={Medical Image Analysis},
  pages={104094},
  year={2026},
  publisher={Elsevier}
}

