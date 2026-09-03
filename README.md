# one_encoder_40140 使用说明

DPT 模型：**UNet 分支**（256×256×128）做 CT 分割 + **点/MLP 重建分支**（512×512×256），分割目标为左心室心肌/血池/右心室。
环境：`conda activate CT_LVM`（Python 3.8）。

## 核心文件

| 文件 | 作用 |
| `entire_image_train.py` | 训练（详细用法见脚本） |
| `entire_image_test.py` | 147 例测试集推理 + Dice（详细用法见脚本） |
| `Model/model_copy.py` | 模型定义 |


## 结果

- `result/ckpt/model_best_mean_dice.pth`：
验证集 mean Dice **0.9343**（class_1=0.8755 / class_2=0.9608 / class_3=0.9665），详见 [result/ckpt/README.md]
- `result/test/`：最终保留了 5 例最好结果的可视化（`images/`、`label_GT/`、`label_pred/` + `dice.json`）


## PS:原始版本在/data/zhongjuntao/Model_Data/Model/one_encoder_40140
- `本版本只修改了钟俊涛方法使其可运行(/data/zhongjuntao/Model_Data/zjtxiugaihao/one_encoder_40140)`
