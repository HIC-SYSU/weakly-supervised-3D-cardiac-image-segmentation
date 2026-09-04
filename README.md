# Usage

## Model

DPT model consisting of:

- **UNet branch** (256×256×128) for CT segmentation
- **Point/MLP reconstruction branch** (512×512×256) for high-resolution reconstruction

The segmentation targets include the **left ventricular myocardium, left ventricular blood pool, and right ventricle**.

**Environment:** Python 3.8


## Core Files

| File | Description |
| --- | --- |
| `entire_image_train.py` | Model training (see the script for detailed usage) |
| `entire_image_test.py` | Test-set inference and Dice evaluation (see the script for detailed usage) |
| `Model/model_copy.py` | Model definition |


## Results

- `result/ckpt/model_best_mean_dice.pth`  
  Best validation checkpoint with a **mean Dice score of 0.9343**  
  (class_1 = 0.8755 / class_2 = 0.9608 / class_3 = 0.9665)

- `result/test/`  
  Contains visualizations of the segmentation results:
  - `images/`
  - `label_GT/`
  - `label_pred/`
  - `dice.json`


## Citation
@article{lin2026adversarial,
  title={Adversarial-consistency enhanced implicit segmentation field for weakly supervised 3D cardiac image segmentation},
  author={Lin, Weiyuan and Zhong, Juntao and Gao, Zhifan and Chen, Jinfeng and Zhao, Jichao and Wu, Weiwen and Xu, Chenchu and Shi, Changzheng and Zhang, Zhihui and Liu, Xiujian},
  journal={Medical Image Analysis},
  pages={104094},
  year={2026},
  publisher={Elsevier}
}

