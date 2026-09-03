import argparse
import os
usage_text = (
    "SDNet Pytorch Implementation"
    "Usage:  python train.py [options],"
    "   with [options]:"
)
parser = argparse.ArgumentParser(description=usage_text)


'''
基本参数：
'''
parser.add_argument("--root_path",type=str,default='/data/zhongjuntao/Model_Data')
parser.add_argument("--data_path", type=str, default='Data', help='数据集总目录')
parser.add_argument('--dataset',type=str,default='LVM',help='要加载的数据集名称',choices=['LVM',"WORD"])
parser.add_argument('--dataset_infor_root',type=str,
                    default='/data/zhongjuntao/Model_Data/Data/{}/dataset.json',help='数据集分割保存路径')

parser.add_argument('--save_path', type=str,default='Model/{}/result'.format(os.path.abspath('').split('/')[-1]), help='模型训练参数/记录保存结果')
parser.add_argument('--pretrain',type=bool,default=True,help='is pretrain')
parser.add_argument('--pretrained_weights',type=str,default='Model/{}/pre_train'.format(os.path.abspath('').split('/')[-1]))
parser.add_argument('-wi', '--weight_init', type=str, default="xavier",help='模型随机初始化方式')
parser.add_argument('-n_labels', '--n_labels', type=int, default=3, help='LVM:3, WORD:8')
parser.add_argument('-n_points', '--n_points', type=int, default=5, help='点标记中每层标记几个点，因为点标记有好几个label，所以要设置好选择哪个数据集label')
parser.add_argument('--labels_type', type=str, default='point', help='标签种类,scribble,point')
'''
训练参数
'''
parser.add_argument('--mode',type=str,default='train',help='现在的mode',choices=['train','test'])
parser.add_argument('--epochs',type=int,default=1500)
parser.add_argument('--eval_step',type=int,default=10,help='几个epoch计算一次验证集')
parser.add_argument('--mse_epoch',type=int,default=0,help='重建轮次')
parser.add_argument('--pre_train_epoch',type=int,default=0,help='全监督的epoch数量')
parser.add_argument('-batch_size', '--batch_size', type=int, default=1, help='Number of inputs per batch')
parser.add_argument('-g', '--gpu', type=str, default='0',help='gpu序号')
parser.add_argument('-lr', '--learning_rate', type=dict, default={'encoder':0.01,'unet_decoder':0.0001,'mlp_decoder':0.0001/2}, help='学习率')
parser.add_argument('--k',type=float,default=0.2,help='测试集占总训练集的数量')
parser.add_argument('--ignore_label',type=int,default=255,help='未标记的label的标签')
parser.add_argument('--sample_num',type=int,default=0,help='采样的标记点个数，前期可能不够，就直接按照最小的样本量采')
parser.add_argument('--ema_decay_origin',type=float,default=0.999,help='ema的调整参数')


'''
预处理参数
'''
parser.add_argument('--img_size',type=tuple,default=(512,512,256),help='缩放的大小.')
parser.add_argument('--patch_size',type=tuple,default=(128,128,64),help='缩放的大小.')
parser.add_argument('--axis',type=list,default=(0,1,2,3),help='转置的参数，原来的坐标轴为X,Y,Z,冠状：0231，矢状位：0132')
parser.add_argument('--pix_value',type=list,default=[0,700],help='像素值的范围，超出的直接截断,LVM:[0,700],WORD:[-400,1000]')

'''
数据集读取参数
'''
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--cache_num',type=int,default=2)
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use for dataload')







args = parser.parse_args()
args.dataset_infor_root = args.dataset_infor_root.format(args.dataset)
args.data_path=os.path.join(args.root_path,args.data_path)
args.save_path=os.path.join(args.root_path,args.save_path)
args.pretrained_weights=os.path.join(args.root_path,args.pretrained_weights,args.dataset,'pre_model.pth')
args.save_path=os.path.join(args.save_path,args.dataset)