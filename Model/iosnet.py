import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from Model.unet3d.unet import UNet
from Model.attention.Decode_attention import Decoder_Attention
from Model.attention.Gauss_attention import Gauss_Attention

from utils.position_embed import get_embedder
from utils.norm_coord import make_coord

import random



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class Iosnet(nn.Module):
    def __init__(
            self,
            args,
            init_dim=1,#输入特征维度，初始是灰度图，所以是1
            features=32,#第一层的特征维度
            layer_blocks=[1,2,2,4],#unet encoder的特征层，如果是None，那就默认是[1,2,2,4]
            num_class=3,
            L=5,#坐标嵌入的维度
            mlp_decoder_hidden_dim=[512, 256, 128, 64]
    ):

        super(Iosnet, self).__init__()
        self.args=args

        # 编码部分
        self.backbone=UNet(n_features=init_dim,base_width=features,n_outputs=num_class,encoder_blocks=layer_blocks)
        self.feature_dim=[32,64,128,256]
        self.layer_blocks = layer_blocks

        #解码部分
        self.mlp_decoder_dim=mlp_decoder_hidden_dim
        self.mlp_seg_dim=[32,16]
        self.mlp_reconstruct_dim=[32,16]

        mlp_input_dim=np.sum(self.feature_dim)+3

        #self.mlp_student=MLP(mlp_input_dim,num_class,hidden_list=self.mlp_decoder_dim)
        self.mlp_student_decoder=MLP(mlp_input_dim,self.mlp_decoder_dim[-1],hidden_list=self.mlp_decoder_dim[:-1])
        self.mlp_student_seg=MLP(self.mlp_decoder_dim[-1],num_class,hidden_list=self.mlp_seg_dim)


    def feature_sample(self,feat, coord):
        # feat: [B, C, h, w, z]
        # coord: [B, N, 3], N <= H * W,是归一化的
        b, c, h, w, z = feat.shape  # lr
        B, N, _ = coord.shape

        feat_coord = make_coord((h, w, z), flatten=False).to(feat.device).permute(3, 0, 1, 2).contiguous().unsqueeze(
            0).expand(b, 3, h, w, z)  # batch * 3 * h * w * z,归一化坐标

        rx = 1 / h
        ry = 1 / w
        rz = 1 / z

        preds = []

        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                for vz in [-1, 1]:
                    coord_ = coord.clone()
                    coord_[:, :, 0] += (vx) * rx
                    coord_[:, :, 1] += (vy) * ry
                    coord_[:, :, 2] += (vz) * rz
                    k += 1
                    # feat: [B, c, h, w], coord_: [B, N, 3] --> [B, 1, 1， N, 3], out: [B, c, 1, 1, N] --> [B, c, N] --> [B, N, c]
                    #flip把坐标从[x,y,z]->[z,y,x]
                    q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                             :, 0, 0, :].permute(0, 2, 1).contiguous()  # [B, N, c]
                    q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1), mode='nearest',
                                            align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1).contiguous()  # [B, N, 3]

                    rel_coord = coord - q_coord  # 真实坐标和偏移的采样坐标之间的偏差
                    rel_coord=self.pos_embed(rel_coord)
                    q_coord = self.pos_embed(q_coord)
                    # rel_coord[:, :, 0] *= h
                    # rel_coord[:, :, 1] *= w
                    # rel_coord[:, :, 2] *= z

                    inp = torch.cat([q_feat, q_coord, rel_coord], dim=-1)  # [B,N,c+3],特征+相对坐标
                    preds.append(inp)

        preds = torch.stack(preds, dim=2)  # [B, N, k*k*k=8, c+3]

        b, n, num,c = preds.shape

        preds=preds.view(b*n,num,c)

        return preds
    def forward_test(self, x, coords_all):
        B,N,_=coords_all.shape

        x = self.backbone.encoder(x)
        unet_out = self.backbone.final_convolution(self.backbone.decoder(x))
        # x:[layer4_feature(256),layer3_feature(128),layer2_feature(64),layer1_feature(32)]

        split_num = 512
        cut_length = N // split_num
        result = []
        for item in range(split_num):
            print(item ,' / ', split_num)
            coords = coords_all[:, cut_length * item:cut_length * (item + 1), :]

            sample_feature = []
            for step, i in enumerate(x):
                # i:batch*feature_dim*h*w*z
                layer_feature = self.feature_sample(i, coords)
                layer_feature = getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature)
                sample_feature.append(layer_feature)
                # sample_feature:4,B*N,1,c

            sample_feature=torch.concat(sample_feature,dim=-1)#B*N*8,(c1+c2+c3+c4)
            sample_feature=sample_feature.reshape((B,cut_length,-1))#B,N,c

            mlp_decoder_feature = self.mlp_student_decoder(sample_feature)
            mlp_decoder_feature = torch.relu(mlp_decoder_feature)
            student_class = self.mlp_student_seg(mlp_decoder_feature)
            result.append(student_class)# 这里是标记的像素的输出，直接计算ce损失

        result = torch.concat(result,dim=1)
        result=result.view(B,512,512,256,3)#B,N,num_classes
        result = result.permute(0,4,1,2,3)

        return unet_out,result

    def forward(self, x, coords, teacher_need=False):

        B,N,_=coords.shape


        x = self.backbone.encoder(x)
        unet_out = self.backbone.final_convolution(self.backbone.decoder(x))
        # x:[layer4_feature(256),layer3_feature(128),layer2_feature(64),layer1_feature(32)]

        sample_feature = []
        for step, i in enumerate(x):
            # i:batch*feature_dim*h*w*z
            layer_feature = self.feature_sample(i, coords)
            layer_feature = getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature)
            sample_feature.append(layer_feature)
            # sample_feature:4,B*N,1,c

        sample_feature=torch.concat(sample_feature,dim=-1)#B*N*8,(c1+c2+c3+c4)
        sample_feature=sample_feature.reshape((B,N,-1))#B,N,c

        #sample_feature=torch.rand_like(sample_feature,dtype=torch.float32,device=sample_feature.device)
        # mlp_out = self.mlp_student(sample_feature)
        mlp_decoder_feature = self.mlp_student_decoder(sample_feature)
        mlp_decoder_feature = torch.relu(mlp_decoder_feature)
        student_class = self.mlp_student_seg(mlp_decoder_feature[:, :-self.args.sample_num, :])  # 这里是标记的像素的输出，直接计算ce损失
        student_reconstruct = self.mlp_student_reconstruct(mlp_decoder_feature)  # 用于计算重建mse损失，包含了有标签坐标和没标签坐标，正常输入


        return unet_out,student_class,student_reconstruct


