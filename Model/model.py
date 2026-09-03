import time

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

class DPT(nn.Module):
    def __init__(
            self,
            args,
            init_dim=1,#输入特征维度，初始是灰度图，所以是1
            features=32,#第一层的特征维度
            layer_blocks=[1,2,2,4],#unet encoder的特征层，如果是None，那就默认是[1,2,2,4]
            L=5,#坐标嵌入的维度
            mlp_decoder_hidden_dim=[512, 256, 128, 64]
    ):

        super(DPT, self).__init__()
        self.args=args

        self.coords_all_ = make_coord(args.img_size, flatten=False)
        self.coords_patch_ = make_coord(args.patch_size, flatten=False)

        # 编码部分
        self.backbone=UNet(n_features=init_dim,base_width=features,n_outputs=args.n_labels,encoder_blocks=layer_blocks)
        self.feature_dim=[32,64,128,256]
        self.layer_blocks = layer_blocks

        #采样部分（特征提取）
        self.pos_embed,pos_dim=get_embedder(L)
        self.layer4_uni = Decoder_Attention(self.feature_dim[0] + pos_dim*2,head_dim=self.feature_dim[0]+pos_dim*2)
        self.layer3_uni = Decoder_Attention(self.feature_dim[1] + pos_dim * 2,head_dim=self.feature_dim[1]+pos_dim*2)
        self.layer2_uni = Decoder_Attention(self.feature_dim[2] + pos_dim * 2,head_dim=self.feature_dim[2]+pos_dim*2)
        self.layer1_uni = Decoder_Attention(self.feature_dim[3] + pos_dim * 2,head_dim=self.feature_dim[3]+pos_dim*2)
        self.norm_layer=torch.nn.GroupNorm
        self.norm=[self.create_norm_layer(i).cuda() for i in [256,128,64,32]]

        #self.batchnorm1d=torch.nn.BatchNorm1d(744).cuda()

        #解码部分
        self.mlp_decoder_dim=mlp_decoder_hidden_dim
        self.mlp_seg_dim=[32,16]
        self.mlp_reconstruct_dim=[32,16]

        mlp_input_dim=np.sum(self.feature_dim)+pos_dim*2*len(self.feature_dim)

        self.mlp_student_decoder=MLP(mlp_input_dim,self.mlp_decoder_dim[-1],hidden_list=self.mlp_decoder_dim[:-1])
        self.mlp_student_seg=MLP(self.mlp_decoder_dim[-1],args.n_labels,hidden_list=self.mlp_seg_dim)
        self.mlp_student_reconstruct=MLP(self.mlp_decoder_dim[-1],1,hidden_list=self.mlp_reconstruct_dim)

        self.mlp_teacher_decoder=MLP(mlp_input_dim,self.mlp_decoder_dim[-1],hidden_list=self.mlp_decoder_dim[:-1])
        self.mlp_teacher_seg=MLP(self.mlp_decoder_dim[-1],args.n_labels,hidden_list=self.mlp_seg_dim)


    def create_norm_layer(self, planes, norm_groups=8,error_on_non_divisible_norm_groups=False):
        if planes < norm_groups:
            return self.norm_layer(planes, planes)
        elif not error_on_non_divisible_norm_groups and (planes % norm_groups) > 0:
            print("Setting number of norm groups to {} for this convolution block.".format(planes))
            return self.norm_layer(planes, planes)
        else:
            return self.norm_layer(norm_groups, planes)


    def feature_sample(self,feat, coord_all,coords_patch):
        # feat: [B, C, h, w, z]
        # coord: [B, N, 3], N <= H * W,是归一化的
        b, c, h, w, z = feat.shape  # lr
        B, N, _ = coord_all.shape

        coord_embed=self.pos_embed(coord_all)#编码的坐标

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
                    coord_ = coords_patch.clone()
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

                    rel_coord = coords_patch - q_coord  # 真实坐标和偏移的采样坐标之间的偏差
                    rel_coord=self.pos_embed(rel_coord)
                    # rel_coord[:, :, 0] *= h
                    # rel_coord[:, :, 1] *= w
                    # rel_coord[:, :, 2] *= z

                    inp = torch.cat([q_feat, coord_embed, rel_coord], dim=-1)  # [B,N,c+3],特征+相对坐标
                    preds.append(inp)

        preds = torch.stack(preds, dim=2)  # [B, N, k*k*k=8, c+3]

        b, n, num,c = preds.shape

        preds=preds.view(b*n,num,c)

        return preds

    def forward(self, x, coords_all,coords_patch=None, teacher_need=False  , is_Test=False):

        if is_Test:
            a=time.time()
            affine=torch.tensor([[i[2].start,i[3].start,i[4].start] for i in coords_all])
            coords_patch_int=np.where(np.ones((128,128,64))>0)
            coords_patch_int=torch.stack([torch.tensor(coords_patch_int[0]),torch.tensor(coords_patch_int[1]),torch.tensor(coords_patch_int[2])],dim=-1)#1048576,3
            coords_patch_=torch.stack([coords_patch_int for i in range(x.shape[0])],dim=0)
            coords_all_=torch.stack([coords_patch_int for i in range(x.shape[0])],dim=0)


            for i in range(x.shape[0]):
                coords_all_[i,:, 0]+= affine[i,0]
                coords_all_[i,:, 1] += affine[i,1]
                coords_all_[i,:, 2] += affine[i,2]


            B, N, _ = coords_patch_.shape
            B=x.shape[0]

            coords_all=coords_all_.to(torch.float32)
            coords_patch=coords_patch_.to(torch.float32)

            for i in range(B):
                coords_all[i] = self.coords_all_[coords_all_[i, :, 0],
                                      coords_all_[i, :, 1], coords_all_[i, :, 2], :]
                coords_patch[i] = self.coords_patch_[coords_patch_[i, :, 0],
                             coords_patch_[i, :, 1], coords_patch_[i, :, 2], :]

            coords_patch=coords_patch.to(x.device)
            coords_all=coords_all.to(x.device)

            x = self.backbone.encoder(x)
            unet_out = self.backbone.final_convolution(self.backbone.decoder(x))
            return unet_out
            # x:[layer4_feature(256),layer3_feature(128),layer2_feature(64),layer1_feature(32)]
            n = 50000
            tmp = []
            for start in range(0, N, n):
                end = min(N, start + n)
                sample_feature = []
                for step, i in enumerate(x):
                    # i:batch*feature_dim*h*w*z
                    i = self.norm[step](i)
                    layer_feature = self.feature_sample(i,coords_all[:,start:end,:], coords_patch[:,start:end,:])
                    layer_feature = getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature)
                    sample_feature.append(layer_feature)

                sample_feature = torch.concat(sample_feature, dim=-1)  # B*N*8,(c1+c2+c3+c4)
                # sample_feature=self.batchnorm1d(sample_feature)
                sample_feature = sample_feature.reshape((B, end-start, -1))  # B,N,c

                # sample_feature=torch.rand_like(sample_feature,dtype=torch.float32,device=sample_feature.device)
                mlp_decoder_feature = self.mlp_student_decoder(sample_feature)
                mlp_decoder_feature = torch.relu(mlp_decoder_feature)
                student_class = self.mlp_student_seg(
                    mlp_decoder_feature)  # 这里是标记的像素的输出，直接计算ce损失,B

                tmp.append(student_class)
            tmp = torch.cat(tmp, dim=1).permute(0, 2, 1).contiguous()
            img_size = self.args.patch_size[-3:]
            res = tmp.reshape(B, -1, img_size[0], img_size[1] ,img_size[2])
            print(time.time()-a)
            return res

        B,N,_=coords_all.shape

        x = self.backbone.encoder(x)
        unet_out = self.backbone.final_convolution(self.backbone.decoder(x))
        # x:[layer4_feature(256),layer3_feature(128),layer2_feature(64),layer1_feature(32)]


        sample_feature = []
        for step, i in enumerate(x):
            # i:batch*feature_dim*h*w*z
            i=self.norm[step](i)
            layer_feature = self.feature_sample(i, coords_all,coords_patch)
            layer_feature = getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature)
            sample_feature.append(layer_feature)
            # sample_feature:4,B*N,1,c

        sample_feature=torch.concat(sample_feature,dim=-1)#B*N*8,(c1+c2+c3+c4)
        #sample_feature=self.batchnorm1d(sample_feature)
        sample_feature=sample_feature.reshape((B,N,-1))#B,N,c

        #sample_feature=torch.rand_like(sample_feature,dtype=torch.float32,device=sample_feature.device)
        mlp_decoder_feature = self.mlp_student_decoder(sample_feature)
        mlp_decoder_feature = torch.relu(mlp_decoder_feature)
        student_class = self.mlp_student_seg(mlp_decoder_feature[:,:-self.args.sample_num,:])#这里是标记的像素的输出，直接计算ce损失
        student_reconstruct = self.mlp_student_reconstruct(mlp_decoder_feature)#用于计算重建mse损失，包含了有标签坐标和没标签坐标，正常输入


        student_aug_class,teacher_class=None,None

        #半监督部分
        if teacher_need:
        #区别1：数据增强
            sample_feature_aug = []
            if random.uniform(0, 1) < 0.5:
                for step, i in enumerate(x):
                    # i:batch*feature_dim*h*w*z
                    i = self.norm[step](i)
                    layer_feature = self.feature_sample(i, coords_all,coords_patch)
                    sample_feature_aug.append(getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature,True) )#数据增强，注意这里标注和没标注还在一起
                sample_feature_aug = torch.concat(sample_feature_aug, dim=-1)  # B*N*8,(c1+c2+c3+c4)
                sample_feature_aug = sample_feature_aug.reshape((B, N, -1))

            else:
                sample_feature_aug=sample_feature

        #区别2：student模型输出未标记像素增强特征的预测值
            sample_feature_aug=self.mlp_teacher_decoder(sample_feature_aug[:,-self.args.sample_num:,:])#未标记像素增强后的输出，用于计算一致性损失
            student_aug_class=self.mlp_student_seg(sample_feature_aug)#未标记像素增强后的输出，用于计算一致性损失

        # 区别3：teacher模型输出为标记像素原特征伪标签
            with torch.no_grad():
                sample_feature = self.mlp_teacher_decoder(sample_feature[:,-self.args.sample_num:,:])
                teacher_class=self.mlp_teacher_seg(sample_feature)#未标记像素原始特征的输出，计算一致性损失



        return unet_out,student_class,student_reconstruct,student_aug_class,teacher_class


