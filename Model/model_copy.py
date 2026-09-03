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


def normalize_coords(coords, spatial_size):
    """Convert voxel coordinates to [-1, 1] without allocating a full grid."""
    coords = coords.to(torch.float32)
    scale = coords.new_tensor(spatial_size) - 1
    return coords * (2.0 / scale) - 1.0



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
            num_class=4,
            L=5,#坐标嵌入的维度
            mlp_decoder_hidden_dim=[512, 256, 128, 64]
    ):

        super(DPT, self).__init__()
        self.args=args
        self.num_class = num_class

        # 编码部分
        self.backbone=UNet(n_features=init_dim,base_width=features,n_outputs=num_class,encoder_blocks=layer_blocks)
        self.feature_dim=[32,64,128,256]
        self.layer_blocks = layer_blocks

        #采样部分（特征提取）
        self.pos_embed,pos_dim=get_embedder(L)
        self.layer4_uni=Decoder_Attention(self.feature_dim[0]+pos_dim*2,h_dim=self.feature_dim[0]+pos_dim ,head_dim=self.feature_dim[0]+pos_dim*2)
        self.layer3_uni = Decoder_Attention(self.feature_dim[1] + pos_dim * 2,h_dim=self.feature_dim[1]+pos_dim ,head_dim=self.feature_dim[1]+pos_dim*2)
        self.layer2_uni = Decoder_Attention(self.feature_dim[2] + pos_dim * 2,h_dim=self.feature_dim[2]+pos_dim ,head_dim=self.feature_dim[2]+pos_dim*2)
        self.layer1_uni = Decoder_Attention(self.feature_dim[3] + pos_dim * 2,h_dim=self.feature_dim[3]+pos_dim ,head_dim=self.feature_dim[3]+pos_dim*2)

        # self.layer4_gauss_uni = Gauss_Attention(self.feature_dim[0]+pos_dim*2,head_dim=self.feature_dim[0]+pos_dim*2)
        # self.layer3_gauss_uni = Gauss_Attention(self.feature_dim[1] + pos_dim * 2,head_dim=self.feature_dim[1]+pos_dim*2)
        # self.layer2_gauss_uni = Gauss_Attention(self.feature_dim[2] + pos_dim * 2,head_dim=self.feature_dim[2]+pos_dim*2)
        # self.layer1_gauss_uni = Gauss_Attention(self.feature_dim[3] + pos_dim * 2,head_dim=self.feature_dim[3]+pos_dim*2)



        #解码部分
        self.mlp_decoder_dim=mlp_decoder_hidden_dim
        self.mlp_seg_dim=[32,16]
        self.mlp_reconstruct_dim=[32,16]

        mlp_input_dim=np.sum(self.feature_dim)+pos_dim*2*len(self.feature_dim)

        #self.mlp_student=MLP(mlp_input_dim,num_class,hidden_list=self.mlp_decoder_dim)
        self.mlp_student_decoder=MLP(mlp_input_dim,self.mlp_decoder_dim[-1],hidden_list=self.mlp_decoder_dim[:-1])
        self.mlp_student_seg=MLP(self.mlp_decoder_dim[-1],num_class,hidden_list=self.mlp_seg_dim)
        self.mlp_student_reconstruct=MLP(self.mlp_decoder_dim[-1],1,hidden_list=self.mlp_reconstruct_dim)

        self.mlp_teacher_decoder=MLP(mlp_input_dim,self.mlp_decoder_dim[-1],hidden_list=self.mlp_decoder_dim[:-1])
        self.mlp_teacher_seg=MLP(self.mlp_decoder_dim[-1],num_class,hidden_list=self.mlp_seg_dim)

        self.spatial_size = (512, 512, 256)




    def feature_sample(self,feat, coord_patch, center, patch_size):
        # feat: [B, C, h, w, z]
        # coord_patch: [B, N, 3], N <= H * W,是归一化的,原始的位置
        # center: B,3
        b, c, h, w, z = feat.shape  # lr
        B, N, _ = coord_patch.shape

        H, W, Z = self.spatial_size

        feat_coord = make_coord((h, w, z), flatten=False).to(feat.device).permute(3, 0, 1, 2).contiguous().unsqueeze(
            0).expand(b, 3, h, w, z)  # batch * 3 * h * w * z,归一化坐标


        rx = 1 / (h-1)
        ry = 1 / (w-1)
        rz = 1 / (z-1)

        preds = []

        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                for vz in [-1, 1]:
                    coord_ = coord_patch.clone()
                    coord_[:, :, 0] += (vx) * rx
                    coord_[:, :, 1] += (vy) * ry
                    coord_[:, :, 2] += (vz) * rz

                    coord_[coord_<-1] = -1
                    coord_[coord_>1] = 1

                    #变换到原始位置
                    coord_ += 1
                    coord_ /= 2
                    coord_[:,:,0]*=(h-1)
                    coord_[:,:,1]*=(w-1)
                    coord_[:,:,2]*=(z-1)

                    coord_ = torch.round(coord_).to(torch.long)#B,N,3


                    q_feat = []
                    q_coord = []
                    for i in range(B):
                        q_feat.append(feat[i, :, coord_[i,:,0], coord_[i,:,1], coord_[i,:,2]] ) # (B, N, C)
                        q_coord.append(feat_coord[i, :, coord_[i,:,0], coord_[i,:,1], coord_[i,:,2]])
                    q_feat = torch.stack(q_feat,dim=0).permute(0,2,1)
                    q_coord = torch.stack(q_coord,dim=0).permute(0,2,1)

                    # k += 1
                    # # feat: [B, c, h, w], coord_: [B, N, 3] --> [B, 1, 1， N, 3], out: [B, c, 1, 1, N] --> [B, c, N] --> [B, N, c]
                    # #flip把坐标从[x,y,z]->[z,y,x]
                    # q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                    #          :, 0, 0, :].permute(0, 2, 1).contiguous()  # [B, N, c]
                    # q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1), mode='nearest',
                    #                         align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1).contiguous()  # [B, N, 3]

                    try:
                        rel_coord = q_coord - coord_patch  # 真实坐标和偏移的采样坐标之间的偏差
                        rel_coord=self.pos_embed(rel_coord)

                    except:
                        None

                    #恢复邻居点真实的坐标
                    q_coord[:,:,0] = (q_coord[:,:,0]+1)*(patch_size[0]-1)/2
                    q_coord[:,:,1] = (q_coord[:,:,1]+1)*(patch_size[1]-1)/2
                    q_coord[:,:,2] = (q_coord[:,:,2]+1)*(patch_size[2]-1)/2

                    expanded_center = center.unsqueeze(1).expand(B, N, 3)
                    q_coord += expanded_center
                    q_coord = torch.round(q_coord).to(torch.long)
                    q_coord[:,:,0][q_coord[:,:,0]>=H] = H-1
                    q_coord[:,:,1][q_coord[:,:,1]>=W] = W-1
                    q_coord[:,:,2][q_coord[:,:,2]>=Z] = Z-1
                    q_coord[:, :, 0][q_coord[:, :, 0] < 0] = 0
                    q_coord[:,:,1][q_coord[:,:,1] < 0] = 0
                    q_coord[:,:,2][q_coord[:,:,2] < 0] = 0

                    q_coord = normalize_coords(q_coord, self.spatial_size).to(feat.device)

                    q_coord = self.pos_embed(q_coord)
                    # rel_coord[:, :, 0] *= h
                    # rel_coord[:, :, 1] *= w
                    # rel_coord[:, :, 2] *= z

                    inp = torch.cat([q_feat, rel_coord, q_coord], dim=-1)  # [B,N,c+3],特征+相对坐标
                    preds.append(inp)

        preds = torch.stack(preds, dim=2)  # [B, N, k*k*k=8, c+3]

        b, n, num,c = preds.shape

        preds=preds.view(b*n,num,c)

        return preds
    # def forward_test(self, x, coords_all):
    #     B,N,_=coords_all.shape
    #
    #     x = self.backbone.encoder(x)
    #     unet_out = self.backbone.final_convolution(self.backbone.decoder(x))
    #     return unet_out,unet_out
    #     # x:[layer4_feature(256),layer3_feature(128),layer2_feature(64),layer1_feature(32)]
    #
    #     split_num = 512
    #     cut_length = N // split_num
    #     result = []
    #     for item in range(split_num):
    #         print(item ,' / ', split_num)
    #         coords = coords_all[:, cut_length * item:cut_length * (item + 1), :]
    #
    #         sample_feature = []
    #         for step, i in enumerate(x):
    #             # i:batch*feature_dim*h*w*z
    #             layer_feature = self.feature_sample(i, coords)
    #             layer_feature = getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature)
    #             sample_feature.append(layer_feature)
    #             # sample_feature:4,B*N,1,c
    #
    #         sample_feature=torch.concat(sample_feature,dim=-1)#B*N*8,(c1+c2+c3+c4)
    #         sample_feature=sample_feature.reshape((B,cut_length,-1))#B,N,c
    #
    #         mlp_decoder_feature = self.mlp_student_decoder(sample_feature)
    #         mlp_decoder_feature = torch.relu(mlp_decoder_feature)
    #         student_class = self.mlp_student_seg(mlp_decoder_feature)
    #         result.append(student_class)# 这里是标记的像素的输出，直接计算ce损失
    #
    #     result = torch.concat(result,dim=1)
    #     result=result.view(B,512,512,256,self.num_class)#B,N,num_classes
    #     result = result.permute(0,4,1,2,3)
    #
    #     return unet_out,result

    def forward(self, x, coords_patch, center, patch_size = (128,128,64), teacher_need=False):

        #coords_patch是归一化的

        B, N, _ = coords_patch.shape

        x = self.backbone.encoder(x)
        unet_out = self.backbone.final_convolution(self.backbone.decoder(x))
        # x:[layer4_feature(256),layer3_feature(128),layer2_feature(64),layer1_feature(32)]

        sample_feature = []

        coords_patch_clong = coords_patch.clone()

        #回复coords_patch的全局坐标
        coords_patch[coords_patch < -1] = -1
        coords_patch[coords_patch > 1] = 1

        # 恢复采样点的全局坐标
        coords_patch[:, :, 0] = (coords_patch[:, :, 0] + 1) * (patch_size[0]-1) / 2
        coords_patch[:, :, 1] = (coords_patch[:, :, 1] + 1) * (patch_size[1] - 1) / 2
        coords_patch[:, :, 2] = (coords_patch[:, :, 2] + 1) * (patch_size[2] - 1) / 2

        expanded_center = center.unsqueeze(1).expand(B, N, 3)
        coords_patch += expanded_center
        coords_patch = torch.round(coords_patch).to(torch.long)

        H, W, Z = self.spatial_size
        coords_patch[coords_patch[:, :, 0] < 0] = 0
        coords_patch[coords_patch[:, :, 0] >= H] = H - 1
        coords_patch[coords_patch[:, :, 1] < 0] = 0
        coords_patch[coords_patch[:, :, 1] >= W] = W - 1
        coords_patch[coords_patch[:, :, 2] < 0] = 0
        coords_patch[coords_patch[:, :, 2] >= Z] = Z - 1

        coords_patch = normalize_coords(coords_patch, self.spatial_size)

        coords_patch = self.pos_embed(coords_patch).to(center.device)




        for step, i in enumerate(x):
            # i:batch*feature_dim*h*w*z
            layer_feature = self.feature_sample(i, coords_patch_clong, center, patch_size)
            layer_feature = getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature, coords_patch)
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
            num_class=4,
            L=5,#坐标嵌入的维度
            mlp_decoder_hidden_dim=[512, 256, 128, 64]
    ):

        super(DPT, self).__init__()
        self.args=args
        self.num_class = num_class

        # 编码部分
        self.backbone=UNet(n_features=init_dim,base_width=features,n_outputs=num_class,encoder_blocks=layer_blocks)
        self.feature_dim=[32,64,128,256]
        self.layer_blocks = layer_blocks

        #采样部分（特征提取）
        self.pos_embed,pos_dim=get_embedder(L)
        self.layer4_uni=Decoder_Attention(self.feature_dim[0]+pos_dim*2,h_dim=self.feature_dim[0]+pos_dim ,head_dim=self.feature_dim[0]+pos_dim*2)
        self.layer3_uni = Decoder_Attention(self.feature_dim[1] + pos_dim * 2,h_dim=self.feature_dim[1]+pos_dim ,head_dim=self.feature_dim[1]+pos_dim*2)
        self.layer2_uni = Decoder_Attention(self.feature_dim[2] + pos_dim * 2,h_dim=self.feature_dim[2]+pos_dim ,head_dim=self.feature_dim[2]+pos_dim*2)
        self.layer1_uni = Decoder_Attention(self.feature_dim[3] + pos_dim * 2,h_dim=self.feature_dim[3]+pos_dim ,head_dim=self.feature_dim[3]+pos_dim*2)

        # self.layer4_gauss_uni = Gauss_Attention(self.feature_dim[0]+pos_dim*2,head_dim=self.feature_dim[0]+pos_dim*2)
        # self.layer3_gauss_uni = Gauss_Attention(self.feature_dim[1] + pos_dim * 2,head_dim=self.feature_dim[1]+pos_dim*2)
        # self.layer2_gauss_uni = Gauss_Attention(self.feature_dim[2] + pos_dim * 2,head_dim=self.feature_dim[2]+pos_dim*2)
        # self.layer1_gauss_uni = Gauss_Attention(self.feature_dim[3] + pos_dim * 2,head_dim=self.feature_dim[3]+pos_dim*2)



        #解码部分
        self.mlp_decoder_dim=mlp_decoder_hidden_dim
        self.mlp_seg_dim=[32,16]
        self.mlp_reconstruct_dim=[32,16]

        mlp_input_dim=np.sum(self.feature_dim)+pos_dim*2*len(self.feature_dim)

        #self.mlp_student=MLP(mlp_input_dim,num_class,hidden_list=self.mlp_decoder_dim)
        self.mlp_student_decoder=MLP(mlp_input_dim,self.mlp_decoder_dim[-1],hidden_list=self.mlp_decoder_dim[:-1])
        self.mlp_student_seg=MLP(self.mlp_decoder_dim[-1],num_class,hidden_list=self.mlp_seg_dim)
        self.mlp_student_reconstruct=MLP(self.mlp_decoder_dim[-1],1,hidden_list=self.mlp_reconstruct_dim)

        self.mlp_teacher_decoder=MLP(mlp_input_dim,self.mlp_decoder_dim[-1],hidden_list=self.mlp_decoder_dim[:-1])
        self.mlp_teacher_seg=MLP(self.mlp_decoder_dim[-1],num_class,hidden_list=self.mlp_seg_dim)

        self.spatial_size = (512, 512, 256)




    def feature_sample(self,feat, coord_patch, center, patch_size):
        # feat: [B, C, h, w, z]
        # coord_patch: [B, N, 3], N <= H * W,是归一化的,原始的位置
        # center: B,3
        b, c, h, w, z = feat.shape  # lr
        B, N, _ = coord_patch.shape

        H, W, Z = self.spatial_size

        feat_coord = make_coord((h, w, z), flatten=False).to(feat.device).permute(3, 0, 1, 2).contiguous().unsqueeze(
            0).expand(b, 3, h, w, z)  # batch * 3 * h * w * z,归一化坐标


        rx = 1 / (h-1)
        ry = 1 / (w-1)
        rz = 1 / (z-1)

        preds = []

        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                for vz in [-1, 1]:
                    coord_ = coord_patch.clone()
                    coord_[:, :, 0] += (vx) * rx
                    coord_[:, :, 1] += (vy) * ry
                    coord_[:, :, 2] += (vz) * rz

                    coord_[coord_<-1] = -1
                    coord_[coord_>1] = 1

                    #变换到原始位置
                    coord_ += 1
                    coord_ /= 2
                    coord_[:,:,0]*=(h-1)
                    coord_[:,:,1]*=(w-1)
                    coord_[:,:,2]*=(z-1)

                    coord_ = torch.round(coord_).to(torch.long)#B,N,3


                    q_feat = []
                    q_coord = []
                    for i in range(B):
                        q_feat.append(feat[i, :, coord_[i,:,0], coord_[i,:,1], coord_[i,:,2]] ) # (B, N, C)
                        q_coord.append(feat_coord[i, :, coord_[i,:,0], coord_[i,:,1], coord_[i,:,2]])
                    q_feat = torch.stack(q_feat,dim=0).permute(0,2,1)
                    q_coord = torch.stack(q_coord,dim=0).permute(0,2,1)

                    # k += 1
                    # # feat: [B, c, h, w], coord_: [B, N, 3] --> [B, 1, 1， N, 3], out: [B, c, 1, 1, N] --> [B, c, N] --> [B, N, c]
                    # #flip把坐标从[x,y,z]->[z,y,x]
                    # q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1).unsqueeze(1), mode='nearest', align_corners=False)[:,
                    #          :, 0, 0, :].permute(0, 2, 1).contiguous()  # [B, N, c]
                    # q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1).unsqueeze(1), mode='nearest',
                    #                         align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1).contiguous()  # [B, N, 3]

                    try:
                        rel_coord = q_coord - coord_patch  # 真实坐标和偏移的采样坐标之间的偏差
                        rel_coord=self.pos_embed(rel_coord)

                    except:
                        None

                    #恢复邻居点真实的坐标
                    q_coord = coord_.clone().to(torch.float32) #
                    q_coord[:,:,0] = q_coord[:,:,0]*patch_size[0]/h - patch_size[0]/2
                    q_coord[:,:,1] = q_coord[:,:,1]*patch_size[1]/w - patch_size[1]/2
                    q_coord[:,:,2] = q_coord[:,:,2]*patch_size[2]/z - patch_size[2]/2

                    # q_coord[:,:,0] = (q_coord[:,:,0]+1)*(patch_size[0]-1)/2 - patch_size[0]/2
                    # q_coord[:,:,1] = (q_coord[:,:,1]+1)*(patch_size[1]-1)/2 - patch_size[1]/2
                    # q_coord[:,:,2] = (q_coord[:,:,2]+1)*(patch_size[2]-1)/2 - patch_size[2]/2

                    expanded_center = center.unsqueeze(1).expand(B, N, 3)
                    q_coord += expanded_center
                    #q_coord = torch.round(q_coord).to(torch.long)
                    q_coord[:,:,0][q_coord[:,:,0]>=H] = H-1
                    q_coord[:,:,1][q_coord[:,:,1]>=W] = W-1
                    q_coord[:,:,2][q_coord[:,:,2]>=Z] = Z-1
                    q_coord[:, :, 0][q_coord[:, :, 0] < 0] = 0
                    q_coord[:,:,1][q_coord[:,:,1] < 0] = 0
                    q_coord[:,:,2][q_coord[:,:,2] < 0] = 0

                    q_coord = normalize_coords(q_coord, self.spatial_size).to(feat.device)

                    q_coord = self.pos_embed(q_coord)
                    # rel_coord[:, :, 0] *= h
                    # rel_coord[:, :, 1] *= w
                    # rel_coord[:, :, 2] *= z

                    inp = torch.cat([q_feat, rel_coord, q_coord], dim=-1)  # [B,N,c+3],特征+相对坐标
                    preds.append(inp)

        preds = torch.stack(preds, dim=2)  # [B, N, k*k*k=8, c+3]

        b, n, num,c = preds.shape

        preds=preds.view(b*n,num,c)

        return preds
    def forward_test(self, x, coords_patch, patch_size):
        B,N,_=coords_patch.shape

        # x = self.backbone.encoder(x)
        # unet_out = self.backbone.final_convolution(self.backbone.decoder(x))
        # return unet_out,unet_out
        # # x:[layer4_feature(256),layer3_feature(128),layer2_feature(64),layer1_feature(32)]


        mlp_result = torch.zeros((1,4,512,512,256))
        unet_result = torch.zeros((1,4,512,512,256))

        for x_step in range(512//128):
            for y_step in range(512//128):
                for z_step in range(256//64):
                    print(x_step,y_step,z_step)
                    x_patch = x[:,:,x_step*128:(x_step+1)*128,y_step*128:(y_step+1)*128,z_step*64:(z_step+1)*64]


                    # unet_result[:,:,x_step*128:(x_step+1)*128,y_step*128:(y_step+1)*128,z_step*64:(z_step+1)*64] = self.forward(x_patch,None,None,None)
                    # continue



                    center = torch.tensor((64+128*x_step,64+128*y_step,32+64*z_step)).unsqueeze(dim=0).to(x_patch.device)

                    all_mlp = []
                    step_xx = 128 * 128 * 64 // 21
                    for i in range(20):
                        _, mlp_temp,_ = self.forward(x_patch,coords_patch[:,step_xx*i:step_xx*(i+1)],center,patch_size)
                        all_mlp.append(mlp_temp)

                    _, mlp_tempend,_ = self.forward(x_patch, coords_patch[:,step_xx*20:,:], center, patch_size)
                    all_mlp.append(mlp_tempend)
                    all_mlp = torch.concat(all_mlp,dim=1)
                    all_mlp = all_mlp.reshape((1,128,128,64,4)).permute(0,4,1,2,3)
                    mlp_result[:,:,x_step*128:(x_step+1)*128,y_step*128:(y_step+1)*128,z_step*64:(z_step+1)*64] = all_mlp

        return unet_result,mlp_result


                    # mlp_result[0,:,x_step*128:(x_step+1)*128,y_step*128:(y_step+1)*128,z_step*64:(z_step+1)*64] = mlp_temp1.reshape(patch_size)




        # split_num = 512
        # cut_length = N // split_num
        # result = []
        # for item in range(split_num):
        #     print(item ,' / ', split_num)
        #     coords = coords_all[:, cut_length * item:cut_length * (item + 1), :]
        #
        #     sample_feature = []
        #     for step, i in enumerate(x):
        #         # i:batch*feature_dim*h*w*z
        #         layer_feature = self.feature_sample(i, coords)
        #         layer_feature = getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature)
        #         sample_feature.append(layer_feature)
        #         # sample_feature:4,B*N,1,c
        #
        #     sample_feature=torch.concat(sample_feature,dim=-1)#B*N*8,(c1+c2+c3+c4)
        #     sample_feature=sample_feature.reshape((B,cut_length,-1))#B,N,c
        #
        #     mlp_decoder_feature = self.mlp_student_decoder(sample_feature)
        #     mlp_decoder_feature = torch.relu(mlp_decoder_feature)
        #     student_class = self.mlp_student_seg(mlp_decoder_feature)
        #     result.append(student_class)# 这里是标记的像素的输出，直接计算ce损失
        #
        # result = torch.concat(result,dim=1)
        # result=result.view(B,512,512,256,self.num_class)#B,N,num_classes
        # result = result.permute(0,4,1,2,3)
        #
        # return unet_out,result

    def forward(self, x, coords_patch, center, patch_size = (128,128,64), teacher_need=False):

        #coords_patch是归一化的


        x = self.backbone.encoder(x)
        unet_out = self.backbone.final_convolution(self.backbone.decoder(x))
        # return unet_out
        # x:[layer4_feature(256),layer3_feature(128),layer2_feature(64),layer1_feature(32)]

        B, N, _ = coords_patch.shape
        sample_feature = []

        coords_patch_clong = coords_patch.clone()

        #回复coords_patch的全局坐标
        coords_patch[coords_patch < -1] = -1
        coords_patch[coords_patch > 1] = 1

        # 恢复采样点的全局坐标
        coords_patch[:, :, 0] = (coords_patch[:, :, 0] + 1) * (patch_size[0]-1) / 2 - patch_size[0] / 2
        coords_patch[:, :, 1] = (coords_patch[:, :, 1] + 1) * (patch_size[1] - 1) / 2 - patch_size[1] / 2
        coords_patch[:, :, 2] = (coords_patch[:, :, 2] + 1) * (patch_size[2] - 1) / 2 - patch_size[2] / 2


        expanded_center = center.unsqueeze(1).expand(B, N, 3)
        coords_patch += expanded_center
        coords_patch = torch.round(coords_patch).to(torch.long)

        H, W, Z = self.spatial_size
        coords_patch[coords_patch[:, :, 0] < 0] = 0
        coords_patch[coords_patch[:, :, 0] >= H] = H - 1
        coords_patch[coords_patch[:, :, 1] < 0] = 0
        coords_patch[coords_patch[:, :, 1] >= W] = W - 1
        coords_patch[coords_patch[:, :, 2] < 0] = 0
        coords_patch[coords_patch[:, :, 2] >= Z] = Z - 1

        coords_patch = normalize_coords(coords_patch, self.spatial_size)

        coords_patch = self.pos_embed(coords_patch).to(center.device)




        for step, i in enumerate(x):
            # i:batch*feature_dim*h*w*z
            layer_feature = self.feature_sample(i, coords_patch_clong, center, patch_size)
            layer_feature = getattr(self, 'layer{}_uni'.format(step + 1))(layer_feature, coords_patch)
            sample_feature.append(layer_feature)
            # sample_feature:4,B*N,1,c

        sample_feature=torch.concat(sample_feature,dim=-1)#B*N*8,(c1+c2+c3+c4)
        sample_feature=sample_feature.reshape((B,N,-1))#B,N,c

        #sample_feature=torch.rand_like(sample_feature,dtype=torch.float32,device=sample_feature.device)
        # mlp_out = self.mlp_student(sample_feature)
        mlp_decoder_feature = self.mlp_student_decoder(sample_feature)

        #return mlp_decoder_feature, mlp_decoder_feature, mlp_decoder_feature

        mlp_decoder_feature = torch.relu(mlp_decoder_feature)
        #student_class = self.mlp_student_seg(mlp_decoder_feature[:, :-self.args.sample_num, :])  # 这里是标记的像素的输出，直接计算ce损失

        student_class = self.mlp_student_seg(mlp_decoder_feature)
        student_reconstruct = self.mlp_student_reconstruct(mlp_decoder_feature)  # 用于计算重建mse损失，包含了有标签坐标和没标签坐标，正常输入


        return unet_out,student_class,student_reconstruct


