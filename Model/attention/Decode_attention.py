import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=3, head_dim=None, qkv_bias=True, attn_drop=0., proj_drop=0.):
        '''
        主要问题就是无法解决不同层之间的差距
        :param dim:特征维度
        :param num_heads:
        :param window_size:
        :param head_dim:
        :param qkv_bias:
        :param attn_drop:
        :param proj_drop:
        '''
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)#缩放因子，nn.parameter是为了把他参数化

        self.guodu_ = nn.Linear(dim,dim)


        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_norm=torch.nn.BatchNorm1d(dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x , gauss=False):
        """
        Args:
            x: [B*Num_sample, feature_num(8), D]
        Returns:
            x: [B*Num_sample,D]代表某一个特征图的新点特征
        """

        x = self.guodu_(x)

        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1,4)  # [kv(2), B_, nheads, N, dim_per_head]
        q, k, v = qkv.unbind(0)  # [B_, nheads, N, dim_per_head]

        # scaled cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # [B*wh*ww, nheads, 1, WH*WW]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        attn = self.softmax(attn)  # B,num_headm,1,N

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, 9, -1)  # [B*wh*ww, WH*WW, nheads*dim_per_head]=[B*wh*ww, WH*WW, D]
        x = self.proj(x)
        x = x.reshape(-1,C)
        x=self.proj_norm(x.squeeze())
        x=x.reshape(B_, -1, C)
        x = self.proj_drop(x)
        return x





class Decoder_Attention(nn.Module):
    def __init__(self, dim, h_dim, num_heads=3, head_dim=None, qkv_bias=True, attn_drop=0., proj_drop=0.):
        '''
        主要问题就是无法解决不同层之间的差距
        :param dim:特征维度
        :param num_heads:
        :param window_size:
        :param head_dim:
        :param qkv_bias:
        :param attn_drop:
        :param proj_drop:
        '''
        super().__init__()

        self.latent_h=nn.Linear(1, h_dim)#第一层的h
        self.h_dim_2_dim = nn.Linear(dim,dim)


        self.first_layer = Self_Attention(dim=dim, head_dim=head_dim)


        self.dim = dim
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)#缩放因子，nn.parameter是为了把他参数化

        self.guodu_ = nn.Linear(dim,dim)



        self.q=nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, attn_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_norm=torch.nn.BatchNorm1d(dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.fc_mu = nn.Sequential(nn.Linear(dim, dim//2),
                                    nn.Linear(dim//2,1))
        self.fc_var = nn.Sequential(nn.Linear(dim, dim//2),
                                    nn.Linear(dim//2,1))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x , coords, gauss=False):
        """
        Args:
            x: [B*Num_sample, feature_num(8), D]
        Returns:
            x: [B*Num_sample,D]代表某一个特征图的新点特征
        """

        x = self.guodu_(x)#x=邻居特征+xz-x+xz

        B_, N, C = x.shape

        One = torch.ones(B_, 1,1).to(x.device)
        h=self.latent_h(One)

        coords = coords.permute(1,0,2) # (N, M, C)
        h = torch.cat([coords, h], dim=2)  # h=h+x
        h=self.h_dim_2_dim(h)

        h_x = torch.cat([h,x],dim=1)#B,9,C,

        h_x = self.first_layer(h_x)

        h = h_x[:, 0, :]
        x = h_x[:, 1:, :]
        q=self.q(h)
        q=q.reshape(B_, 1, 1, self.num_heads, -1)
        q=q.permute(2, 0, 3, 1,4)
        q=q.unbind(0)[0]#[B,num_head,1(N),dim],unbind是按照某一维度划分，这一维度就消失了

        kv = self.kv(x).reshape(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1,4)  # [kv(2), B_, nheads, N, dim_per_head]
        k, v = kv.unbind(0)  # [B_, nheads, N, dim_per_head]


        if gauss ==False:
            # scaled cosine attention
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # [B*wh*ww, nheads, 1, WH*WW]
            logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
            attn = attn * logit_scale
            attn = self.softmax(attn)  # B,num_headm,1,N



        else:
            attn_label = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # [B*wh*ww, nheads, 1, WH*WW]
            logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
            attn_label = attn_label * logit_scale
            attn_label = self.softmax(attn_label)  # B,num_headm,1,N

            q_u = self.fc_mu(F.normalize(q,dim=-1))
            k_var = self.fc_var(F.normalize(k,dim=-1))
            q_u = q_u.squeeze(dim=-2)
            sigma = []
            for i in range(N):
                sigma.append(self.reparameterize(q_u, k_var[:, :, i, :]))
            sigma = torch.stack(sigma, dim=-1)
            attn = torch.softmax(sigma, dim=-1)  # 2000*7


        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, 1, -1)  # [B*wh*ww, WH*WW, nheads*dim_per_head]=[B*wh*ww, WH*WW, D]
        x = self.proj(x)
        x=self.proj_norm(x.squeeze())
        x=x.unsqueeze(dim=1)
        x = self.proj_drop(x)
        return x

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        logvar=torch.sigmoid(logvar)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std + mu