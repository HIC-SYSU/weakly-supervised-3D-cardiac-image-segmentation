import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class Gauss_Attention(nn.Module):
    def __init__(self, dim, num_heads=3, head_dim=None, qkv_bias=True, attn_drop=0., proj_drop=0.,latent_dim=1):
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

        self.latent_h=nn.Linear(1, dim)#第一层的h

        self.dim = dim
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)#缩放因子，nn.parameter是为了把他参数化


        self.q=nn.Linear(dim, attn_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, attn_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # gauss sample
        self.fc_mu = nn.Linear(dim, latent_dim)
        self.fc_var = nn.Linear(dim, latent_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.norm=torch.nn.InstanceNorm1d(attn_dim)


    def forward(self, x):
        """
        Args:
            x: [B*Num_sample, feature_num(8), D]
        Returns:
            x: [B*Num_sample,D]代表某一个特征图的新点特征
        """
        B_, N, C = x.shape

        One = torch.ones(B_, 1,1).to(x.device)
        h=self.latent_h(One)

        q=self.q(h)
        q=q.reshape(B_, 1, 1, self.num_heads, -1)
        q=q.permute(2, 0, 3, 1,4)
        q=q.unbind(0)[0]#[B,num_head,1(N),dim],unbind是按照某一维度划分，这一维度就消失了

        kv = self.kv(x).reshape(B_, N, 2, self.num_heads, -1).permute(2, 0, 3, 1,4)  # [kv(2), B_, nheads, N, dim_per_head]
        k, v = kv.unbind(0)  # [B_, nheads, N, dim_per_head]

        k[:,:,0,:]=self.norm(k[:,:,0,:])


        q_u = self.fc_mu(F.relu(q))
        k_var = self.fc_var(F.relu(k))

        q_u=q_u.squeeze(dim=-2)
        sigma = []
        for i in range(N):
            sigma.append(self.reparameterize(q_u , k_var[:, :, i, :]))



        sigma=torch.stack(sigma,dim=-1)
        attn=torch.softmax(sigma,dim=-1)#2000*7

        x = (attn @ v).transpose(1, 2).reshape(B_, 1, -1)  # [B*wh*ww, WH*WW, nheads*dim_per_head]=[B*wh*ww, WH*WW, D]
        x = self.proj(x)
        x = self.proj_drop(x)
        #sigma=sigma.unsqueeze(dim=-1)
        #v=torch.sum(sigma*v,dim=1)

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

