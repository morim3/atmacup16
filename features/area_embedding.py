import torch
import torch.nn as nn


class AreaEmbedding(nn.Module):
    def __init__(self, area_num, embedding_dim, wid_num, ken_num, lrg_num, sml_num, yad_num, yad_sml):
        super(AreaEmbedding, self).__init__()
        self.area_num = area_num
        self.embedding_dim = embedding_dim
        self.wid_num = wid_num
        self.ken_num = ken_num
        self.lrg_num = lrg_num
        self.sml_num = sml_num
        self.yad_num = yad_num

        # yad_sml means where sml resion each yad belongs.
        # shape: (yad_num, )
        self.yad_sml = yad_sml

        self.wid_pos_mu = nn.Parameter(torch.randn(wid_num, embedding_dim))
        self.ken_pos_mu = nn.Parameter(torch.randn(ken_num, embedding_dim))
        self.lrg_pos_mu = nn.Parameter(torch.randn(lrg_num, embedding_dim))
        self.sml_pos_mu = nn.Parameter(torch.randn(sml_num, embedding_dim))
        self.yad_pos = nn.Parameter(torch.randn(yad_num, embedding_dim))

        ## sigma is fixed
        self.wid_pos_sigma = torch.ones(wid_num, )
        self.ken_pos_sigma = torch.ones(ken_num, ) * 0.3
        self.lrg_pos_sigma = torch.ones(lrg_num, ) * 0.1
        self.sml_pos_sigma = torch.ones(sml_num, ) * 0.05
        self.alpha = 0.1


    def forward(self, x):
        """
        :param x: (yad_num, *) * means the number of co-occurence yad

        """

        loss = 0
        loss += (self.wid_pos_mu - self.ken_pos_mu).pow(2).sum() + (self.wid_pos_mu - self.lrg_pos_mu).pow(2).sum() + (self.lrg_pos_mu - self.sml_pos_mu).pow(2).sum() + (self.sml_pos_mu - self.yad_pos).pow(2).sum()


        # calc triplet loss
        for i in range(self.yad_num):
            for j in range(self.yad_num):
                for k in range(self.yad_num):
                    if i == j or j == k or i == k:
                        continue
                    if j in x[i] and k not in x[i]:
                        triplet_loss = (self.yad_pos[i] - self.yad_pos[j]).pow(2).sum() - (self.yad_pos[i] - self.yad_pos[k]).pow(2).sum() + self.alpha
                        loss += (triplet_loss if triplet_loss > 0 else 0)



