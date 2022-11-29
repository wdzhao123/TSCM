import torch
import torch.nn as nn


# Cn_Sm indicates feature combined by Content from n & Style from m


class Instance_norm(nn.Module):
    def __init__(self, eps=1e-6):
        super(Instance_norm, self).__init__()
        self.eps = eps

    def forward(self, x):
        if len(x.size()) == 4:
            N, C, H, W = x.size()
            x = x.view(N, C, -1)  # kepp N,C channel ; put flat H,W channel ; N,C,H,W ---> N,C,H*W
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(N, C, H, W)
            return x, mean, var
        elif len(x.size()) == 2:
            N, M = x.size()
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
            return x, mean, var


class Instance_unnorm(nn.Module):
    def __init__(self, eps=1e-6, ):
        super(Instance_unnorm, self).__init__()
        self.eps = eps
        # self.mean = mean
        # self.var = var

    def forward(self, x, mean, var):
        if len(x.size()) == 4:
            N, C, H, W = x.size()
            x = x.view(N, C, -1)  # kepp N,C channel ; put flat H,W channel ; N,C,H,W ---> N,C,H*W
            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)
            return x
        elif len(x.size()) == 2:
            N, M = x.size()
            x = x * (var + self.eps).sqrt() + mean
            return x


def img123_IN(x):  # input channel (3N*1024*14*14)
    instance_norm = Instance_norm()
    instance_unnorm = Instance_unnorm()

    x_norm, mean, var = instance_norm(x)
    # print(x_norm.sum().item())
    split_sector_size = int(x_norm.size()[0] / 3)
    x_norm_tuple = torch.split(x_norm, split_sector_size, dim=0)  # 3*(N*C*H*W)
    x_norm_img1 = x_norm_tuple[0]  # img1 class1 C1
    x_norm_img2 = x_norm_tuple[1]  # img2 class1 C2
    x_norm_img3 = x_norm_tuple[2]  # img3 class2 C3

    mean_tuple = torch.split(mean, split_sector_size, dim=0)  # 3*(N*C*1)
    mean_img1 = mean_tuple[0]  # style1
    mean_img2 = mean_tuple[1]  # style2
    mean_img3 = mean_tuple[2]  # style3

    var_tuple = torch.split(var, split_sector_size, dim=0)  # 3*(N*C*1)
    var_img1 = var_tuple[0]  # style1
    var_img2 = var_tuple[1]  # style2
    var_img3 = var_tuple[2]  # style3

    C1_S2 = instance_unnorm(x_norm_img1, mean_img2, var_img2)  # close
    C1_S3 = instance_unnorm(x_norm_img1, mean_img3, var_img3)  # close

    C2_S1 = instance_unnorm(x_norm_img2, mean_img1, var_img1)  # far
    C3_S1 = instance_unnorm(x_norm_img3, mean_img1, var_img1)  # far

    return C1_S2, C1_S3, C2_S1, C3_S1


if __name__ == '__main__':
    x = torch.ones([36, 128])
    y = img123_IN(x)
    for i in y:
        print(i.size(), '  should be 12*3*224*224',
              i.sum().item(), 'should be 12*3*224*224*50=90316800')
    pass
