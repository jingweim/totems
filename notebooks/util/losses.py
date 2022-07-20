import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class LPIPS_Loss(nn.Module):
    def __init__(self, model='net-lin', net='vgg', use_gpu=True, spatial=False):
        super(LPIPS_Loss, self).__init__()
        self.lpips = lpips.LPIPS(net=net, spatial=True).eval()

    def forward(self, pred, ref):
        dist = self.lpips(pred, ref)
        return dist



