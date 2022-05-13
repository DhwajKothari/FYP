import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.wmad_estimator import Wmad_estimator

class SRResDNet(nn.Module):
    def __init__(self, model, scale):
        super(SRResDNet, self).__init__()
        
        self.model = model
        self.upscale_factor = scale
        self.noise_estimator = Wmad_estimator()
        self.alpha = nn.Parameter(torch.Tensor(np.linspace(np.log(2),np.log(1), 1)))
        self.bbproj = nn.Hardtanh(min_val = 0., max_val = 255.)

    def forward(self, input):        
        input = F.interpolate(input, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        sigma = self.noise_estimator(input)
        sigma *= 255.
        output = self.model(input, sigma, self.alpha)
        output = input - output
        output = self.bbproj(output)
        return output