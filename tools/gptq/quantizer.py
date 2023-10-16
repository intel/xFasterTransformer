import numpy as np
import torch
import torch.nn as nn
    
def quantize_to_int(x, scale, zero, maxq):
    return torch.clamp(torch.round(x / scale) + zero, 0, maxq)

def dequantize_to_float(x, scale, zero):
    return scale * (x - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=True, sym=False, 
        mse=False, norm=2.4, grid=100, maxshrink=.8
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 

    def find_params(self, x):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if not self.perchannel:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[1], device=dev)
        xmin = torch.minimum(x.min(0)[0], tmp)
        xmax = torch.maximum(x.max(0)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize_to_int(x, scale1, zero1, self.maxq)
                q = dequantize_to_float(q, scale1, zero1)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

    def quantize_to_int(self, x):
        if self.ready():
            return quantize_to_int(x, self.scale, self.zero, self.maxq)
        return x

    def dequantize_to_float(self, x):
        if self.ready():
            return dequantize_to_float(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)