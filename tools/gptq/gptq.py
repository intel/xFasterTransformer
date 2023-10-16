import math
import time

import torch
import torch.nn as nn
# import transformers

from quantizer import *

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class LLM_GPTQ:

    def __init__(self, weight, wbits, sym):
        self.weight = weight
        self.wbits = wbits
        self.rows = self.weight.shape[0]
        self.columns = self.weight.shape[1]
        self.H = torch.zeros((self.rows, self.rows))
        self.nsamples = 0
        self.quantizer = Quantizer()
        self.quantizer.configure(wbits, perchannel=True, sym=sym, mse=False)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
    ):
        W = self.weight.clone()
        W = W.float()
        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[perm, :]
            H = H[perm][:, perm]

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.rows)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H)
        Hinv = H

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        Qint = torch.zeros_like(W)

        for i1 in range(0, self.rows, blocksize):
            i2 = min(i1 + blocksize, self.rows)
            count = i2 - i1

            W1 = W[i1:i2, :].clone()
            Err1 = torch.zeros_like(W1)

            for i in range(count):
                w = W1[i, :]
                d = Hinv[i1 + i, i1 + i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[(i1 + i):(i1 + i + groupsize), :])

                q_int = quantize_to_int(w, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
                Qint[i + i1, :] = q_int
                q = dequantize_to_float(q_int, self.quantizer.scale, self.quantizer.zero)
                Q[i + i1, :] = q

                Losses[i + i1, :] = (w - q) ** 2 / ( 2 * d ** 2)
                err1 = (w - q) / d
                Err1[i, :] = err1

                h = Hinv[(i1 + i):i2, i1 + i]
                err1 = err1.reshape([1, err1.shape[0]])
                h = h.reshape([h.shape[0], 1])
                W1[i:, :] -= h.matmul(err1)

            W[i2:, :] -= Hinv[i2:, i1:i2].matmul(Err1)

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[invperm, :]

        # if isinstance(self.layer, transformers.Conv1D):
        #     Q = Q.t()

        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        print("Input Weight (float):", self.weight)
        print("Output quantized weight (int{})".format(self.wbits), Qint)
        print("Output quantized weight (float)", Q)
        return Qint, self.quantizer.scale, self.quantizer.zero

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
