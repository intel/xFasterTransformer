import torch
import os

torch.classes.load_library(os.path.dirname(os.path.abspath(__file__)) + "/libxfastertransformer_pt.so")

from .automodel import AutoModel
