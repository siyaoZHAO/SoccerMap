import copy
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
import train
import torch
import arg
from pylab import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

argument = arg.arg

def main(arg):
    torch.cuda.empty_cache()
    torch.cuda.set_device(2)  # 设置显卡号
    atrain = train.Trainer(arg)
    atrain.train()


main(argument)