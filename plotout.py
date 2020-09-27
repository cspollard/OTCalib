#!/usr/bin/env python
# coding: utf-8


device='cuda'
outprefix = 'paper/'

# TODO
# I don't know how many of these imports are strictly necessary.

import numpy as np
import matplotlib    
matplotlib.use('Agg')    
import matplotlib.pyplot as plt    
import torch
from math import e


import torch.nn as nn
from collections import OrderedDict

from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.tensorboard import SummaryWriter
from math import log, exp

from itertools import product
import os

from otcalibutils import *

# toys for validation samples
nval = 2**15
valtoys = torch.rand(nval, device=device)


nbatches = 2**10
nepochs = 100


acts = [("lrelu", nn.LeakyReLU)] #, ("sig", nn.Sigmoid), ("tanh", nn.Tanh)]
bss = [2**n for n in [8]]
npss = [1]
nlayer = [4, 3]
latent = [1024, 256]
lrs = [(0.5, 10**l) for l in [-2, -3]]
dss = [int(1e5), int(1e7)]


for ((actname, activation), batchsize, nps, nlay, nlat, (alr, tlr), datasize)   in product(acts, bss, npss, nlayer, latent, lrs, dss):

    transport = fullyConnected(nlay, 2, nlat, 1+nps, activation)

    adversary = fullyConnected(nlay, 2, nlat*2, 1, nn.LeakyReLU)


    transport.to(device)
    adversary.to(device)
    
    toptim = torch.optim.SGD(transport.parameters(), lr=tlr)
    aoptim = torch.optim.SGD(adversary.parameters(), lr=alr)

    name = "scan7_sgd_%s_%d_%d_%d_%d_%0.2e_%d" % (actname, batchsize, nps, nlay, nlat, tlr, datasize)

    # os.makedirs(outprefix + name + "/plots", exist_ok=True)

    load(outprefix + name + ".pth", transport, adversary, toptim, aoptim)

    writer = SummaryWriter(outprefix + name)

    # make validation plots once per epoch
    plotPtTheta(transport, 25, valtoys, nps, writer, "pt25", 0, device=device)

    plotPtTheta(transport, 100, valtoys, nps, writer, "pt100", 0, device=device)

    plotPtTheta(transport, 500, valtoys, nps, writer, "pt500", 0, device=device)
  
    plotPtTheta(transport, 1000, valtoys, nps, writer, "pt1000", 0, device=device)