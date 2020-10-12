#!/usr/bin/env python
# coding: utf-8


outprefix = 'gaussloss/'
device='cuda'

# TODO
# I don't know how many of these imports are strictly necessary.

from time import time
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

from otcalibutils import *

nmc = int(2e6)
maxtime = 90*60

ncritics = [3]
decays = [0.97]
acts = [("lrelu", nn.LeakyReLU)] #, ("sig", nn.Sigmoid), ("tanh", nn.Tanh)]
bss = [4096]
npss = [16]
nlayer = [2]
latent = [1024]
lrs = [(x, x) for x in [5e-5]]
dss = [int(1e4), int(1e5)]

controlplots(int(1e6))
plt.savefig("controlplots.pdf")


plt.figure(figsize=(6, 6))

# weights = tonp(bootstrap(nmc, device))
# _ = plt.hist(weights, bins=25, label="bootstraps")
# plt.title("bootstrap weights")
# plt.show()
# plt.savefig("bootstraps.pdf")
# del weights

for (decay, (actname, activation), batchsize, nps, nlay, nlat, (alr, tlr), datasize, ncritic) \
  in product(decays, acts, bss, npss, nlayer, latent, lrs, dss, ncritics):

    alldata = genData(datasize, device)
    allmc = genMCWithDataPt(nmc, device)
    validmc = genMCWithDataPt(nmc, device)
    validdata = genData(datasize, device)
    # bootstraps = bootstrap(datasize*nps, device).reshape((datasize, nps))
    # bootstraps = torch.randn((datasize, nps), device=device)
    # bootstraps.clamp_(0, 99999)

    transport1 = fullyConnected(nlay, 2, nlat, nlat, activation)
    transport2 = fullyConnected(4, nlat+nps, nps, 1, activation)
    transports = (transport1, transport2)

    adversary = fullyConnected(nlay, 2, 2*nlat, 1, activation)


    transport1.to(device)
    transport2.to(device)
    adversary.to(device)
    
    toptim = \
      torch.optim.RMSprop(
          list(transport1.parameters()) \
          + list(transport2.parameters())
        , lr=tlr
        )
    aoptim = torch.optim.RMSprop(adversary.parameters(), lr=alr)

    if decay:
      tsched = torch.optim.lr_scheduler.ExponentialLR(toptim, decay)
      asched = torch.optim.lr_scheduler.ExponentialLR(aoptim, decay)


    name = \
      "gaussloss_ptsame_thetavars_wasserstein_rmsprop_act_%s_batch_%d_nps_%d_layers_%d_latent_%d_tlr_%0.2e_alr_%0.2e_datasize_%d" \
        % (actname, batchsize, nps, nlay, nlat, tlr, alr, datasize)

    name += "_ncritic_%d" % ncritic

    if decay:
      name = name + "_decay_%0.2e" % decay

    writer = SummaryWriter(outprefix + name)

    epoch = 0
    advloss = 0
    ttransloss = 0
    nbatches = 0
    starttime = time()
    lastplot = starttime
    print("epoch:", 0)

    while True:
      nbatches += 1

      # plot every 120 seconds
      now = time()
      if now - lastplot > 60:
        lastplot = now
        print("plotting")

        plotPtTheta(transports, ptbin(25, 50, validmc), ptbin(25, 50, validdata), nps, writer, "pt_25_50", "$25 < p_T < 50$ [GeV]", epoch, device, nmax=250)

        # plotPtTheta(transports, ptbin(50, 75, validmc), ptbin(50, 75, validdata), nps, writer, "pt_50_75", "$50 < p_T < 75$ [GeV]", epoch, device, nmax=250)

        # plotPtTheta(transports, ptbin(75, 100, validmc), ptbin(75, 100, validdata), nps, writer, "pt_75_100", "$75 < p_T < 100$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(100, 150, validmc), ptbin(100, 150, validdata), nps, writer, "pt_100_150", "$100 < p_T < 150$ [GeV]", epoch, device, nmax=250)

        # plotPtTheta(transports, ptbin(150, 200, validmc), ptbin(150, 200, validdata), nps, writer, "pt_150_200", "$150 < p_T < 200$ [GeV]", epoch, device, nmax=250)

        # plotPtTheta(transports, ptbin(200, 300, validmc), ptbin(200, 300, validdata), nps, writer, "pt_200_300", "$200 < p_T < 300$ [GeV]", epoch, device, nmax=250)

        # plotPtTheta(transports, ptbin(300, 500, validmc), ptbin(300, 500, validdata), nps, writer, "pt_300_500", "$300 < p_T < 500$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(500, 1000, validmc), ptbin(500, 1000, validdata), nps, writer, "pt_500_1000", "$500 < p_T < 1000$ [GeV]", epoch, device, nmax=250)

        save(outprefix + name + ".pth", transports, adversary, toptim, aoptim)

        # write tensorboard info once per epoch
        writer.add_scalar('advloss', advloss / nbatches, epoch)
        writer.add_scalar('ttransloss', ttransloss / nbatches, epoch)

        if now - starttime > maxtime:
          break

        epoch += 1
        advloss = 0
        ttransloss = 0
        nbatches = 0

        print("epoch:", epoch)


      for _ in range(ncritic):
        dataidxs = torch.randint(alldata.size()[0], size=(batchsize,), device=device)
        data = alldata[dataidxs]

        thetas = torch.randn((batchsize, nps), device=device)

        mc = allmc[torch.randint(allmc.size()[0], size=(batchsize,), device=device)]

        toptim.zero_grad()
        aoptim.zero_grad()

        real = adversary(data)

        rloss = -torch.mean(real) + torch.abs(torch.std(real) - 1)


        transporting = trans(transports, mc, thetas)
        transported = transporting + mc[:,0:1]

        fake = adversary(torch.cat([transported, mc[:,1:]], axis=1))

        floss = torch.mean(fake)

        loss = rloss + floss

        advloss += loss.item()

        loss.backward()
        aoptim.step()

        for p in adversary.parameters():
          p.data.clamp_(-0.1, 0.1)


        toptim.zero_grad()
        aoptim.zero_grad()


      data = alldata[torch.randint(alldata.size()[0], size=(batchsize,), device=device)]

      mc = allmc[torch.randint(allmc.size()[0], size=(batchsize,), device=device)]

      thetas = torch.zeros((batchsize, nps), device=device)
      transporting = trans(transports, mc, thetas)
      transported = transporting + mc[:,0:1]

      fake = adversary(torch.cat([transported, mc[:,1:]], axis=1))

      thetas = torch.randn((batchsize, nps), device=device)
      transporting = trans(transports, mc, thetas)
      transported = transporting + mc[:,0:1]
      spread = adversary(torch.cat([transported, mc[:,1:]], axis=1))

      loss = -torch.mean(fake) + torch.abs(torch.std(spread) - 1) 

      loss.backward()
      toptim.step()

      del transported, transporting, loss


    if decay:
      tsched.step()
      asched.step()