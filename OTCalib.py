#!/usr/bin/env python
# coding: utf-8


outprefix = 'paper/'
device='cuda'

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

from otcalibutils import *

totalevents = 2**27
print("total events to train on:", totalevents)
nmc = int(2e6)


decays = [0.95]
acts = [("lrelu", nn.LeakyReLU)] #, ("sig", nn.Sigmoid), ("tanh", nn.Tanh)]
bss = [4096]
npss = [10]
nlayer = [2]
latent = [1024]
lrs = [(x, x) for x in [3e-5]]
dss = [int(1e4), int(1e5)]
gradnorm = False

controlplots(int(1e6))
plt.savefig("controlplots.pdf")


plt.figure(figsize=(6, 6))

weights = tonp(bootstrap(nmc, device))
_ = plt.hist(weights, bins=25, label="bootstraps")
plt.title("bootstrap weights")
plt.show()
plt.savefig("bootstraps.pdf")

for (decay, (actname, activation), batchsize, nps, nlay, nlat, (alr, tlr), datasize) \
  in product(decays, acts, bss, npss, nlayer, latent, lrs, dss):

    nbatches = max(2**7, datasize // batchsize)

    nepochs = totalevents // (nbatches * batchsize)

    print ("number of batches:", nbatches)
    print ("number of epochs:", nepochs)

    alldata = genData(datasize, device)
    allmc = genMC(nmc, device)
    validmc = genMCWithDataPt(nmc, device)
    validdata = genData(datasize, device)

    transport1 = fullyConnected(nlay, 2, nlat, nlat, activation)
    transport2 = fullyConnected(2, nlat+nps, 64, 1, activation)
    transports = (transport1, transport2)

    adversary = fullyConnected(nlay, 2, nlat*2, 1, nn.LeakyReLU)


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
      "thetainputs_bootstrap_rmsprop_act_%s_batch_%d_nps_%d_layers_%d_latent_%d_tlr_%0.2e_alr_%0.2e_datasize_%d" \
        % (actname, batchsize, nps, nlay, nlat, tlr, alr, datasize)

    if gradnorm:
      name = name + "_gradnorm"

    if decay:
      name = name + "_decay_%0.2e" % decay

    writer = SummaryWriter(outprefix + name)


    for epoch in range(nepochs):
      radvloss = 0
      fadvloss = 0
      tadvloss = 0
      ttransloss = 0
      realavg = 0
      fakeavg = 0
      weights = bootstrap(batchsize, device).reshape((batchsize, 1))

      print("epoch:", epoch)

      for batch in range(nbatches):

        data = alldata[torch.randint(alldata.size()[0], size=(batchsize,), device=device)]

        mc = allmc[torch.randint(allmc.size()[0], size=(batchsize,), device=device)]

        toptim.zero_grad()
        aoptim.zero_grad()

        real = adversary(data)
        realavg += torch.mean(real).item()

        tmp1 = \
          binary_cross_entropy_with_logits(
              real
            , torch.ones_like(real)
            , reduction='mean'
            , weight=weights
            )

        radvloss += tmp1.item()


        # add gradient regularization
        if gradnorm:
          grad_params = torch.autograd.grad(tmp1, adversary.parameters(), create_graph=True, retain_graph=True)
          grad_norm = 0
          for grad in grad_params:
              grad_norm += grad.pow(2).sum()
          grad_norm = grad_norm.sqrt()


        thetas = torch.randn((batchsize, nps), device=device)
        transporting = trans(transports, mc, thetas)
        transported = transporting + mc[:,0:1]

        fake = adversary(torch.cat([transported, mc[:,1:]], axis=1))

        fakeavg += torch.mean(fake).item()

        tmp2 = \
          binary_cross_entropy_with_logits(
              fake
            , torch.zeros_like(real)
            , reduction='mean'
            )

        fadvloss += tmp2.item()

        loss = tmp1 + tmp2
        if gradnorm:
          loss += 0.1*grad_norm

        loss.backward()
        aoptim.step()


        toptim.zero_grad()
        aoptim.zero_grad()

        thetas = torch.randn((batchsize, nps), device=device)
        transporting = trans(transports, mc, thetas)
        transported = transporting + mc[:,0:1]

        fake = adversary(torch.cat([transported, mc[:,1:]], axis=1))

        # tmp1 = tloss(transporting)
        # ttransloss += tmp1.item()

        tmp2 =\
          - binary_cross_entropy_with_logits(
            fake
          , torch.zeros_like(real)
          , reduction='mean'
          )

        tadvloss += tmp2.item()

        loss = tmp2 # tmp1 + tmp2

        loss.backward()
        toptim.step()

        del transported, transporting, loss, tmp2 # , tmp1


      if decay:
        tsched.step()
        asched.step()

      # write tensorboard info once per epoch
      writer.add_scalar('radvloss', radvloss / nbatches, epoch)
      writer.add_scalar('fadvloss', fadvloss / nbatches, epoch)
      writer.add_scalar('tadvloss', tadvloss / nbatches, epoch)
      writer.add_scalar('ttransloss', ttransloss / nbatches, epoch)
      writer.add_scalar('realavg', realavg / nbatches, epoch)
      writer.add_scalar('fakeavg', fakeavg / nbatches, epoch)

      # make validation plots once per 10 epochs

      if epoch % 1 == 0:
        print("plotting")

        plotPtTheta(transports, ptbin(25, 50, validmc), ptbin(25, 50, validdata), nps, writer, "pt_25_50", "$25 < p_T < 50$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(50, 75, validmc), ptbin(50, 75, validdata), nps, writer, "pt_50_75", "$50 < p_T < 75$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(75, 100, validmc), ptbin(75, 100, validdata), nps, writer, "pt_75_100", "$75 < p_T < 100$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(100, 150, validmc), ptbin(100, 150, validdata), nps, writer, "pt_100_150", "$100 < p_T < 150$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(150, 200, validmc), ptbin(150, 200, validdata), nps, writer, "pt_150_200", "$150 < p_T < 200$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(200, 300, validmc), ptbin(200, 300, validdata), nps, writer, "pt_200_300", "$200 < p_T < 300$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(300, 500, validmc), ptbin(300, 500, validdata), nps, writer, "pt_300_500", "$300 < p_T < 500$ [GeV]", epoch, device, nmax=250)

        plotPtTheta(transports, ptbin(500, 1000, validmc), ptbin(500, 1000, validdata), nps, writer, "pt_500_1000", "$500 < p_T < 1000$ [GeV]", epoch, device, nmax=250)

      save(outprefix + name + ".pth", transports, adversary, toptim, aoptim)