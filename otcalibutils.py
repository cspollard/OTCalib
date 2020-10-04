# TODO
# I don't know how many of these imports are strictly necessary.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from math import e
from scipy.special import lambertw


import torch.nn as nn
from collections import OrderedDict

from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.tensorboard import SummaryWriter
from math import log, exp

from itertools import product

def poly(cs, xs):
  ys = torch.zeros_like(xs)
  for (i, c) in enumerate(cs):
    ys += c * xs**i

  return ys


def discMC(xs, logpts):
  return poly([0, 0.6, 1.2, 1.1], xs) * poly([1, 0.1, -0.01], logpts)

def discData(xs, logpts):
  return poly([0, 0.7, 1.1, 1.3], xs) * poly([1, -0.1, 0.02], logpts)


def logptMC(xs):
  return torch.log(poly([25, 200, -2], -torch.log(xs)))

def logptData(xs):
  return torch.log(poly([25, 220, 5], -torch.log(xs)))



# need to add a small number to avoid values of zero.
def genMC(n, device):
  xs = torch.rand(n, device=device) + 1e-5
  logpts = logptMC(xs)
  ys = torch.rand(n, device=device) + 1e-5
  ds = discMC(ys, logpts)
  return torch.stack([ds, logpts]).transpose(0, 1)

def genData(n, device):
  xs = torch.rand(n, device=device) + 1e-5
  logpts = logptData(xs)
  ys = torch.rand(n, device=device) + 1e-5
  ds = discData(ys, logpts)
  return torch.stack([ds, logpts]).transpose(0, 1)


def genMCWithDataPt(n, device):
  xs = torch.rand(n, device=device) + 1e-5
  logpts = logptData(xs)
  ys = torch.rand(n, device=device) + 1e-5
  ds = discMC(ys, logpts)
  return torch.stack([ds, logpts]).transpose(0, 1)

# give high-pT jets more weight to improve convergence
# similar idea to boosting
def ptWeight(logpts):
    pts = torch.exp(logpts)
    w = torch.exp(pts / e**5.5)
    return w / torch.mean(w)


def histcurve(bins, fills, default):
  xs = [x for x in bins for i in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)


def controlplots(n):
  mc = genMC(n, 'cpu').numpy()
  data = genData(n, 'cpu').numpy()

  plt.figure(figsize=(30, 10))

  plt.subplot(1, 3, 1)

  _ = plt.hist([np.exp(mc[:,1]), np.exp(data[:,1])], bins=25, label=["mc", "data"])
  plt.title("pT")
  plt.yscale("log")
  plt.legend()

  plt.subplot(1, 3, 2)


  _ = plt.hist([mc[:,1], data[:,1]], bins=25, label=["mc", "data"])
  plt.title("log pT")
  plt.legend()

  plt.subplot(1, 3, 3)

  _ = plt.hist([mc[:,0], data[:,0]], bins=25, label=["mc", "data"])
  plt.title("discriminant")
  plt.legend()
  plt.show()

  plt.close()

# test(int(1e6))

def bootstrap(n, device):
  xs = np.random.rand(n)
  ws = lambertw((xs-1)/np.e, k=-1).astype(np.float)
  return - torch.from_numpy(ws).to(device) - 1


def layer(n, m, act):
  return \
    nn.Sequential(
      nn.Linear(n, m)
    , act(inplace=True)
    )


def sequential(xs):
    d = OrderedDict()
    for (i, x) in enumerate(xs):
        d[str(i)] = x

    return nn.Sequential(d)


def fullyConnected(nl, n0, nmid, nf, act):
  return \
    nn.Sequential(
      nn.Linear(n0, nmid)
    , act(inplace=True)
    , sequential([layer(nmid, nmid, act) for i in range(nl)])
    , nn.Linear(nmid, nf)
    )


def tloss(xs):
  return torch.mean(xs**2)


def save(path, transports, adversary, toptim, aoptim):
  torch.save(
      { 'transport1_state_dict' : transports[0].state_dict()
      , 'transport2_state_dict' : transports[1].state_dict()
      , 'adversary_state_dict' : adversary.state_dict()
      , 'toptim_state_dict' : toptim.state_dict()
      , 'aoptim_state_dict' : aoptim.state_dict()
      }
    , path
  )

def load(path, transports, adversary, toptim, aoptim):
  checkpoint = torch.load(path)
  transports[0].load_state_dict(checkpoint["transport1_state_dict"])
  transports[1].load_state_dict(checkpoint["transport2_state_dict"])
  adversary.load_state_dict(checkpoint["adversary_state_dict"])
  toptim.load_state_dict(checkpoint["toptim_state_dict"])
  aoptim.load_state_dict(checkpoint["aoptim_state_dict"])



def tonp(xs):
  return xs.cpu().detach().numpy()


def plotPtTheta(transports, predict, targ, nps, writer, label, title, epoch, device, nmax=-1):

  target = tonp(targ)
  prediction = tonp(predict)

  zeros = torch.zeros((predict.size()[0], nps), device=device)

  thetas = zeros.clone()
  transporting = trans(transports, predict, thetas)
  transported = transporting + predict[:,0:1]

  nom = tonp(transported)

  postrans = []
  negtrans = []
  pos = []
  neg = []

  if nmax < 0:
    nmax = predict.size()[0]


  for i in range(nps):
    thetas = zeros.clone()
    thetas[:,i] = 1
    transporting = trans(transports, predict, thetas)
    transported = transporting + predict[:,0:1]
    postrans.append(tonp(transporting))
    pos.append(tonp(transported)[:,0])

    thetas = zeros.clone()
    thetas[:,i] = -1
    transporting = trans(transports, predict, thetas)
    transported = transporting + predict[:,0:1]
    negtrans.append(tonp(transporting))
    neg.append(tonp(transported)[:,0])


  rangex = (-0.5, 4.5)
  rangey = (-1, 1)
  nbins = 25
  binw = (rangex[1] - rangex[0]) / nbins
  
  b = np.arange(nbins+1) * binw + rangex[0]

  def clip(x):
    return np.clip(rangex[0], rangex[1], x)

  h, _, _ = \
    plt.hist(
        list(map(clip, [ prediction[:,0], nom[:,0], target[:,0] ] ))
      , bins=b
      , range=rangex
      , label=["original prediction", "transported prediction", "target"]
      , density=True
      )

  htmp, _, _ = \
    plt.hist(
      target[:,0]
    , bins=b
    , range=rangex
    , density=False
    )

  targuncerts = np.sqrt(htmp) / np.sum(htmp) / binw

  hpred = h[0]
  htrans = h[1]
  htarg = h[2]



  hpos, _, _ = \
    plt.hist(
      list(map(clip, pos))
    , bins=b
    , range=rangex
    , density=True
    )

  hneg, _, _ = \
    plt.hist(
      list(map(clip, neg))
    , bins=b
    , range=rangex
    , density=True
    )

  # numpy is a total pile of crap.
  # if the number of nps is one, then e.g. "hpos" is a list of bin
  # counts
  # if the number of nps is anything else, then e.g. "hpos" is a list
  # of list of bin counts.

  if nps == 1:
    hpos = [hpos]
    hneg = [hneg]

  plt.close()

  writer.add_scalar('binnedkldiv_' + label, binnedkldiv(htrans, htarg), epoch)

  # for i in range(len(hpos)):
  #   hup = hpos[i]
  #   hdown = hneg[i]

  #   writer.add_scalar('binnedkldiv_' + label + "_theta%d_up" % i, binnedkldiv(hup, htarg), epoch)
  #   writer.add_scalar('binnedkldiv_' + label + "_theta%d_down" % i, binnedkldiv(hdown, htarg), epoch)



  cols = ["green", "orange", "magenta", "blue"]*20


  fig = plt.figure(figsize=(6, 6))

  herr2 = np.zeros_like(htrans)
  for i in range(len(hpos)):
    hup = hpos[i]
    hdown = hneg[i]

    herr2 += ((np.abs(hup - htrans) + np.abs(hdown - htrans)) / 2.0)**2

  herr = np.sqrt(herr2)
  hup = htrans + herr
  hdown = htrans - herr

  (xs, yups) = histcurve(b, hup, 0)
  (xs, ydowns) = histcurve(b, hdown, 0)
  plt.fill_between(xs, yups, ydowns, color="gray", alpha=0.5, label="transported uncertainty")

  (xs, ys) = histcurve(b, htrans, 0)
  plt.plot(xs, ys, linewidth=2, color="black", label="transported prediction")


  (xs, ys) = histcurve(b, hpred, 0)
  plt.plot(
      xs
    , ys
    , linewidth=2
    , color="red"
    , linestyle="dashed"
    , label="original prediction"
    )


  plt.errorbar(
      (b[:-1] + b[1:]) / 2.0
    , htarg
    , label="target"
    , color='black'
    , linewidth=0
    , yerr=targuncerts
    , fmt='o'
    , ecolor='black'
    , elinewidth=1
    )

  plt.title(title)
  plt.xlim(rangex)
  plt.ylim(0, 1.5)
  plt.xlabel("discriminant")
  plt.ylabel("event density")
  plt.legend()

  writer.add_figure("hist_%s" % label, fig, global_step=epoch)
  plt.close()



  fig = plt.figure(figsize=(6, 6))

  plt.plot((rangex[0], rangex[1]), (1, 1), color='black', linewidth=1, alpha=0.5)


  hup = hup / htrans
  hdown = hdown / htrans
  (xs, yups) = histcurve(b, hup, 1)
  (xs, ydowns) = histcurve(b, hdown, 1)
  plt.fill_between(xs, yups, ydowns, color="gray", alpha=0.5, label="transported uncertainty")

  (xs, ys) = histcurve(b, hpred / htrans, 1)
  plt.plot(
      xs
    , ys
    , label="original prediction"
    , linewidth=3
    , color="red"
    , linestyle="dashed"
    )

  plt.errorbar(
      (b[:-1] + b[1:]) / 2.0
    , htarg / htrans
    , label="target"
    , color='black'
    , linewidth=0
    , yerr = targuncerts / htrans
    , fmt='o'
    , ecolor='black'
    , elinewidth=1
    )


  plt.title(title)
  plt.ylim(0.5, 1.5)
  plt.xlim(0, 4)
  plt.xlabel("discriminant")
  plt.ylabel("ratio to transported prediction")
  plt.legend()

  writer.add_figure("ratio_%s" % label, fig, global_step=epoch)
  plt.close()



  fig = plt.figure(figsize=(6, 6))

  plt.plot((rangex[0], rangex[1]), (0, 0), color='black', linewidth=1, alpha=0.5)

  # for (i, ys) in enumerate(postrans):
  #   _ = \
  #     plt.scatter(
  #         prediction[:nmax,0]
  #       , ys[:nmax]
  #       , c=cols[i]
  #       , marker='.'
  #       , alpha=0.5
  #       , label="$\\theta_%d$ variation" % i
  #     )

  # for (i, ys) in enumerate(negtrans):
  #   _ = \
  #     plt.scatter(
  #         prediction[:nmax,0]
  #       , ys[:nmax]
  #       , c=cols[i]
  #       , marker='.'
  #       , alpha=0.5
  #     )

  thetas = torch.randn_like(thetas[:nmax])
  predict = predict[:nmax]
  transporting = trans(transports, predict, thetas)
  transported = transporting + predict[:,0:1]

  _ =  \
    plt.scatter(
        tonp(predict[:,0])
      , tonp(transporting)
      , c="black"
      , marker='.'
      , label='stochastic transport'
    )



  plt.xlim(rangex)
  plt.ylim(rangey)
  plt.title(title)
  plt.xlabel("original predicted discriminant")
  plt.ylabel("transport vector")
  plt.legend()

  writer.add_figure("transport_%s" % label, fig, global_step=epoch)
  plt.close()

  return



def trans(transports, mc, thetas):
    x = transports[0](mc)
    x = transports[1](torch.cat((x, thetas), axis=1))
    return x + mc[:,0:1]

    # tmp = transport(mc)
    # cv = tmp[:,0:1] # central value
    # var = tmp[:,1:] # eigen variations
    # coeffs = var - cv

    # corr = torch.bmm(thetas.unsqueeze(1), coeffs.unsqueeze(2))

    # return cv + corr.squeeze(2)



def ptbin(low, high, samps):
  pts = torch.exp(samps[:,1])
  lowcut = low < pts
  highcut = pts < high

  cut = torch.logical_and(lowcut, highcut)

  return samps[cut]


def binnedkldiv(nom, var):
  valid = np.logical_and(nom != 0, var != 0)
  nom = nom[valid]
  var = var[valid]
  return np.sum(nom * np.log(nom / var))