# TODO
# I don't know how many of these imports are strictly necessary.

import numpy as np
import matplotlib    
matplotlib.use('Agg')    
import matplotlib.pyplot as plt    
import torch
from math import e
import otcalibutils


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
  return torch.log(poly([25, 200, 7], -torch.log(xs)))

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


# give high-pT jets more weight to improve convergence
# similar idea to boosting
def ptWeight(logpts):
    pts = torch.exp(logpts)
    w = torch.exp(pts / e**5.5)
    return w / torch.mean(w)




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

# test(int(1e6))


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


def save(path, transport, adversary, toptim, aoptim):
  torch.save(
      { 'transport_state_dict' : transport.state_dict()
      , 'adversary_state_dict' : adversary.state_dict()
      , 'toptim_state_dict' : toptim.state_dict()
      , 'aoptim_state_dict' : aoptim.state_dict()
      }
    , path
  )

def load(path, transport, adversary, toptim, aoptim):
  checkpoint = torch.load(path)
  transport.load_state_dict(checkpoint["transport_state_dict"])
  adversary.load_state_dict(checkpoint["adversary_state_dict"])
  toptim.load_state_dict(checkpoint["toptim_state_dict"])
  aoptim.load_state_dict(checkpoint["aoptim_state_dict"])



def tonp(xs):
  return xs.cpu().detach().numpy()


def plotPtTheta(transport, pt, toys, nps, writer, label, epoch, device):
  logpt = log(pt)

  zeros = torch.zeros((toys.size()[0], nps), device=device)
  logpts = torch.ones(toys.size()[0], device=device)*logpt

  data = torch.stack([torch.sort(discData(toys, logpts))[0], logpts]).transpose(0, 1)
  mc = torch.stack([torch.sort(discMC(toys, logpts))[0], logpts]).transpose(0, 1)

  thetas = zeros.clone()
  transporting = trans(transport, mc, thetas)
  nomtrans = tonp(transporting)
  nom = tonp(transporting + mc[:,0:1])

  postrans = []
  negtrans = []
  pos = []
  neg = []

  for i in range(nps):
    thetas = zeros.clone()
    thetas[:,i] = 1
    transporting = trans(transport, mc, thetas)
    postrans.append(tonp(transporting))
    pos.append(tonp(transporting + mc[:,0:1])[:,0])

    thetas = zeros.clone()
    thetas[:,i] = -1
    transporting = trans(transport, mc, thetas)
    negtrans.append(tonp(transporting))
    neg.append(tonp(transporting + mc[:,0:1])[:,0])



  data = tonp(data)
  mc = tonp(mc)

  fig = plt.figure(figsize=(6, 6))

  rangex = (0, 5)
  rangey = (-1, 1)

  h, b, _ = \
    plt.hist(
        [mc[:,0], nom[:,0], data[:,0]]
      , bins=25
      , range=rangex
      , density=True
      , label=["original prediction", "transported prediction", "data"]
      )
  
  plt.title("discriminant distribution, (pT = %0.2f)" % exp(logpt))
  plt.xlabel("discriminant")
  plt.legend()
    
  writer.add_figure("%shist" % label, fig, global_step=epoch)
  plt.close()
    
    
  cols = ["red", "green", "red", "orange", "magenta", "blue"]

  hpos, _, _ = plt.hist(
        pos
      , bins=b
      , range=rangex
      , density=True
      )
  
  hneg, _, _ = plt.hist(
        neg
      , bins=b
      , range=rangex
      , density=True
      )
  

  fig = plt.figure(figsize=(6, 6))


  # numpy is a total pile of crap.
  # if the number of nps is one, then e.g. "hpos" is a list of bin
  # counts
  # if the number of nps is anything else, then e.g. "hpos" is a list
  # of list of bin counts.


  if nps == 1:
    _ = \
      plt.plot(
          (b[:-1] + b[1:]) / 2.0
        , hpos - h[2]
        , linewidth=1
        , color=cols[i]
        , linestyle='dashed'
        )

    _ = \
      plt.plot(
          (b[:-1] + b[1:]) / 2.0
        , hneg - h[2]
        , linewidth=1
        , color=cols[i]
        , linestyle='dashed'
        )
    
  else:
      for (i, hvar) in enumerate(hpos):
        _ = plt.plot(           (b[:-1] + b[1:]) / 2.0
            , hvar - h[2]
            , linewidth=1
            , color=cols[i]
            , linestyle='dashed'
            )
    
      for (i, hvar) in enumerate(hneg):
        _ = plt.plot(           (b[:-1] + b[1:]) / 2.0
            , hvar - h[2]
            , linewidth=1
            , color=cols[i]
            , linestyle='dashed'
            )
    

  _ = plt.plot(         (b[:-1] + b[1:]) / 2.0
      , h[0] - h[2]
      , label="mc"
      , linewidth=3
      )

  _ = plt.plot(         (b[:-1] + b[1:]) / 2.0
      , h[1] - h[2]
      , label="nominal transported"
      , linewidth=3
      )



  plt.ylim(-0.5, 0.5)
  plt.title("discriminant difference to data, (pT = %0.2f)" % exp(logpt))
  plt.xlabel("discriminant")
  plt.ylabel("prediction - data")
  plt.legend()
    
  writer.add_figure("%sdiff" % label, fig, global_step=epoch)
  plt.close()
    


  fig = plt.figure(figsize=(6, 6))
  
  for (i, ys) in enumerate(postrans):
    _ = \
      plt.plot(
          mc[:,0]
        , ys
        , c=cols[i]
      )

  for (i, ys) in enumerate(negtrans):
    _ = \
      plt.plot(
          mc[:,0]
        , ys
        , c=cols[i]
      )


  _ =  \
    plt.plot(
        mc[:,0]
      , nomtrans
      , c="black"
      , lw=4
    )


  
  plt.xlim(rangex)
  plt.ylim(rangey)
  plt.title("discriminant transport, (pT = %0.2f)" % exp(logpt))
  plt.xlabel("mc discriminant")
  plt.ylabel("transport vector")
    

        
  writer.add_figure("%strans" % label, fig, global_step=epoch)
  plt.close()
    
  return



def trans(transport, mc, thetas):
    tmp = transport(mc)
    cv = tmp[:,0:1] # central value
    var = tmp[:,1:] # eigen variations 
    coeffs = var - cv

    corr = torch.bmm(thetas.unsqueeze(1), coeffs.unsqueeze(2))
    
    return cv + corr.squeeze(2)
    

