import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle

# compute gradient d(obj)/d(var)
def grad(outs, ins):
    g = torch.autograd.grad( \
        outs
      , ins
      , grad_outputs = torch.ones_like(outs)
      , create_graph = True
      )

    return g[0]


def trans(scalar, pred, thetas):
  vals = scalar(thetas, pred)
  return grad(vals, pred)


def pkl(obj, fname):
  f = open(fname, 'wb')
  pickle.dump(obj, f)
  f.close()
  return


def of_length(xs, ys):
  n = ys.size()[0]
  return xs[:n]


# zip two tensors along dim=1
def zipt(xs, ys):
  return torch.cat([xs, ys], dim=1)


def detach(obj):
  return obj.squeeze().cpu().detach().numpy()


def cat(xs, ys):
  return torch.cat([xs, ys], dim=0)



def histcurve(bins, fills, default):
  xs = [x for x in bins for _ in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)



def binned_kldiv(targ, pred):
  htarg = np.histogram(targ, bins=100, range=(-1, 1), density=True)[0]
  hpred = np.histogram(pred, bins=100, range=(-1, 1), density=True)[0]
  return np.sum(htarg * np.log(htarg / hpred))


def sort(t):
  return torch.sort(t, dim=0)[0]