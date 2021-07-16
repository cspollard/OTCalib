# TODO
# change back to histograms

import functools

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sys import argv
from ICNN import ICNN, quad_LReLU, smooth_leaky_ReLU
import pickle

print("torch version:", torch.__version__)

outprefix="ftag/"


from sys import argv, stdout
import shutil
import json


if len(argv) < 2:
  print("please provide a json steering file")
  exit(-1)

fconfig = open(argv[1])
config = json.load(fconfig)
fconfig.close()


outdir = config["outdir"]
device = config["device"]

number_samples_target = config["number_samples_target"]
number_samples_source = config["number_samples_source"]
normfac = float(number_samples_target) / float(number_samples_source)

f_nonconvex_shape = config["f_nonconvex_shape"]
f_convex_shape = config["f_convex_shape"]
g_nonconvex_shape = config["g_nonconvex_shape"]
g_convex_shape = config["g_convex_shape"]

assert len(f_nonconvex_shape) > 0 and len(g_nonconvex_shape) > 0
assert f_nonconvex_shape[0] == g_nonconvex_shape[0]

assert len(f_convex_shape) > 0 and len(g_convex_shape) > 0
assert f_convex_shape[0] == g_convex_shape[0]

number_dims = f_convex_shape[0]
number_thetas = f_nonconvex_shape[0]

# the number of nps needed to make his module run.
# any nps that aren't input to the networks will be set to zero.
number_nps = 1

batch_size = config["batch_size"]
epoch_size = config["epoch_size"]
number_epochs = config["number_epochs"]
modeljson = config["model"]

lr_f = config["lr_f"]
lr_g = config["lr_g"]
g_per_f = config["g_per_f"]


def cat(xs, ys):
  return torch.cat([xs, ys], dim=0)


sigcenter = 0.1
sigwidth1 = 0.1
sigwidth2 = 0.25
sigfrac1 = 0.5
bkgcenter = -0.5
bkgwidth = 1
sigfrac = 0.5


def targetmodel(n):
  nsig = int(n * sigfrac)
  nsig1 = int(nsig * sigfrac1)
  nsig2 = nsig - nsig1

  sig1 = torch.randn((nsig1, 1), device=device)*sigwidth1 + sigcenter
  sig2 = torch.randn((nsig2, 1), device=device)*sigwidth2 + sigcenter

  sig = cat(sig1, sig2)

  bkg = torch.randn((n - nsig, 1), device=device)*bkgwidth + bkgcenter

  sig.requires_grad = True
  bkg.requires_grad = True

  return cat(sig, bkg)


def signalmodel(thetas):
  n = thetas.size()[0]
  thissigcenter = - 0.5 + 0.1*thetas[:,:1]
  thissigwidth = 0.25 + 0.1*thetas[:,:1]
  thissigwidth = torch.clamp(thissigwidth, 0.05, 99999)
  return torch.randn((n, 1), device=device)*thissigwidth + thissigcenter


def backgroundmodel(thetas):
  n = thetas.size()[0]
  thisbkgcenter = bkgcenter
  thisbkgwidth = bkgwidth
  return torch.randn((n, 1), device=device)*thisbkgwidth + thisbkgcenter


def prediction(thetas):
  n = thetas.size()[0]

  nsig = int(n * sigfrac)
  sig = signalmodel(thetas[:nsig])
  bkg = backgroundmodel(thetas[nsig:])

  sig.requires_grad = True
  bkg.requires_grad = True

  return (sig, bkg)


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


def grid(ranges):
  return np.mgrid(ranges)


def histcurve(bins, fills, default):
  xs = [x for x in bins for _ in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)


def pkl(obj, fname):
  f = open(fname, 'wb')
  pickle.dump(obj, f)
  f.close()
  return


def plot_hist(name, epoch, target, pred, trans, predm1=None, predp1=None, transm1=None, transp1=None):
      bins = [-1 + x*0.1 for x in range(21)]

      fig = plt.figure(figsize=(6, 6))
      ax = fig.add_subplot(111)

      if any(x is None for x in [predm1, predp1, transm1, transp1]):
        variations = False
        hists = [ target , pred , trans ]
      else:
        variations = True
        hists = [ target , pred , trans , predm1 , predp1 , transm1 , transp1 ]

      hs, bins, _ = ax.hist( hists , bins=bins , density=False)

      (xs, htarget) = histcurve(bins, hs[0], 0)
      (xs, hprednom) = histcurve(bins, hs[1], 0)
      (xs, htransnom) = histcurve(bins, hs[2], 0)
      if variations:
        (xs, hpredup) = histcurve(bins, hs[3], 0)
        (xs, hpreddown) = histcurve(bins, hs[4], 0)
        (xs, htransup) = histcurve(bins, hs[5], 0)
        (xs, htransdown) = histcurve(bins, hs[6], 0)

      fig.clear()
      
      plt.scatter(
          (bins[:-1] + bins[1:]) / 2.0
        , hs[0]
        , label="target data"
        , color='black'
        , linewidth=0
        , marker='o'
        , zorder=5
        )

      plt.plot(
          xs
        , hprednom
        , color='red'
        , linewidth=2
        , linestyle="dotted"
        , label="original prediction" + (" (nominal)" if variations else "")
        , zorder=3
        )

      if variations:
        plt.plot(
            xs
          , hpredup
          , linewidth=2
          , color="blue"
          , linestyle="dotted"
          , label="original prediction ($+1$)"
          , zorder=3
          )

        plt.plot(
            xs
          , hpreddown
          , linewidth=2
          , color="green"
          , linestyle="dotted"
          , label="original prediction ($- 1$)"
          , zorder=3
          )

      plt.plot(
          xs
        , htransnom
        , linewidth=2
        , color="red"
        , linestyle="solid"
        , label="transported prediction" + (" (nominal)" if variations else "")
        , zorder=3
        )

      if variations:
        plt.plot(
            xs
          , htransup
          , linewidth=2
          , color="blue"
          , linestyle="solid"
          , label="transported prediction ($+1$)"
          , zorder=3
          )

        plt.plot(
            xs
          , htransdown
          , linewidth=2
          , color="green"
          , linestyle="solid"
          , label="transported prediction ($- 1$)"
          , zorder=3
          )

      if variations:
        fig.legend(loc=(0.15, 0.60))
        plt.ylim(0, max(htarget)*2)
      else:
        fig.legend(loc=(0.15, 0.75))
        plt.ylim(0, max(htarget)*1.5)

      plt.xlim(-1, 1)
      plt.xlabel("$x$")
      plt.ylabel("$counts\\, /\\, bin$")

      return fig


def binned_kldiv(targ, pred):
  htarg = np.histogram(targ, bins=100, range=(-1, 1), density=True)[0]
  hpred = np.histogram(pred, bins=100, range=(-1, 1), density=True)[0]
  return np.sum(htarg * np.log(htarg / hpred))


def of_length(xs, ys):
  n = ys.size()[0]
  return xs[:n]


# zip two tensors along dim=1
def zipt(xs, ys):
  return torch.cat([xs, ys], dim=1)


def detach(obj):
  return obj.squeeze().cpu().detach().numpy()

def get_thetas(n):
  thetas = torch.rand((n, number_nps), device=device)*4 - 2
  for i in range(number_nps):
    if i >= number_thetas:
      thetas[:,i] = 0

  return thetas


def plot_callback(f, g, writer, global_step, outfolder=None):
  thetas = torch.zeros((number_samples_source,1), device=device)

  (sig, bkg) = prediction(thetas)
  prednom = cat(sig, bkg)
  transported = trans(g, sig, of_length(thetas, sig))
  transportednom = cat(transported, bkg)

  writer.add_scalar("kldiv_nom", binned_kldiv(alltarget.detach(), transportednom.detach()), global_step=epoch)


  if number_thetas == 0:
    fig = plot_hist(
        "hist_nom"
      , epoch
      , alltarget.detach().squeeze().numpy()
      , prednom.detach().squeeze().numpy()
      , transportednom.detach().squeeze().numpy()
      )

    if outfolder is not None:
      plt.savefig(outfolder + "/hist_nom.pdf")
      pkl(fig, outfolder + "/hist_nom.pkl")

    writer.add_figure("hist_nom", fig, global_step=epoch)

    fig.clear()
    plt.close()

    # plot the transport vs prediction
    xval = torch.Tensor(np.mgrid[-1:0:100j]).unsqueeze(1)
    xval.requires_grad = True

    yvalnom = trans(g, xval, thetas[:100])

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "transport")

    ax.set_xlim(-1, 0)
    plt.ylim(-0.5, 0.5)
    plt.xlabel("$x$")
    plt.ylabel("$T x$")

    if outfolder is not None:
      plt.savefig(outfolder + "/transport_nom.pdf")
      pkl(fig, outfolder + "/transport_nom.pkl")

    writer.add_figure("transport_nom", fig, global_step = global_step)

    fig.clear()
    plt.close()

    # plot g vs prediction
    yvalnom = g(thetas[:100], xval)
    yvalnom = yvalnom - torch.min(yvalnom)

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "g_func")

    ax.set_xlim(-1, 0)
    plt.xlabel("$x$")
    plt.ylabel("$g(x)$")

    if outfolder is not None:
      plt.savefig(outfolder + "/g_func_nom.pdf")
      pkl(fig, outfolder + "/g_func_nom.pkl")

    writer.add_figure("g_func_nom", fig, global_step = global_step)

    fig.clear()
    plt.close()

    # plot f vs prediction
    yvalnom = f(thetas[:100], xval)
    yvalnom = yvalnom - torch.min(yvalnom)

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "f_func")

    ax.set_xlim(-1, 0)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")

    if outfolder is not None:
      plt.savefig(outfolder + "/f_func_nom.pdf")
      pkl(fig, outfolder + "/f_func_nom.pkl")

    writer.add_figure("f_func_nom", fig, global_step = global_step)

    fig.clear()
    plt.close()

  else:
    for itheta in range(number_thetas):
      thetas = torch.zeros((number_samples_source, number_nps), device=device)

      # get the syst variation for the first theta
      thetas[:,itheta] = 1
      (sig, bkg) = prediction(thetas)
      predup = cat(sig, bkg)
      transported = trans(g, sig, of_length(thetas, sig))
      transportedup = cat(transported, bkg)

      thetas[:,itheta] = -1
      (sig, bkg) = prediction(thetas)
      preddown = cat(sig, bkg)
      transported = trans(g, sig, of_length(thetas, sig))
      transporteddown = cat(transported, bkg)

      writer.add_scalar("kldiv_up_theta%d" % itheta, binned_kldiv(alltarget.detach(), transportedup.detach()), global_step=epoch)
      writer.add_scalar("kldiv_down_theta%d" % itheta, binned_kldiv(alltarget.detach(), transporteddown.detach()), global_step=epoch)


      fig = plot_hist(
          "hist_theta%d" % itheta
        , epoch
        , alltarget.detach().squeeze().numpy()
        , prednom.detach().squeeze().numpy()
        , transportednom.detach().squeeze().numpy()
        , predup.detach().squeeze().numpy()
        , preddown.detach().squeeze().numpy()
        , transportedup.detach().squeeze().numpy()
        , transporteddown.detach().squeeze().numpy()
        )

      if outfolder is not None:
        plt.savefig(outfolder + "/hist_theta%d.pdf" % itheta)
        pkl(fig, outfolder + "/hist_theta%d.pkl" % itheta)

      writer.add_figure("hist_theta%d" % itheta, fig, global_step=epoch)

      fig.clear()
      plt.close()


      xval = torch.Tensor(np.mgrid[-1:0:100j]).unsqueeze(1)
      xval.requires_grad = True

      thetas[:,:] = 0
      yvalnom = trans(g, xval, thetas[:100])

      thetas[:,itheta] = 1
      yvalup = trans(g, xval, thetas[:100])

      thetas[:,itheta] = -1
      yvaldown = trans(g, xval, thetas[:100])

      fig = plt.figure(figsize = (6, 6))
      ax = fig.add_subplot(111)

      ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "nominal")
      ax.plot(detach(xval), detach(yvalup), color = "blue", lw = 2, label = "$\\theta_%d = +1$" % itheta)
      ax.plot(detach(xval), detach(yvaldown), color = "green", lw = 2, label = "$\\theta_%d = - 1$" % itheta)

      fig.legend(loc=(0.15, 0.75))

      ax.set_xlim(-1, 0)
      plt.ylim(-1, 1)
      plt.xlabel("$x$")
      plt.ylabel("$T x$")

      if outfolder is not None:
        plt.savefig(outfolder + "/transport_theta%d.pdf" % itheta)
        pkl(fig, outfolder + "/transport_theta%d.pkl" % itheta)

      writer.add_figure("transport_theta%d" % itheta, fig, global_step = global_step)

      fig.clear()
      plt.close()
    

      thetas[:,:] = 0
      yvalnom = g(thetas[:100], xval)
      yvalnom = yvalnom - torch.min(yvalnom)

      thetas[:,itheta] = 1
      yvalup = g(thetas[:100], xval)
      yvalup = yvalup - torch.min(yvalup)

      thetas[:,itheta] = -1
      yvaldown = g(thetas[:100], xval)
      yvaldown = yvaldown - torch.min(yvaldown)

      fig = plt.figure(figsize = (6, 6))
      ax = fig.add_subplot(111)

      ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "nominal")
      ax.plot(detach(xval), detach(yvalup), color = "blue", lw = 2, label = "$\\theta_%d$ = +1" % itheta)
      ax.plot(detach(xval), detach(yvaldown), color = "green", lw = 2, label = "$\\theta_%d$ = - 1" % itheta)

      fig.legend(loc=(0.55, 0.75))

      ax.set_xlim(-1, 0)
      plt.xlabel("$x$")
      plt.ylabel("$g(x)$")

      if outfolder is not None:
        plt.savefig(outfolder + "/g_func_theta%d.pdf" % itheta)
        pkl(fig, outfolder + "/g_func_theta%d.pkl" % itheta)

      writer.add_figure("g_func_theta%d" % itheta, fig, global_step = global_step)

      fig.clear()
      plt.close()

      # plot f vs prediction
      thetas[:,:] = 0
      yvalnom = f(thetas[:100], xval)
      yvalnom = yvalnom - torch.min(yvalnom)

      thetas[:,itheta] = 1
      yvalup = f(thetas[:100], xval)
      yvalup = yvalup - torch.min(yvalup)

      thetas[:,itheta] = -1
      yvaldown = f(thetas[:100], xval)
      yvaldown = yvaldown - torch.min(yvaldown)

      fig = plt.figure(figsize = (6, 6))
      ax = fig.add_subplot(111)

      ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "nominal")
      ax.plot(detach(xval), detach(yvalup), color = "blue", lw = 2, label = "$\\theta_%d$ = +1" % itheta)
      ax.plot(detach(xval), detach(yvaldown), color = "green", lw = 2, label = "$\\theta_%d$ = - 1" % itheta) 


      fig.legend(loc=(0.55, 0.75))

      ax.set_xlim(-1, 0)
      plt.xlabel("$x$")
      plt.ylabel("$f(x)$")

      if outfolder is not None:
        plt.savefig(outfolder + "/f_func_theta%d.pdf" % itheta)
        pkl(fig, outfolder + "/f_func_theta%d.pkl" % itheta)

      writer.add_figure("f_func_theta%d" % itheta, fig, global_step = global_step)

      fig.clear()
      plt.close()
    
  return


from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
import os
runname = os.path.join(outdir, time_suffix)

# always keep a copy of the steering file
shutil.copyfile(argv[1], outdir + "/" + time_suffix + ".json")
writer = SummaryWriter(runname)


alltarget = targetmodel(number_samples_target)

f_func = \
  ICNN(
      quad_LReLU(0.05, 1)
    , quad_LReLU(0.05, 1)
    #   smooth_leaky_ReLU(0.1)
    # , smooth_leaky_ReLU(0.1)
    , f_nonconvex_shape
    , f_convex_shape
    )
f_func.enforce_convexity()


g_func = \
  ICNN(
      quad_LReLU(0.05, 1)
    , quad_LReLU(0.05, 1)
    #   smooth_leaky_ReLU(0.1)
    # , smooth_leaky_ReLU(0.1)
    , g_nonconvex_shape
    , g_convex_shape
    )
g_func.enforce_convexity()

# build the optimisers
f_func_optim = torch.optim.Adam(f_func.parameters(), lr = lr_f)
g_func_optim = torch.optim.Adam(g_func.parameters(), lr = lr_g)


f_func.to(device)
g_func.to(device)


os.mkdir(runname + ".plots")

for epoch in range(number_epochs):

  print("plotting.")
  plot_callback(f_func, g_func, writer, epoch, outfolder = runname + ".plots")

  writer.add_scalar("g_func-L2reg", g_func.get_convexity_regularisation_term(), global_step=epoch)
  writer.add_scalar("f_func-L2reg", f_func.get_convexity_regularisation_term(), global_step=epoch)
  print("starting epoch %03d" % epoch)

  for batch in range(epoch_size):

    for i in range(g_per_f):

      g_func.zero_grad()

      thetas = get_thetas(batch_size)

      (sig, bkg) = prediction(thetas)
      pred = cat(sig, bkg)

      sig_vals = g_func(of_length(thetas, sig), sig)

      grad_g = cat(grad(sig_vals, sig), bkg)

      lag_g = \
        torch.sum(grad_g * pred, keepdim = True, dim = 1) \
        - f_func(thetas, grad_g)

      # need to maximise the lagrangian
      loss_g = torch.mean(-lag_g) + g_func.get_convexity_regularisation_term() # can use a regulariser to keep it close to convexity ...
      loss_g.backward()
      g_func_optim.step()
      # g_func.enforce_convexity() # ... or enforce convexity explicitly


    f_func_optim.zero_grad()

    target = targetmodel(batch_size)

    thetas = get_thetas(batch_size)

    (sig, bkg) = prediction(thetas)
    pred = cat(sig, bkg)

    sig_vals = g_func(of_length(thetas, sig), sig)

    grad_g = cat(grad(sig_vals, sig), bkg)

    lag_g = \
      torch.sum(grad_g * pred, keepdim = True, dim = 1) \
      - f_func(thetas, grad_g)

    # evaluate the lagrangian for f
    lag_f = f_func(thetas, target) 

    lag_total = lag_g + lag_f
    loss_total = torch.mean(lag_total) + f_func.get_convexity_regularisation_term() 
    loss_total.backward()
    f_func_optim.step()
    # f_func.enforce_convexity()


  print("f-loss lag")
  print(torch.mean(lag_total).item())
  print("f-loss conv")
  print(f_func.get_convexity_regularisation_term().item())

  print("g-loss lag")
  print(torch.mean(-lag_g).item())
  print("g-loss conv")
  print(g_func.get_convexity_regularisation_term().item())

