import functools

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sys import argv
from ICNN import ICNN, quad_LReLU

print("torch version:", torch.__version__)

device="cpu"
outprefix="ftag/"


from sys import argv, stdout
import shutil
import json


from scipy.stats import gaussian_kde

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

batch_size = config["batch_size"]
epoch_size = config["epoch_size"]
number_epochs = config["number_epochs"]
modeljson = config["model"]

lr_f = config["lr_f"]
lr_g = config["lr_g"]
f_per_g = config["f_per_g"]


def cat(xs, ys):
  return torch.cat([xs, ys], dim=0)


sigcenter = 0.0
sigwidth = 0.1
bkgcenter = -0.5
bkgwidth = 0.5
sigfrac = 0.5

def targetmodel(n):
  nsig = int(n * sigfrac)

  sig = torch.randn((nsig, 1), device=device)*sigwidth + sigcenter
  bkg = torch.randn((n - nsig, 1), device=device)*bkgwidth + bkgcenter

  sig.requires_grad = True
  bkg.requires_grad = True

  return cat(sig, bkg)


def signalmodel(thetas):
  n = thetas.size()[0]
  thissigcenter = sigcenter - 0.25 + 0.1*thetas[:,:1]
  thissigwidth = sigwidth + torch.clamp(sigwidth + 0.05*thetas[:,1:], 0.05, 99999)
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
def grad(obj, var):
    g = torch.autograd.grad( \
        obj
      , var
      , grad_outputs = torch.ones_like(obj)
      , create_graph = True
      )

    return g[0]


def trans(scalar, pred, thetas):
  vals = scalar(thetas, pred)
  return grad(vals, pred)


def kde(xs):
  return gaussian_kde(detach(xs))


def grid(ranges):
  return np.mgrid(ranges)


def histcurve(bins, fills, default):
  xs = [x for x in bins for _ in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)


def of_length(xs, ys):
  n = ys.size()[0]
  return xs[:n]


# zip two tensors along dim=1
def zipt(xs, ys):
  return torch.cat([xs, ys], dim=1)


def detach(obj):
  return obj.squeeze().cpu().detach().numpy()



def add_network_plot(network, name, writer, global_step, ylims):

  fig = plt.figure(figsize = (6, 6))
  ax = fig.add_subplot(111)

  # prepare grid for the evaluation of the critic
  xvals = torch.from_numpy(np.linspace(-1.0, 1.0, 51, dtype = np.float32)).to(device).unsqueeze(1)
  xvals.requires_grad = True

  # evaluate critic
  yvals = detach(network(xvals))
  yvals = np.reshape(yvals, xvals.shape)

  ax.set_xlim(-1, 1)
  ax.set_ylim(ylims[0], ylims[1])

  plt.plot(detach(xvals), yvals)

  writer.add_figure(name, fig, global_step = global_step)

  fig.clear()
  plt.close()


# def add_histogram_comparison(targethist, name, writer, global_step):

def plot_callback(g, writer, global_step):

  thetas = torch.zeros((number_samples_source, number_thetas), device=device)
  (sig, bkg) = prediction(thetas)
  prednom = cat(sig, bkg)
  transported = trans(g, sig, of_length(thetas, sig))
  transportednom = cat(transported, bkg)
  g.zero_grad()

  # get the syst variation for the first theta
  thetas[:,0] = 1
  (sig, bkg) = prediction(thetas)
  predup = cat(sig, bkg)
  transported = trans(g, sig, of_length(thetas, sig))
  transportedup = cat(transported, bkg)
  g.zero_grad()

  thetas[:,0] = -1
  (sig, bkg) = prediction(thetas)
  preddown = cat(sig, bkg)
  transported = trans(g, sig, of_length(thetas, sig))
  transporteddown = cat(transported, bkg)
  g.zero_grad()

  fig = plt.figure(figsize = (6, 6))

  kdetarget = kde(alltarget)
  kdeprednom = kde(prednom)
  kdepredup = kde(predup)
  kdepreddown = kde(preddown)
  kdetransportednom = kde(transportednom)
  kdetransportedup = kde(transportedup)
  kdetransporteddown = kde(transporteddown)


  if number_dims == 1:
    xs = np.mgrid[-1:1:100j]

    plt.plot(
        xs
      , kdetarget(xs)
      , label="observed target"
      , color='black'
      , linewidth=2
      , zorder=5
      )

    plt.plot(
        xs
      , kdeprednom(xs)
      , linewidth=2
      , color="red"
      , linestyle="dotted"
      , label="original prediction (nominal)"
      , zorder=3
      )

    plt.plot(
        xs
      , kdepredup(xs)
      , linewidth=2
      , color="blue"
      , linestyle="dotted"
      , label="original prediction (up)"
      , zorder=3
      )

    plt.plot(
        xs
      , kdepreddown(xs)
      , linewidth=2
      , color="green"
      , linestyle="dotted"
      , label="original prediction (down)"
      , zorder=3
      )

    plt.plot(
        xs
      , kdetransportednom(xs)
      , linewidth=2
      , color="red"
      , linestyle="solid"
      , label="transported prediction (nominal)"
      , zorder=3
      )

    plt.plot(
        xs
      , kdetransportedup(xs)
      , linewidth=2
      , color="blue"
      , linestyle="solid"
      , label="transported prediction (up)"
      , zorder=3
      )

    plt.plot(
        xs
      , kdetransporteddown(xs)
      , linewidth=2
      , color="green"
      , linestyle="solid"
      , label="transported prediction (down)"
      , zorder=3
      )


    fig.legend()

    plt.xlim(-1, 1)
    plt.xlabel("$x$")
    plt.ylim(0, max(kdetarget(xs))*1.3)
    plt.ylabel("$p(x)$")

    writer.add_figure("hist", fig, global_step=epoch)
    fig.clear()
    plt.close()
    

    # plot the transport vs prediction
    xval = torch.sort(prednom, dim=0)[0]

    thetas[:,0] = 0
    yvalnom = trans(g, xval, thetas)
    g.zero_grad()

    thetas[:,0] = 1
    yvalup = trans(g, xval, thetas)
    g.zero_grad()

    thetas[:,0] = -1
    yvaldown = trans(g, xval, thetas)
    g.zero_grad()

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "transport (nominal)")
    ax.plot(detach(xval), detach(yvalup), color = "blue", lw = 2, label = "transport (up)")
    ax.plot(detach(xval), detach(yvaldown), color = "green", lw = 2, label = "transport (down)")

    fig.legend()

    ax.set_xlim(-2, 2)
    plt.ylim(-1, 1)
    plt.xlabel("$x$")
    plt.ylabel("$x + \\Delta x$")

    writer.add_figure("transport", fig, global_step = global_step)

    fig.clear()
    plt.close()
  

    # plot g vs prediction
    xval = torch.sort(prednom, dim=0)[0]

    thetas[:,0] = 0
    yvalnom = g(thetas, xval)

    thetas[:,0] = 1
    yvalup = g(thetas, xval)

    thetas[:,0] = -1
    yvaldown = g(thetas, xval)
    g.zero_grad()

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(detach(xval), detach(yvalnom), color = "red", lw = 2, label = "g_func (nominal)")
    ax.plot(detach(xval), detach(yvalup), color = "blue", lw = 2, label = "g_func (up)")
    ax.plot(detach(xval), detach(yvaldown), color = "green", lw = 2, label = "g_func (down)")

    fig.legend()

    ax.set_xlim(-1, 1)
    plt.xlabel("$x$")
    plt.ylabel("$g(x)$")

    writer.add_figure("g_func", fig, global_step = global_step)

    fig.clear()
    plt.close()
  

  elif number_dims == 2:
    pass



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
      quad_LReLU(0.1, 1)
    , quad_LReLU(0.1, 1)
    , f_nonconvex_shape
    , f_convex_shape
    )
f_func.enforce_convexity()

g_func = \
  ICNN(
      quad_LReLU(0.1, 1)
    , quad_LReLU(0.1, 1)
    , g_nonconvex_shape
    , g_convex_shape
    )
g_func.enforce_convexity()


# build the optimisers
f_func_optim = torch.optim.SGD(f_func.parameters(), lr = lr_f)
g_func_optim = torch.optim.SGD(g_func.parameters(), lr = lr_g)


f_func.to(device)
g_func.to(device)


for epoch in range(number_epochs):

  print("plotting.")
  plot_callback(g_func, writer, epoch)

  writer.add_scalar("g_func-L2reg", g_func.get_convexity_regularisation_term(), global_step=epoch)
  writer.add_scalar("f_func-L2reg", f_func.get_convexity_regularisation_term(), global_step=epoch)
  print("starting epoch %03d" % epoch)

  for batch in range(epoch_size):

    for i in range(f_per_g):
      f_func_optim.zero_grad()

      target = targetmodel(batch_size)

      thetas = torch.randn((batch_size, number_thetas), device=device)

      (sig, bkg) = prediction(thetas)
      nsig = sig.size()[0]
      pred = cat(sig, bkg)

      sig_vals = g_func(of_length(thetas, sig), sig)
      bkg_vals = 0.5 * bkg * bkg

      grad_g = cat(grad(sig_vals, sig), grad(bkg_vals, bkg))

      lag_g = \
        torch.sum(grad_g * pred, keepdim = True, dim = 1) \
        - f_func(thetas, grad_g)

      # evaluate the lagrangian for f
      lag_f = f_func(thetas, target)

      lag_total = lag_g + lag_f
      loss_total = torch.mean(lag_total)
      loss_total.backward()
      f_func_optim.step()
      f_func.enforce_convexity()



    g_func.zero_grad()

    thetas = torch.randn((batch_size, number_thetas), device=device)

    (sig, bkg) = prediction(thetas)
    nsig = sig.size()[0]
    pred = cat(sig, bkg)

    sig_vals = g_func(of_length(thetas, sig), sig)
    bkg_vals = 0.5 * bkg * bkg

    grad_g = cat(grad(sig_vals, sig), grad(bkg_vals, bkg))

    lag_g = \
      torch.sum(grad_g * pred, keepdim = True, dim = 1) \
      - f_func(thetas, grad_g)

    # need to maximise the lagrangian
    loss_g = torch.mean(-lag_g) + g_func.get_convexity_regularisation_term() # can use a regulariser to keep it close to convexity ...
    loss_g.backward()
    g_func_optim.step()
    # g_func.enforce_convexity() # ... or enforce convexity explicitly

