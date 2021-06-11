# TODO
# currently imposing convexivity on thetas!!!!

import functools

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sys import argv
from ICNN import ICNN, smooth_leaky_ReLU

print("torch version:", torch.__version__)

device="cpu"
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

number_thetas = config["number_thetas"]
batch_size = config["batch_size"]
epoch_size = config["epoch_size"]
number_epochs = config["number_epochs"]
modeljson = config["model"]

lr_f = config["lr_f"]
lr_g = config["lr_g"]
# critic_updates_per_batch = config["critic_updates_per_batch"]

def cat(xs, ys):
  return torch.cat([xs, ys], dim=0)

def targetmodel(n):
  # target is
  # 80% signal centered at 0, width of 0.25
  # 20% background centered at 0.2, width 0.1
  nsig = int(n * 0.8)
  sig = torch.randn((nsig,), device=device)*0.25 + 0.0
  bkg = torch.randn((n - nsig,), device=device)*0.1 + 0.2
  sig.requires_grad = True
  bkg.requires_grad = True
  return cat(sig, bkg).unsqueeze(1)


def signalmodel(n):
  # signal centered at -0.3, width 0.4
  return torch.randn((n,), device=device).unsqueeze(1)*0.4 - 0.3


def backgroundmodel(n):
  # background centered at 0.2, width 0.1
  return torch.randn((n,), device=device).unsqueeze(1)*0.1 + 0.2


def prediction(thetas):
  # theta[0] -> signal center
  # theta[1] -> signal width
  n = thetas.size()[0]
  sigfrac = 0.8
  unc_sigcenter = 0.4

  nsig = int(n * sigfrac)
  sig = signalmodel(nsig) + unc_sigcenter*thetas[:nsig,0:1]
  bkg = backgroundmodel(n - nsig)

  sig.requires_grad = True
  bkg.requires_grad = True

  return (sig, bkg)



def trans(scalar, pred, thetas):
  vals = scalar(thetas, pred)
  return grad(vals, pred)

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

# compute gradient d(obj)/d(var)
def grad(obj, var):
    g = torch.autograd.grad( \
        obj
      , var
      , grad_outputs = torch.ones_like(obj)
      , create_graph = True
      # , allow_unused = True
      )

    return g[0]



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

def plot_callback(scalar, critic, writer, global_step):

  thetas = torch.zeros((number_samples_source, number_thetas), device=device)
  (sig, bkg) = prediction(thetas)
  transported = trans(g_func, sig, of_length(thetas, sig))
  prednom = cat(sig, bkg)
  transportednom = cat(transported, bkg)

  # draw the syst variations
  thetas[:,0] = 1
  (sig, bkg) = prediction(thetas)
  transported = trans(g_func, sig, of_length(thetas, sig))
  predup = cat(sig, bkg)
  transportedup = cat(transported, bkg)

  thetas[:,0] = -1
  (sig, bkg) = prediction(thetas)
  transported = trans(g_func, sig, of_length(thetas, sig))
  preddown = cat(sig, bkg)
  transporteddown = cat(transported, bkg)

  fig = plt.figure(figsize = (6, 6))
  hs, bins, _ = \
    plt.hist(
      [ alltargetcpu
      , detach(prednom)
      , detach(predup)
      , detach(preddown)
      , detach(transportednom)
      , detach(transportedup)
      , detach(transporteddown)
      ]
    , bins = np.linspace(-1, 1, 21)
    )

  fig.clear()

  (xs, htarget) = histcurve(bins, hs[0], 0)
  (xs, hsource) = histcurve(bins, hs[1]*normfac, 0)
  (xs, hsource1) = histcurve(bins, hs[2]*normfac, 0)
  (xs, hsource2) = histcurve(bins, hs[3]*normfac, 0)
  (xs, htrans) = histcurve(bins, hs[4]*normfac, 0)
  (xs, htrans1) = histcurve(bins, hs[5]*normfac, 0)
  (xs, htrans2) = histcurve(bins, hs[6]*normfac, 0)

  plt.scatter(
      (bins[:-1] + bins[1:]) / 2.0
    , hs[0]
    , label="observed target"
    , color='black'
    , linewidth=0
    , marker='o'
    , zorder=5
    )

  plt.plot(
      xs
    , hsource
    , linewidth=2
    , color="black"
    , linestyle="dotted"
    , label="original prediction (nominal)"
    , zorder=3
    )

  plt.plot(
      xs
    , hsource1
    , linewidth=2
    , color="blue"
    , linestyle="dotted"
    , label="original prediction (up)"
    , zorder=3
    )

  plt.plot(
      xs
    , hsource2
    , linewidth=2
    , color="green"
    , linestyle="dotted"
    , label="original prediction (down)"
    , zorder=3
    )


  plt.plot(
      xs
    , htrans
    , linewidth=2
    , color="black"
    , label="transported prediction (nominal)"
    , zorder=2
    )

  plt.plot(
      xs
    , htrans1
    , linewidth=2
    , color="blue"
    , label="transported prediction (up)"
    , zorder=2
    )

  plt.plot(
      xs
    , htrans2
    , linewidth=2
    , color="green"
    , label="transported prediction (down)"
    , zorder=2
    )

  fig.legend()

  plt.xlim(-1, 1)
  plt.ylim(0, max(htarget)*1.3)

  writer.add_figure("hist", fig, global_step=epoch)

  fig.clear()
  plt.close()


from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
import os
runname = os.path.join(outdir, time_suffix)

# always keep a copy of the steering file
shutil.copyfile(argv[1], outdir + "/" + time_suffix + ".json")
writer = SummaryWriter(runname)


alltarget = targetmodel(number_samples_target)
alltargetcpu = detach(alltarget)

activations = \
  [functools.partial(smooth_leaky_ReLU, a = 0), functools.partial(smooth_leaky_ReLU, a = 0.2)]

f_func = \
  ICNN(
      number_thetas
    , 1
    , torch.nn.ReLU()
    , functools.partial(smooth_leaky_ReLU, a = 0.2)
    , [number_thetas, 16, 16]
    , [1, 16, 16, 16, 1]
    )
f_func.enforce_convexity()

g_func = \
  ICNN(
      number_thetas
    , 1
    , torch.nn.ReLU()
    , functools.partial(smooth_leaky_ReLU, a = 0.2)
    , [number_thetas, 16, 16]
    , [1, 16, 16, 16, 1]
    )
g_func.enforce_convexity()


# build the optimisers
f_func_optim = torch.optim.RMSprop(f_func.parameters(), lr = lr_f)
g_func_optim = torch.optim.RMSprop(g_func.parameters(), lr = lr_g)


f_func.to(device)
g_func.to(device)


for epoch in range(number_epochs):
  print("starting epoch %03d" % epoch)

  targetmeansum = 0
  transportedmeansum = 0
  criticlosssum = 0

  for batch in range(epoch_size):

    g_func.zero_grad()

    thetas = torch.randn((batch_size, number_thetas), device=device)

    (sig, bkg) = prediction(thetas)
    nsig = sig.size()[0]
    pred = cat(sig, bkg)

    sig_vals = g_func(of_length(thetas, sig), sig)
    bkg_vals = 0.5 * bkg * bkg

    grad_g = cat(grad(sig_vals, sig), grad(bkg_vals, bkg))

    # print(grad_g)
    # print(source)
    lag_g = \
      torch.sum(grad_g * pred, keepdim = True, dim = 1) \
      - f_func(thetas, grad_g)

    # need to maximise the lagrangian
    loss_g = torch.mean(-lag_g + g_func.get_convexity_regularisation_term()) # can use a regulariser to keep it close to convexity ...
    loss_g.backward()
    g_func_optim.step()
    #g_func.enforce_convexity() # ... or enforce convexity explicitly


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


  print("plotting.")
  plot_callback(f_func, g_func, writer, epoch)