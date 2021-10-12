import functools

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sys import argv
from ICNN import ICNN, quad_LReLU, smooth_leaky_ReLU
import pickle
import utils
import plotutils
import model
from sys import argv, stdout
import shutil
import json


print("torch version:", torch.__version__)

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
g_per_f = config["g_per_f"]


grad_clip = config["grad_clip"]

from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
import os
runname = os.path.join(outdir, time_suffix)

# always keep a copy of the steering file
shutil.copyfile(argv[1], outdir + "/" + time_suffix + ".json")
writer = SummaryWriter(runname)


f_func = \
  ICNN(
      quad_LReLU(0.05, 1)
    , quad_LReLU(0.05, 1)
    , f_nonconvex_shape
    , f_convex_shape
    )
f_func.enforce_convexity()


g_func = \
  ICNN(
      quad_LReLU(0.05, 1)
    , quad_LReLU(0.05, 1)
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
  plotutils.plot_callback(f_func, g_func, model.prediction, model.target, number_thetas, 100000, device, writer=writer, epoch=epoch, outfolder = runname + ".plots")

  writer.add_scalar("f_func-L2reg", f_func.get_convexity_regularisation_term(), global_step=epoch)
  writer.add_scalar("g_func-L2reg", g_func.get_convexity_regularisation_term(), global_step=epoch)

  f_func.save(runname + "/ffunc.pth")
  g_func.save(runname + "/gfunc.pth")

  print("starting epoch %03d" % epoch)

  for batch in range(epoch_size):

    for i in range(g_per_f):

      g_func.zero_grad()

      thetas = model.get_thetas(batch_size, number_thetas)

      (sig, bkg) = model.prediction(thetas)
      pred = utils.cat(sig, bkg)

      sig_vals = g_func(utils.of_length(thetas, sig), sig)

      grad_g = utils.cat(utils.grad(sig_vals, sig), bkg)

      lag_g = \
        torch.sum(grad_g * pred, keepdim = True, dim = 1) \
        - f_func(thetas, grad_g)

      # need to maximise the lagrangian
      loss_g = torch.mean(-lag_g) + g_func.get_convexity_regularisation_term() # can use a regulariser to keep it close to convexity ...
      loss_g.backward()

      if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(g_func.parameters(), grad_clip)

      g_func_optim.step()
      # g_func.enforce_convexity() # ... or enforce convexity explicitly


    f_func_optim.zero_grad()

    target = model.target(batch_size)

    thetas = model.get_thetas(batch_size, number_thetas)

    (sig, bkg) = model.prediction(thetas)
    pred = utils.cat(sig, bkg)

    sig_vals = g_func(utils.of_length(thetas, sig), sig)

    grad_g = utils.cat(utils.grad(sig_vals, sig), bkg)

    lag_g = \
      torch.sum(grad_g * pred, keepdim = True, dim = 1) \
      - f_func(thetas, grad_g)

    # evaluate the lagrangian for f
    lag_f = f_func(thetas, target) 

    lag_total = lag_g + lag_f
    loss_total = torch.mean(lag_total) + f_func.get_convexity_regularisation_term() 
    loss_total.backward()


    if grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(f_func.parameters(), grad_clip)

    f_func_optim.step()
    # f_func.enforce_convexity()