import numpy as np
import matplotlib
import json
import torch
import model
import utils
from ICNN import ICNN, quad_LReLU
import plotutils
from sys import argv


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

f_func = \
  ICNN(
      quad_LReLU(0.05, 1)
    , quad_LReLU(0.05, 1)
    , f_nonconvex_shape
    , f_convex_shape
    )


g_func = \
  ICNN(
      quad_LReLU(0.05, 1)
    , quad_LReLU(0.05, 1)
    , g_nonconvex_shape
    , g_convex_shape
    )

f_func.load(argv[2])
g_func.load(argv[3])

f_func.to(device)
g_func.to(device)

outprefix = argv[4]

plotutils.plot_callback(
    f_func
  , g_func
  , model.prediction
  , model.target
  , number_thetas
  , number_samples_source
  , device
  , outfolder=outprefix
  )