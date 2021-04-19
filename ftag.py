import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sys import argv

print("torch version:", torch.__version__)

device="cuda"
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

lr_scalar = config["lr_scalar"]
lr_critic = config["lr_critic"]
critic_updates_per_batch = config["critic_updates_per_batch"]

scalar_layers = config["scalar_layers"]
critic_layers = config["critic_layers"]


def targetmodel(n):
  # target is
  # 80% signal centered at 0, width of 0.5
  # 20% background centered at 0.4, width 1
  nsig = int(n * 0.8)
  sig = torch.randn((nsig,), device=device)*0.5
  bkg = torch.randn((n - nsig,), device=device) + 0.4
  return torch.cat([sig, bkg], dim=0).unsqueeze(1)


def signalmodel(n):
  # signal centered at 0.2, width 0.3
  return torch.randn((n,), device=device).unsqueeze(1)*0.3 + 0.2


def backgroundmodel(n):
  # background centered at 0.5, width 1
  return torch.randn((n,), device=device).unsqueeze(1) + 0.5


def prediction(scalar, thetas):
  # theta[0] -> signal fraction
  # theta[1] -> bkg center
  n = thetas.size()[0]
  nom_sigfrac = 0.75
  unc_sigfrac = 0.1
  unc_bkgfrac = nom_sigfrac * unc_sigfrac / (1 - nom_sigfrac)
  unc_bkgcenter = 0.2

  nsig = int(n * nom_sigfrac)
  sig = signalmodel(nsig)
  bkg = backgroundmodel(n - nsig) + unc_bkgcenter*thetas[nsig:,1:2]

  sig.requires_grad = True

  sigweights = 1 + unc_sigfrac*thetas[:nsig,0:1]
  bkgweights = 1 + unc_bkgfrac*thetas[nsig:,0:1]

  if scalar is not None:
    vector = trans(scalar, sig, thetas[:nsig])
    sig = sig + vector

  return \
    ( torch.cat([sig, bkg], dim=0)
    , torch.cat([sigweights, bkgweights], dim=0)
    )


def histcurve(bins, fills, default):
  xs = [x for x in bins for _ in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)



def trans(network, source, thetas):
  phi = network(torch.cat([source, thetas], dim=1))

  grad = \
    torch.autograd.grad(
      phi
    , source
    , grad_outputs = torch.ones_like(phi)
    , create_graph = True
    )[0]

  return grad


def detach(obj):
  return obj.squeeze().cpu().detach().numpy()


alltarget = targetmodel(number_samples_target)
alltargetcpu = detach(alltarget)


def build_network(layers):

  def intersperse(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

    
  network = \
    [ torch.nn.Linear(layers[n], layers[n+1])
      for n in range(len(layers)-1)
    ]

  return torch.nn.Sequential(*list(intersperse(network, torch.nn.Tanh())))


thisscalar = build_network([1+number_thetas] + scalar_layers + [1])
# make this approximately identity
thisscalar[-1].weight.data *= 0.01
thisscalar[-1].bias.data *= 0.01


thiscritic = build_network([1] + critic_layers + [1])


thisscalar.to(device)
thiscritic.to(device)

scalar_optim = torch.optim.RMSprop(thisscalar.parameters(), lr=lr_scalar)
critic_optim = torch.optim.RMSprop(thiscritic.parameters(), lr=lr_critic)


def add_network_plot(network, name, writer, global_step, ylims):

  fig = plt.figure(figsize = (6, 6))
  ax = fig.add_subplot(111)

  # prepare grid for the evaluation of the critic
  xvals = np.linspace(-1.0, 1.0, 51, dtype = np.float32)

  # evaluate critic
  yvals = detach(network(torch.from_numpy(xvals).to(device).unsqueeze(1)))
  yvals = np.reshape(yvals, xvals.shape)

  ax.set_xlim(-1, 1)
  ax.set_ylim(ylims[0], ylims[1])
  
  plt.plot(xvals, yvals)
  
  writer.add_figure(name, fig, global_step = global_step)

  fig.clear()
  plt.close()


# def add_histogram_comparison(targethist, name, writer, global_step):

def plot_callback(scalar, critic, writer, global_step):


  add_network_plot(critic, "critic", writer, global_step, (-1, 1))

  def tmp(xs):
    thetas1 = torch.randn((xs.size()[0], number_thetas), device=device)
    return scalar(torch.cat([xs, thetas1], dim=1))

  add_network_plot(tmp, "scalar", writer, global_step, (-1, 1))

  def tmp(xs):
    thetas1 = torch.zeros((xs.size()[0], number_thetas), device=device)
    return scalar(torch.cat([xs, thetas1], dim=1))

  add_network_plot(tmp, "nominal", writer, global_step, (-1, 1))

  def tmp(xs):
    thetas1 = torch.zeros((xs.size()[0], number_thetas), device=device)
    thetas1[:,0] = 1
    return scalar(torch.cat([xs, thetas1], dim=1))

  add_network_plot(tmp, "variation 1", writer, global_step, (-1, 1))

  def tmp(xs):
    thetas1 = torch.zeros((xs.size()[0], number_thetas), device=device)
    thetas1[:,1] = 1
    return scalar(torch.cat([xs, thetas1], dim=1))

  add_network_plot(tmp, "variation 2", writer, global_step, (-1, 1))

  thetas = torch.zeros((number_samples_source, number_thetas), device=device)
  predorig = prediction(None, thetas)

  pred = prediction(scalar, thetas)

  # draw the first two syst variations
  thetas[:,0] = 1
  pred1 = prediction(scalar, thetas)

  thetas[:,0] = 0
  thetas[:,1] = 1
  pred2 = prediction(scalar, thetas)


  fig = plt.figure(figsize = (6, 6))
  hs, bins, _ = \
    plt.hist(
      [ detach(predorig[0])
      , alltargetcpu
      , detach(pred[0])
      , detach(pred1[0])
      , detach(pred2[0])
      ]
    , bins = np.linspace(-1, 1, 21)
    # , weight = 
    #   [ detach(predorig[1])
    #   , np.ones_like(alltargetcpu)
    #   , detach(pred[1])
    #   , detach(pred1[1])
    #   , detach(pred2[1])
    #   ]
    )

  fig.clear()

  (xs, hsource) = histcurve(bins, hs[0]*normfac, 0)
  (xs, htarget) = histcurve(bins, hs[1], 0)
  (xs, htrans) = histcurve(bins, hs[2]*normfac, 0)
  (xs, htrans1) = histcurve(bins, hs[3]*normfac, 0)
  (xs, htrans2) = histcurve(bins, hs[4]*normfac, 0)

  plt.scatter(
      (bins[:-1] + bins[1:]) / 2.0
    , hs[1]
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
    , label="original prediction"
    , zorder=3
    )


  plt.plot(
      xs
    , htrans
    , linewidth=2
    , color="blue"
    , label="transported prediction"
    , zorder=2
    )

  plt.plot(
      xs
    , htrans1
    , linewidth=2
    , color="green"
    , label="transported prediction (variation 1)"
    , zorder=2
    )

  plt.plot(
      xs
    , htrans2
    , linewidth=2
    , color="red"
    , label="transported prediction (variation 2)"
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


for epoch in range(number_epochs):
  print("starting epoch %03d" % epoch)

  targetmeansum = 0
  transportedmeansum = 0
  criticlosssum = 0

  for batch in range(epoch_size):
    for i in range(critic_updates_per_batch):

      target = \
        alltarget[torch.randint(number_samples_target, (batch_size,), device=device)]

      thetas = torch.randn((batch_size, number_thetas), device=device)


      scalar_optim.zero_grad()
      critic_optim.zero_grad()

      pred, weights = prediction(thisscalar, thetas)

      targetscore = thiscritic(target)
      targetmeansum += torch.mean(targetscore).item()

      transportedscore = thiscritic(pred)
      transportedmeansum += torch.mean(transportedscore).item()

      transportedloss = \
        torch.nn.functional.binary_cross_entropy_with_logits(transportedscore,
            torch.zeros_like(transportedscore), weight=weights, reduction="mean")

      targetloss = \
        torch.nn.functional.binary_cross_entropy_with_logits(targetscore,
            torch.ones_like(targetscore), reduction="mean")

      criticloss = transportedloss + targetloss
      criticlosssum += criticloss.item()

      criticloss.backward()
      critic_optim.step()


    target = \
      alltarget[torch.randint(number_samples_target, (batch_size,), device=device)]

    thetas = torch.randn((batch_size, number_thetas), device=device)

    scalar_optim.zero_grad()
    critic_optim.zero_grad()

    pred, weights = prediction(thisscalar, thetas)

    transportedscore = thiscritic(pred)

    transloss = \
      torch.nn.functional.binary_cross_entropy_with_logits(
        transportedscore
      , torch.ones_like(transportedscore)
      , weight=weights
      )

    transloss.backward()
    scalar_optim.step()


  writer.add_scalar('criticloss', criticlosssum / epoch_size, epoch)
  writer.add_scalar('meancritictarget', targetmeansum / epoch_size , epoch)
  writer.add_scalar('meancritictransported', transportedmeansum / epoch_size , epoch)


  print("plotting.")
  thetas = torch.zeros((number_samples_source, number_thetas), device=device)
  # thetas = torch.zeros((number_samples_source, number_thetas), device=device)
  plot_callback(thisscalar, thiscritic, writer, epoch)