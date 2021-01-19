import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sys import argv

print("torch version:", torch.__version__)

device="cuda"
outprefix="gaussvary/"


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
  # target is centered at 0 and has a width of 0.5
  return torch.randn(n, device=device)*0.5

def sourcemodel(n):
  # source is centered at 0.2 and has a width of 0.3
  return torch.randn(n, device=device)*0.3 + 0.2


def histcurve(bins, fills, default):
  xs = [x for x in bins for _ in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)


def tonp(xs):
  return xs.cpu().detach().numpy()


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
    return obj.detach().cpu().numpy()


alltarget = targetmodel((number_samples_target, 1))
alltargetcpu = detach(alltarget.squeeze())

allsource = sourcemodel((number_samples_source, 1))
allsource.requires_grad = True
allsourcecpu = detach(allsource.squeeze())



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

def plot_callback(scalar, critic, thetas, writer, global_step):

  add_network_plot(critic, "critic", writer, global_step, (-1, 1))

  def tmp(xs):
    thetas1 = torch.randn((xs.size()[0], number_thetas), device=device)
    return scalar(torch.cat([xs, thetas1], dim=1))

  add_network_plot(tmp, "scalar", writer, global_step, (-1, 1))

  def tmp(xs):
    thetas1 = torch.zeros((xs.size()[0], number_thetas), device=device)
    return scalar(torch.cat([xs, thetas1], dim=1))

  add_network_plot(tmp, "nominal", writer, global_step, (-1, 1))

  vector = trans(scalar, allsource, thetas)

  transported = allsource + vector
  transportedcpu = detach(transported.squeeze())


  fig = plt.figure(figsize = (6, 6))
  hs, bins, _ = \
    plt.hist(
      [ allsourcecpu
      , alltargetcpu
      , transportedcpu
      ]
    , bins = np.linspace(-1, 1, 21)
    )

  fig.clear()

  (xs, hsource) = histcurve(bins, hs[0]*normfac, 0)
  (xs, htarget) = histcurve(bins, hs[1], 0)
  (xs, htrans) = histcurve(bins, hs[2]*normfac, 0)

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

      source = \
        allsource[torch.randint(number_samples_source, (batch_size,), device=device)]

      thetas = torch.randn((batch_size, number_thetas), device=device)

      scalar_optim.zero_grad()
      critic_optim.zero_grad()

      targetscore = thiscritic(target)
      targetmeansum += torch.mean(targetscore).item()

      delta = trans(thisscalar, source, thetas)

      transported = source + delta
      transportedscore = thiscritic(transported)
      transportedmeansum += torch.mean(transportedscore).item()

      transportedloss = \
        torch.nn.functional.binary_cross_entropy_with_logits(transportedscore,
            torch.zeros_like(transportedscore), reduction="mean")

      targetloss = \
        torch.nn.functional.binary_cross_entropy_with_logits(targetscore,
            torch.ones_like(targetscore), reduction="mean")

      criticloss = transportedloss + targetloss
      criticlosssum += criticloss.item()

      criticloss.backward()
      critic_optim.step()


    target = \
      alltarget[torch.randint(number_samples_target, (batch_size,), device=device)]

    source = \
      allsource[torch.randint(number_samples_source, (batch_size,), device=device)]

    thetas = torch.randn((batch_size, number_thetas), device=device)

    scalar_optim.zero_grad()
    critic_optim.zero_grad()

    delta = trans(thisscalar, source, thetas)
    transported = source + delta
    transportedscore = thiscritic(transported)

    transloss = \
      torch.nn.functional.binary_cross_entropy_with_logits(
        transportedscore
      , torch.ones_like(transportedscore)
      )

    transloss.backward()
    scalar_optim.step()


  writer.add_scalar('criticloss', criticlosssum / epoch_size, epoch)
  writer.add_scalar('meancritictarget', targetmeansum / epoch_size , epoch)
  writer.add_scalar('meancritictransported', transportedmeansum / epoch_size , epoch)


  print("plotting.")
  thetas = torch.randn((number_samples_source, number_thetas), device=device)
  # thetas = torch.zeros((number_samples_source, number_thetas), device=device)
  plot_callback(thisscalar, thiscritic, thetas, writer, epoch)