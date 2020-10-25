import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sys import argv

print("torch version:", torch.__version__)

device="cuda"
outprefix="square/"


nthetas = int(argv[2])
ndata = int(argv[1])
label = argv[3]
nmc = 2**17
batchsize = 1024
epochsize = 2**11
nepochs = 256

ncritic = 10
lr = 5e-5
lam = 0
wgan = True
optim = "adam"
cycle = False # True
lrdecay = 0 # 1e-3

testthetas = torch.zeros((2*nthetas, 1, nthetas), device=device)
for i in range(nthetas):
  testthetas[i,:,i] = 1
  testthetas[nthetas+i,:,i] = -1

testthetas = testthetas.repeat((1, nmc, 1))


def histcurve(bins, fills, default):
  xs = [x for x in bins for _ in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)

def tonp(xs):
  return xs.cpu().detach().numpy()


def combine(seq):
  # expseq = torch.exp(seq)
  # return torch.cat((torch.mean(expseq, 0), torch.std(expseq, 0)), axis=0)
  return torch.cat((torch.mean(seq, 0), torch.std(seq, 0)), axis=0)


def trans(nn, xs):
  # return torch.zeros((xs.size()[0], 1), device=device)
  # return nn(xs)

  original = xs[:,0:1]
  thetas = xs[:,1:]
  tmp = nn(original)

  centralvalue = tmp[:,0:1]
  variations = tmp[:,1:]
  coeffs = variations - centralvalue.repeat((1, nthetas)) # eigen variations

  corr = torch.bmm(thetas.unsqueeze(1), coeffs.unsqueeze(2))

  return centralvalue + corr.squeeze(2)


alldata = torch.randn((ndata,), device=device) / 4 + 0.5
# alldata.clamp_(0, 1 - 1e-8)
# SQUARE
# alldata *= alldata
alldatacpu = alldata.detach().cpu().squeeze()

allmc = torch.sort(torch.randn((nmc,), device=device))[0] / 4 + 0.5
# allmc.clamp_(0, 1 - 1e-8)
allmccpu = allmc.detach().cpu().squeeze()
# SQUARE
# truecpu = allmccpu*allmccpu
truecpu = allmccpu

alldata = alldata.view((ndata,1))
allmc = allmc.view((nmc,1))



for lab in [str(x) for x in range(0, 5)]:
  transport = \
    torch.nn.Sequential(
      torch.nn.Linear(1, 256)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(256, 1+nthetas)
    )

  # transport = torch.nn.Identity()
  
  # start from ~the identity
  # transport[-1].weight.data *= 0.01
  # transport[-1].bias.data *= 0.01


  critic = \
    torch.nn.Sequential(
      torch.nn.Linear(1, 256)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(256, 256)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(256, 1)
    )


  phi = torch.nn.Identity()

  # phi = \
  #   torch.nn.Sequential(
  #     torch.nn.Linear(2, 64)
  #   , torch.nn.LeakyReLU(inplace=True)
  #   , torch.nn.Linear(64, 64)
  #   , torch.nn.LeakyReLU(inplace=True)
  #   , torch.nn.Linear(64, 64)
  #   , torch.nn.LeakyReLU(inplace=True)
  #   , torch.nn.Linear(64, 1)
  #   )


  transport.to(device)
  critic.to(device)
  phi.to(device)


  if optim == "rmsprop":
    toptim = torch.optim.RMSprop(transport.parameters(), lr=lr)
    aoptim = torch.optim.RMSprop(list(critic.parameters()) + list(phi.parameters()), lr=lr)
  elif optim == "adam":
    toptim = torch.optim.Adam(transport.parameters(), lr=lr)
    aoptim = torch.optim.Adam(list(critic.parameters()) + list(phi.parameters()), lr=lr)
  elif optim == "sgd":
    toptim = torch.optim.SGD(transport.parameters(), lr=lr)
    aoptim = torch.optim.SGD(list(critic.parameters()) + list(phi.parameters()), lr=lr)
  else:
    print("Error: unrecognized optim")
    exit(-1)

  if cycle:
    tsched = torch.optim.lr_scheduler.OneCycleLR(
        toptim
      , lr
      , total_steps=nepochs
      , cycle_momentum=False
      , pct_start=0.1
      )

    asched = torch.optim.lr_scheduler.OneCycleLR(
        aoptim
      , lr
      , total_steps=nepochs
      , cycle_momentum=False
      , pct_start=0.1
      )

  elif lrdecay:
    tsched = torch.optim.lr_scheduler.ExponentialLR(toptim, 1-lrdecay)
    asched = torch.optim.lr_scheduler.ExponentialLR(aoptim, 1-lrdecay)

  starttime = time()
  plottime = 0



  name = \
    "identity_gaussuncert_3lay256_critic_2lay256_transport_%d_datasamps_%d_mcsamps_%d_thetas_%d_ncritic_%.2e_lr_%0.2e_lambda" \
      % (ndata, nmc, nthetas, ncritic, lr, lam)

  if cycle:
    name = "onecycle_" + name
  elif lrdecay:
    name = name + "%0.2e_lrdecay" % lrdecay

  name = optim + "_" + name

  if wgan:
    name = "wgan_" + name

  name += "_%d_epochs" % nepochs

  name += "_%d_epochsize" % epochsize

  name += "_%d_batchsize" % batchsize

  name += "_" + lab

  name += "_" + label

  writer = SummaryWriter(outprefix + name)

  for epoch in range(nepochs):
    if cycle:
      lr = asched.get_last_lr()[0]
    elif lrdecay:
      lr *= (1-lrdecay)

    if epoch > 0 and epoch % 1 == 0:
      startplot = time()
      print("epoch", epoch)

      fig = plt.figure(figsize=(6, 6))

      thetas = torch.zeros((nmc, nthetas), device=device)
      mc1 = torch.cat((allmc, thetas), axis=1)
      nom = allmc + trans(transport, mc1)
      nomcpu = nom.detach().squeeze().cpu()
      del mc1, thetas

      variations = []
      for i in range(nthetas):
        mc1 = torch.cat((allmc, testthetas[i]), axis=1)
        variations.append(
            (trans(transport, mc1) + allmc).detach().squeeze().cpu()
          )

        mc1 = torch.cat((allmc, testthetas[nthetas+i]), axis=1)
        variations.append(
            (trans(transport, mc1) + allmc).detach().squeeze().cpu()
          )
        del mc1

      thetas = torch.randn((nmc, nthetas), device=device)
      mc1 = torch.cat((allmc, thetas), axis=1)
      critmc = critic(trans(transport, mc1) + allmc).detach().squeeze().cpu()
      critdata = critic(alldata).detach().squeeze().cpu()
      del mc1


      _, _, _ = \
        plt.hist(
          [critmc, critdata]
        , label=["critic output MC", "critic output data"]
        , density=True
        )

      del critmc, critdata

      fig.legend()
      writer.add_figure("critichist", fig, global_step=epoch)

      fig.clear()

      bins = [-0.05] + [x*0.05 for x in range(22)]
      hs, bins, _ = \
        plt.hist(
          [ allmccpu.clamp(-0.01, 1.01)
          , alldatacpu.clamp(-0.01, 1.01)
          , nomcpu.clamp(-0.01, 1.01)
          , truecpu.clamp(-0.01, 1.01)
          ]
        , bins=bins
        , density=True
        )

      hvars, _, _ = \
        plt.hist(
          list(map(lambda h: h.clamp(-0.01, 1.01), variations))
        , bins=bins
        , density=True
        )


      fig.clear()

      if nthetas == 0:
        hvars = []

      varcurves = []
      for h in hvars:
        varcurves.append(histcurve(bins, h, 0)[1])

      (xs, hmc) = histcurve(bins, hs[0], 0)
      (xs, hdata) = histcurve(bins, hs[1], 0)
      (xs, htrans) = histcurve(bins, hs[2], 0)
      (xs, htrue) = histcurve(bins, hs[3], 0)

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
        , hmc
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
        , htrue
        , linewidth=2
        , linestyle="dashed"
        , color="red"
        , label="true target"
        , zorder=4
        )

      for curve in varcurves:
        plt.plot(xs, curve, color="green", alpha=0.5, zorder=1)

      fig.legend()

      plt.xlim(-0.1, 1.1)
      plt.ylim(0, 6)

      writer.add_figure("hist", fig, global_step=epoch)

      fig.clear()

      idxs = torch.sort(torch.randint(nmc, (1024,)))[0]

      plt.plot(
          allmccpu[idxs]
        , truecpu[idxs]
        , color="red"
        , label="true transport vector"
        , alpha=0.75
        , fillstyle=None
        , linewidth=2
        , zorder=3
      )

      plt.plot(
          allmccpu[idxs]
        , nomcpu[idxs]
        , color="blue"
        , label="transport vector"
        , alpha=0.75
        , linewidth=2
        , fillstyle=None
        , zorder=2
      )

      res = 0
      uncert = 0
      for variation in variations:
        plt.plot(
            allmccpu[idxs]
          , variation[idxs]
          , color="green"
          , alpha=0.25
          , fillstyle=None
          , zorder=1
        )

        uncert += ((variation - nomcpu)**2).detach()
        res += ((variation - truecpu)**2).detach()

      plt.xlim(-0.1, 1.1)
      plt.ylim(-0.1, 1.1)
      plt.legend()

      writer.add_figure("trans", fig, global_step=epoch)

      writer.add_scalar( \
          "transport_uncertainty"
        , np.sqrt(torch.mean(uncert).item())
        , global_step=epoch
        )

      writer.add_scalar( \
          "transport_residual"
        , np.sqrt(torch.mean(res).item())
        , global_step=epoch
        )


      plt.close()
      plottime += time() - startplot
      totaltime = time() - starttime

      print(\
          "fraction of time plotting: %.2e" \
            % (plottime / (time() - starttime))
        )

    realmeansum = 0
    fakemeansum = 0
    realvarsum = 0
    fakevarsum = 0
    advlosssum = 0

    for batch in range(epochsize):
      data = alldata[torch.randint(ndata, (batchsize,), device=device)]
      mc = allmc[torch.randint(nmc, (batchsize,), device=device)]
      # mc = allmc

      toptim.zero_grad()
      aoptim.zero_grad()

      real = critic(data)
      realmeansum += torch.mean(real).item()
      realvarsum += torch.var(real).item()

      thetas = torch.randn((batchsize, nthetas), device=device)
      mc1 = torch.cat((mc, thetas), axis=1)

      fake = critic(mc + trans(transport, mc1))
      fakemeansum += torch.mean(fake).item()
      fakevarsum += torch.var(fake).item()

      if wgan:
        w2dist = \
          (torch.mean(fake) - torch.mean(real))**2 \
          + (torch.std(fake) - torch.std(real))**2

        advloss = -w2dist

      else:
        advloss = \
          torch.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)) \
          + torch.binary_cross_entropy_with_logits(real, torch.ones_like(real))

      advloss.backward()

      aoptim.step()

      advlosssum += advloss.item()

      toptim.zero_grad()
      aoptim.zero_grad()

      if wgan:
        for p in list(critic.parameters()) + list(phi.parameters()):
          p.data.clamp_(-0.1, 0.1)

      if batch % ncritic != 0:
        continue

      real = critic(data)

      fake = critic(mc + trans(transport, mc1))

      if wgan:
        w2dist = \
          (torch.mean(fake) - torch.mean(real))**2 \
          + (torch.std(fake) - torch.std(real))**2

        transloss = w2dist

      else:
        transloss = torch.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))

      transloss.backward()
      toptim.step()

      toptim.zero_grad()
      aoptim.zero_grad()


    writer.add_scalar('advloss', advlosssum / epochsize, epoch)
    writer.add_scalar('meanphidiff', (realmeansum - fakemeansum) / epochsize , epoch)
    writer.add_scalar('stdphidiff', (np.sqrt(realvarsum) - np.sqrt(fakevarsum)) / epochsize , epoch)

    writer.add_scalar('learningrate', lr, epoch)

    if cycle or lrdecay:
      asched.step()
      tsched.step()
