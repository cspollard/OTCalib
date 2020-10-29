import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sys import argv

print("torch version:", torch.__version__)

device="cuda"
outprefix="gaussvary/"


nthetas = int(argv[2])
ndata = int(argv[1])
label = argv[3]
nmc = 2**17
batchsize = 64
epochsize = 2**11
nepochs = 256

ncritic = 5
lr = 4
lam = 0
wgan = True
optim = "sgd"
cycle = False # True
lrdecay = 0 # 2e-2
gradnorm = 0.1

normfac = float(ndata) / float(nmc)

testthetas = torch.zeros((2*nthetas, 1, nthetas), device=device)
for i in range(nthetas):
  testthetas[i,:,i] = 1
  testthetas[nthetas+i,:,i] = -1

testthetas = testthetas.repeat((1, nmc, 1))


def varydata(theta1s, theta2s, xs):
  # small shift left/right
  ys = xs + 0.1*theta1s

  # small change in width
  return ys * torch.clamp(theta2s / 5 + 1, 0, 100)



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
  tmp = nn(xs)
  return (tmp[:, 0:1], tmp[:, 1:])


def spread(centralvalue, variations):
  diffs = variations - centralvalue.repeat((1, variations.size()[1]))
  diffs2 = diffs*diffs
  perbatch = torch.sum(diffs2, 1)
  return torch.mean(perbatch)


def app(centralvalue, variations, thetas):
  coeffs = variations - centralvalue.repeat((1, variations.size()[1]))
  correction = torch.bmm(thetas.unsqueeze(1), coeffs.unsqueeze(2))
  return centralvalue + correction.squeeze(2)


def clampit(xs):
  return xs # torch.clamp(xs, -0.01, 1.01)



alldata = torch.randn((ndata,), device=device) / 4 + 0.5
alldata = alldata
alldatacpu = alldata.detach().cpu().squeeze()

allmc = torch.sort(torch.randn((nmc,), device=device))[0] / 4 + 0.5
allmccpu = allmc.detach().cpu().squeeze()
truecpu = allmccpu

alldata = alldata.view((ndata,1))
allmc = allmc.view((nmc,1))



for lab in [str(x) for x in range(0, 5)]:
  transport = \
    torch.nn.Sequential(
      torch.nn.Linear(1, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 1+nthetas)
    )

  # transport = torch.nn.Identity()
  
  # start from ~the identity
  # transport[-1].weight.data *= 0.1
  # transport[-1].bias.data *= 0.1


  critic = \
    torch.nn.Sequential(
      torch.nn.Linear(1, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 1)
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
    "identity_widthuncert_3lay64_critic_3lay64_transport_%d_datasamps_%d_mcsamps_%d_thetas_%d_ncritic_%.2e_lr_%0.2e_lambda" \
      % (ndata, nmc, nthetas, ncritic, lr, lam)

  if cycle:
    name = "onecycle_" + name
  elif lrdecay:
    name = name + "_%0.2e_lrdecay" % lrdecay

  name = optim + "_" + name

  if wgan:
    name = "wgan_" + name

  if gradnorm:
    name = name + "_%0.2e_gradnorm" % gradnorm

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

      (cv, vs) = trans(transport, allmc)

      nom = clampit(allmc + cv)
      nomcpu = nom.detach().squeeze().cpu()


      variations = []
      for i in range(nthetas):
        thetas = testthetas[i]
        variations.append(
            clampit(app(cv, vs, thetas) + allmc).detach().squeeze().cpu()
          )

        thetas = testthetas[nthetas+i]
        variations.append(
            clampit(app(cv, vs, thetas) + allmc).detach().squeeze().cpu()
          )

      thetas = torch.randn((nmc, nthetas), device=device)
      critmc = critic(clampit(app(cv, vs, thetas) + allmc)).detach().squeeze().cpu()
      critdata = critic(clampit(alldata)).detach().squeeze().cpu()


      _, _, _ = \
        plt.hist(
          [critmc, critdata]
        , label=["critic output MC", "critic output data"]
        , density=True
        )

      del thetas, critmc, critdata

      fig.legend()
      writer.add_figure("critichist", fig, global_step=epoch)

      fig.clear()

      bins = [-0.05] + [x*0.05 for x in range(22)]
      hs, bins, _ = \
        plt.hist(
          [ clampit(allmccpu)
          , alldatacpu
          , nomcpu
          , clampit(truecpu)
          ]
        , bins=bins
        )

      hvars, _, _ = \
        plt.hist(
          variations
        , bins=bins
        )


      fig.clear()

      if nthetas == 0:
        hvars = []

      varcurves = []
      for h in hvars:
        varcurves.append(histcurve(bins, h*normfac, 0)[1])

      (xs, hmc) = histcurve(bins, hs[0]*normfac, 0)
      (xs, hdata) = histcurve(bins, hs[1], 0)
      (xs, htrans) = histcurve(bins, hs[2]*normfac, 0)
      (xs, htrue) = histcurve(bins, hs[3]*normfac, 0)

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
      plt.ylim(0, max(hdata)*1.3)

      writer.add_figure("hist", fig, global_step=epoch)

      fig.clear()

      idxs = torch.sort(torch.randint(nmc, (1024,)))[0]

      plt.plot(
          clampit(allmccpu[idxs])
        , clampit(truecpu[idxs])
        , color="red"
        , label="true transport vector"
        , alpha=0.75
        , fillstyle=None
        , linewidth=2
        , zorder=3
      )

      plt.plot(
          clampit(allmccpu[idxs])
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
            clampit(allmccpu[idxs])
          , clampit(variation[idxs])
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
    spreadsum = 0
    advlosssum = 0
    gradlosssum = 0

    for batch in range(epochsize):
      theta1s = torch.zeros((batchsize, 1), device=device)
      theta2s = torch.randn((batchsize, 1), device=device)

      data = \
        clampit(
          varydata(
            theta1s
          , theta2s
          , alldata[torch.randint(ndata, (batchsize,), device=device)]
          )
        )

      mc = allmc[torch.randint(nmc, (batchsize,), device=device)]

      toptim.zero_grad()
      aoptim.zero_grad()

      real = critic(data)
      realmeansum += torch.mean(real).item()

      thetas = torch.randn((batchsize, nthetas), device=device)
      (cv, vs) = trans(transport, mc)
      delta = app(cv, vs, thetas)

      fake = critic(clampit(mc + delta))
      fakemeansum += torch.mean(fake).item()

      if wgan:
        fakeloss = torch.mean(fake)
        realloss = -torch.mean(real)

      else:
        fakeloss = \
          torch.nn.functional.binary_cross_entropy_with_logits(fake,
              torch.zeros_like(fake), reduction="mean")

        realloss = \
          torch.nn.functional.binary_cross_entropy_with_logits(real,
              torch.ones_like(real), reduction="mean")

      advloss = fakeloss + realloss


      # add gradient regularization
      # from https://avg.is.tuebingen.mpg.de/publications/meschedericml2018
      if gradnorm:
        grad_params = torch.autograd.grad(realloss, critic.parameters(), create_graph=True, retain_graph=True)
        norm = 0
        for grad in grad_params:
            norm += grad.pow(2).sum()
        norm = norm.sqrt()

        gradlosssum += gradnorm*norm.item()
        advloss += gradnorm*norm


      advlosssum += advloss.item()
      advloss.backward()

      aoptim.step()


      toptim.zero_grad()
      aoptim.zero_grad()

      if wgan:
        for p in list(critic.parameters()) + list(phi.parameters()):
          p.data.clamp_(-0.1, 0.1)


      if batch % ncritic != 0:
        continue


      thetas = torch.randn((batchsize, nthetas), device=device)
      (cv, vs) = trans(transport, mc)
      delta = app(cv, vs, thetas)
      fake = critic(clampit(mc + delta))

      spr = spread(cv, vs)

      spreadsum += spr

      if wgan:
        transloss = - lam*spr - torch.mean(fake) 

      else:
        transloss = torch.nn.functional.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))

      transloss.backward()
      toptim.step()

      toptim.zero_grad()
      aoptim.zero_grad()


    writer.add_scalar('advloss', advlosssum / epochsize, epoch)
    writer.add_scalar('gradloss', gradlosssum / epochsize, epoch)
    writer.add_scalar('meanphidiff', (realmeansum - fakemeansum) / epochsize , epoch)
    writer.add_scalar('meanrealphi', realmeansum / epochsize , epoch)
    writer.add_scalar('meanfakephi', fakemeansum / epochsize , epoch)
    writer.add_scalar('meanspread', spreadsum / epochsize , epoch)
    writer.add_scalar('learningrate', lr, epoch)

    if cycle or lrdecay:
      asched.step()
      tsched.step()
