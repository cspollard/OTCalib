import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from time import time

print("torch version:", torch.__version__)

device="cuda"
outprefix="square/"

ndata = 256
nmc = ndata*64
epochsize = ndata
nthetas = 4
nepochs = 2**18

ncritic = 10
lr = 3e-5
lam = 0
wgan = False
rmsprop = True
cycle = False # True
lrdecay = 0 # 1e-5
testthetas = torch.randn((32, 1, nthetas), device=device).repeat((1, nmc, 1))


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
  original = xs[:,0:1]
  thetas = xs[:,1:]

  # SQUARE
  # tmp = nn(original)

  # cv = tmp[:,0:1] # central value
  # var = tmp[:,1:] # eigen variations
  # ENDSQUARE
  coeffs = nn(original)

  corr = torch.bmm(thetas.unsqueeze(1), coeffs.unsqueeze(2))

  return corr.squeeze(2)

  # return nn(xs)


for lab in [str(x) for x in range(1)]:
  transport = \
    torch.nn.Sequential(
      torch.nn.Linear(1, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, nthetas)
    )


  # start from ~the identity
  transport[-1].weight.data *= 0.01
  transport[-1].bias.data *= 0.01


  critic = \
    torch.nn.Sequential(
      torch.nn.Linear(1, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 64)
    , torch.nn.LeakyReLU(inplace=True)
    , torch.nn.Linear(64, 64)
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


  if rmsprop:
    toptim = torch.optim.RMSprop(transport.parameters(), lr=lr)
    aoptim = torch.optim.RMSprop(list(critic.parameters()) + list(phi.parameters()), lr=lr)
  else:
    toptim = torch.optim.SGD(transport.parameters(), lr=lr)
    aoptim = torch.optim.SGD(list(critic.parameters()) + list(phi.parameters()), lr=lr)

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


  alldata = torch.randn((epochsize,), device=device) / 4 + 0.5
  # SQUARE
  # alldata *= alldata
  alldatacpu = alldata.detach().cpu().squeeze()

  allmc = torch.sort(torch.randn((nmc,), device=device))[0] / 4 + 0.5
  allmccpu = allmc.detach().cpu().squeeze()
  # SQUARE
  # truecpu = allmccpu*allmccpu
  truecpu = allmccpu

  alldata = alldata.view((ndata,1))
  allmc = allmc.view((nmc,1))


  name = \
    "identity_repeatthetas_gauss_deeeeeep_onlycritic_%d_datasamps_%d_mcsamps_%d_thetas_%d_ncritic_%.2e_lr_%0.2e_lambda" \
      % (ndata, nmc, nthetas, ncritic, lr, lam)

  if cycle:
    name = "onecycle_" + name
  elif lrdecay:
    name = name + "%0.2e_lrdecay" % lrdecay

  if rmsprop:
    name = "rmsprop_" + name

  if wgan:
    name = "wgan_" + name

  name += "_%d_epochs" % nepochs

  name += "_%d_epochsize" % epochsize

  name += "_" + lab

  writer = SummaryWriter(outprefix + name)

  for epoch in range(nepochs):
    if cycle:
      lr = asched.get_last_lr()[0]
    elif lrdecay:
      lr *= (1-lrdecay)

    if epoch > 0 and epoch % 1000 == 0:
      startplot = time()
      print("epoch", epoch)

      fig = plt.figure(figsize=(6, 6))

      thetas = torch.zeros((nmc, nthetas), device=device)
      mc1 = torch.cat((allmc, thetas), axis=1)
      nom = allmc + trans(transport, mc1)
      nomcpu = nom.detach().squeeze().cpu()

      variations = []
      for i in range(testthetas.size()[0]):
        mc1 = torch.cat((allmc, testthetas[i]), axis=1)
        variations.append(trans(transport, mc1) + allmc)
      del mc1

      mcinputs = torch.cat([nom] + variations, axis=0)
      critmc = critic(mcinputs).detach().squeeze().cpu()
      critdata = critic(data).detach().squeeze().cpu()

      _, _, _ = \
        plt.hist(
          [critmc, critdata]
        , label=["critic output MC", "critic output data"]
        , density=True
        )

      fig.legend()
      writer.add_figure("critichist", fig, global_step=epoch)

      fig.clear()

      _, _, _ = \
        plt.hist(
          [torch.exp(critmc), torch.exp(critdata)]
        , label=["exp(critic output) MC", "exp(critic output) data"]
        , density=True
        )

      fig.legend()
      writer.add_figure("expcritichist", fig, global_step=epoch)

      variations = [v.detach().squeeze().cpu() for v in variations]

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
        plt.plot(xs, curve, color="blue", alpha=0.1, zorder=1)

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
      for variation in variations:
        plt.plot(
            allmccpu[idxs]
          , variation[idxs]
          , color="blue"
          , alpha=0.1
          , fillstyle=None
          , zorder=1
        )

        res += torch.mean((variation - truecpu)**2).item()

      plt.xlim(-0.1, 1.1)
      plt.ylim(-0.1, 1.1)
      plt.legend()

      writer.add_figure("trans", fig, global_step=epoch)

      writer.add_scalar("transport_residual", np.sqrt(res) / len(variations), global_step=epoch)

      plt.close()
      plottime += time() - startplot
      totaltime = time() - starttime

      print(\
          "fraction of time plotting: %.2e" \
            % (plottime / (time() - starttime))
        )

    data = alldata[torch.randint(ndata, (epochsize,), device=device)]
    # mc = allmc[torch.randint(nmc, (20*epochsize,), device=device)]
    # thetas = torch.randn((20*epochsize, nthetas), device=device)
    mc = allmc

    toptim.zero_grad()
    aoptim.zero_grad()

    real = combine(critic(data))

    thetas = torch.randn((1, nthetas), device=device).repeat((nmc, 1))
    mc1 = torch.cat((mc, thetas), axis=1)
    transmc = mc + trans(transport, mc1)
    fake = combine(critic(transmc))

    if wgan:
      advloss = (real[0] - fake[0])**2 + (real[1] - fake[1])**2
    else:
      advloss = \
        torch.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)) \
        + torch.binary_cross_entropy_with_logits(real, torch.ones_like(real))

    advloss.backward()
    aoptim.step()

    writer.add_scalar('advloss', advloss.item(), epoch)
    # writer.add_scalar('meancriticdata', combdata[0].item(), epoch)
    # writer.add_scalar('stdcriticdata', combdata[1].item(), epoch)
    # writer.add_scalar('meancriticmc', combmc[0].item(), epoch)
    # writer.add_scalar('stdcriticmc', combmc[1].item(), epoch)
    writer.add_scalar('phidata', torch.mean(real).item(), epoch)
    writer.add_scalar('phimc', torch.mean(fake).item(), epoch)
    writer.add_scalar('phidiff', torch.mean(real).item() - torch.mean(fake).item(), epoch)

    toptim.zero_grad()
    aoptim.zero_grad()

    if wgan:
      for p in list(critic.parameters()) + list(phi.parameters()):
        p.data.clamp_(-0.1, 0.1)

    writer.add_scalar('learningrate', lr, epoch)

    if cycle or lrdecay:
      asched.step()
      tsched.step()

    if epoch % ncritic != 0:
      continue

    real = combine(critic(data))

    delta = trans(transport, mc1)
    distloss = torch.mean(torch.abs(delta))

    fake = combine(critic(mc + delta))

    if wgan:
      fakeloss = - ((real[0] - fake[0])**2 + (real[1] - fake[1])**2)
    else:
      fakeloss = torch.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))

    transloss = fakeloss + lam * distloss

    transloss.backward()
    toptim.step()

    writer.add_scalar('distloss', distloss.item(), epoch)
    writer.add_scalar('fakeloss', fakeloss.item(), epoch)
    writer.add_scalar('transloss', transloss.item(), epoch)


    toptim.zero_grad()
    aoptim.zero_grad()
