import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

print("torch version:", torch.__version__)

device="cuda"
outprefix="square/"

ndata = 1000
nmc = 100000
nthetas = 2
nepochs = 2**18

alldata = torch.rand((ndata, 1), device=device)
alldata *= alldata
mc = torch.rand((nmc, 1), device=device)

def histcurve(bins, fills, default):
  xs = [x for x in bins for i in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)

def clip(x):
  return np.clip(rangex[0], rangex[1], x)

def tonp(xs):
  return xs.cpu().detach().numpy()

def combine(seq):
  return torch.cat((torch.mean(seq, 0), torch.std(seq, 0)), axis=0)


name = "%d_datasamps_%d_mcsamps_%d_thetas" % (ndata, nmc, nthetas)

writer = SummaryWriter(outprefix + name)

transport = \
  torch.nn.Sequential(
    torch.nn.Linear(1+nthetas, 256)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(256, 256)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(256, 1)
  )

critic = \
  torch.nn.Sequential(
    torch.nn.Linear(1, 512)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(512, 32)
  )


phi = \
  torch.nn.Sequential(
    torch.nn.Linear(64, 64)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(64, 1)
  )

transport.to(device)
critic.to(device)
phi.to(device)


toptim = torch.optim.RMSprop(transport.parameters(), lr=5e-4)
aoptim = torch.optim.RMSprop(list(critic.parameters()) + list(phi.parameters()), lr=5e-4)

for epoch in range(nepochs):
  if epoch > 0 and epoch % 1000 == 0:
    print("epoch", epoch)

    fig = plt.figure(figsize=(6, 6))

    thetas = torch.zeros_like(thetas)
    mc1 = torch.cat((mc, thetas), axis=1)
    trans = transport(mc1)
    nom = mc + trans


    pos = []
    neg = []
    for i in range(nthetas):
      thetas = torch.zeros_like(thetas)
      thetas[:,i] = 1
      mc1 = torch.cat((mc, thetas), axis=1)
      transported = transport(mc1) + mc
      pos.append(tonp(transported))

      thetas = torch.zeros_like(thetas)
      thetas[:,i] = -1
      mc1 = torch.cat((mc, thetas), axis=1)
      transported = transport(mc1) + mc
      neg.append(tonp(transported))


    bins = [-0.05] + [x*0.05 for x in range(21)]
    hs, bins, _ = \
      plt.hist(
        [ mc.squeeze().detach().cpu().clamp(-0.01, 1.01)
        , alldata.squeeze().detach().cpu().clamp(-0.01, 1.01)
        , nom.squeeze().detach().cpu().clamp(-0.01, 1.01)
        ]
      , bins=bins
      , density=True
      )

    hpos, _, _ = \
      plt.hist(
        list(map(lambda h: np.clip(-0.01, 1.01, h).squeeze(), pos))
      , bins=bins
      , density=True
      )

    hneg, _, _ = \
      plt.hist(
        list(map(lambda h: np.clip(-0.01, 1.01, h).squeeze(), neg))
      , bins=bins
      , density=True
      )

    fig.clear()

    htrans = hs[1]
    herr2 = np.zeros_like(htrans)
    for i in range(len(hpos)):
      hup = hpos[i]
      hdown = hneg[i]

      herr2 += ((np.abs(hup - htrans) + np.abs(hdown - htrans)) / 2.0)**2

    herr = np.sqrt(herr2)
    hup = htrans + herr
    hdown = htrans - herr

    (xs, hmc) = histcurve(bins, hs[0], 0)
    (xs, hdata) = histcurve(bins, hs[1], 0)
    (xs, htrans) = histcurve(bins, hs[2], 0)
    (xs, hup) = histcurve(bins, hup, 0)
    (xs, hdown) = histcurve(bins, hdown, 0)

    plt.plot(xs, htrans, linewidth=2, color="black", label="transported prediction")
    plt.fill_between(xs, hup, hdown, color="gray", alpha=0.5, label="transported uncertainty")

    plt.plot(
        xs
      , hmc
      , linewidth=2
      , color="red"
      , linestyle="dashed"
      , label="original prediction"
      )

    plt.scatter(
        (bins[:-1] + bins[1:]) / 2.0
      , hs[1]
      , label="target"
      , color='black'
      , linewidth=0
      , marker='o'
      )


    fig.legend()

    writer.add_figure("hist", fig, global_step=epoch)

    fig.clear()

    plt.scatter(
        mc.squeeze().detach().cpu().clamp(-0.01, 1.01)
      , trans.squeeze().detach().cpu()
    )

    writer.add_figure("trans", fig, global_step=epoch)

    plt.close()

  data = alldata[torch.randint(ndata, (ndata,), device=device)]
  thetas = torch.randn((nmc, nthetas), device=device)

  critdata = critic(data)
  combdata = combine(critdata)
  real = phi(combdata)

  mc1 = torch.cat((mc, thetas), axis=1)
  transmc = mc + transport(mc1)
  critmc = critic(transmc)
  combmc = combine(critmc)
  fake = phi(combmc)

  # advloss = \
  #   torch.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake)) \
  #   + torch.binary_cross_entropy_with_logits(real, torch.ones_like(real))
  advloss = fake - real

  advloss.backward()
  aoptim.step()

  writer.add_scalar('advloss', advloss.item(), epoch)
  # writer.add_scalar('meancriticdata', combdata[0].item(), epoch)
  # writer.add_scalar('stdcriticdata', combdata[1].item(), epoch)
  # writer.add_scalar('meancriticmc', combmc[0].item(), epoch)
  # writer.add_scalar('stdcriticmc', combmc[1].item(), epoch)
  writer.add_scalar('phidata', real.item(), epoch)
  writer.add_scalar('phimc', fake.item(), epoch)
  writer.add_scalar('phidiff', real.item() - fake.item(), epoch)

  toptim.zero_grad()
  aoptim.zero_grad()

  if epoch % 5 != 0:
    continue

  fake = phi(combine(critic(mc + transport(mc1))))

  # transloss = \
  #   -torch.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))

  transloss = -fake

  transloss.backward()
  toptim.step()

  toptim.zero_grad()
  aoptim.zero_grad()