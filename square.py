import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

print("torch version:", torch.__version__)

device="cuda"
outprefix="square/"

ndata = 1000
nmc = max(100000, 20*ndata)
epochsize = 2**10
nthetas = 4
nepochs = 2**18

ncritic = 5
lr = 5e-5
lam = 0
wgan = True
rmsprop = True
cycle = False # True
lrdecay = 1e-5


alldata = torch.rand((ndata, 1), device=device)
alldata *= alldata
allmc = torch.sort(torch.rand((nmc, 1), device=device))[0]
true = allmc*allmc

def histcurve(bins, fills, default):
  xs = [x for x in bins for _ in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)

def tonp(xs):
  return xs.cpu().detach().numpy()

def combine(seq):
  return torch.cat((torch.mean(seq, 0), torch.std(seq, 0)), axis=0)


name = \
  "batched_%d_datasamps_%d_mcsamps_%d_thetas_%d_ncritic_%.2e_lr_%0.2e_lambda" \
    % (ndata, nmc, nthetas, ncritic, lr, lam)

if cycle:
  name = "onesycle_" + name
elif lrdecay:
  name = name + "%0.2e_lrdecay" % lrdecay

if rmsprop:
  name = "rmsprop_" + name

if wgan:
  name = "wgan_" + name

name += "_%d_epochs" % nepochs

writer = SummaryWriter(outprefix + name)

transport = \
  torch.nn.Sequential(
    torch.nn.Linear(1, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 1+nthetas)
  )


def trans(nn, xs):
  original = xs[:,0:1]
  thetas = xs[:,1:]

  tmp = nn(original)

  cv = tmp[:,0:1] # central value
  var = tmp[:,1:] # eigen variations
  coeffs = var - cv

  corr = torch.bmm(thetas.unsqueeze(1), coeffs.unsqueeze(2))

  return cv + corr.squeeze(2)


critic = \
  torch.nn.Sequential(
    torch.nn.Linear(1, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
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

for epoch in range(nepochs):
  if cycle:
    lr = asched.get_last_lr()[0]
  elif lrdecay:
    lr *= (1-lrdecay)
    
  if epoch > 0 and epoch % 500 == 0:
    print("epoch", epoch)
    print("learning rate:", lr)

    fig = plt.figure(figsize=(6, 6))

    thetas = torch.zeros((nmc, nthetas), device=device)
    mc1 = torch.cat((allmc, thetas), axis=1)
    delta = trans(transport, mc1)
    nom = allmc + delta


    pos = []
    neg = []
    for i in range(nthetas):
      thetas = torch.zeros((nmc, nthetas), device=device)
      thetas[:,i] = 1
      mc1 = torch.cat((allmc, thetas), axis=1)
      transported = trans(transport, mc1) + allmc
      pos.append(tonp(transported))

      thetas = torch.zeros((nmc, nthetas), device=device)
      thetas[:,i] = -1
      mc1 = torch.cat((allmc, thetas), axis=1)
      transported = trans(transport, mc1) + allmc
      neg.append(tonp(transported))


    bins = [-0.05] + [x*0.05 for x in range(21)]
    hs, bins, _ = \
      plt.hist(
        [ allmc.squeeze().detach().cpu().clamp(-0.01, 1.01)
        , alldata.squeeze().detach().cpu().clamp(-0.01, 1.01)
        , nom.squeeze().detach().cpu().clamp(-0.01, 1.01)
        , true.squeeze().detach().cpu().clamp(-0.01, 1.01)
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

    if nthetas == 1:
      hpos = [hpos]
      hneg = [hneg]

    htrans = hs[2]
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
    (xs, htrue) = histcurve(bins, hs[3], 0)
    (xs, hup) = histcurve(bins, hup, 0)
    (xs, hdown) = histcurve(bins, hdown, 0)

    plt.plot(xs, htrans, linewidth=2, color="black", label="transported prediction")
    plt.plot(xs, htrue, linewidth=2, linestyle="dotted", color="blue", label="true target")
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

    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 6)

    writer.add_figure("hist", fig, global_step=epoch)

    fig.clear()

    thetas = torch.randn((nmc, nthetas), device=device)
    mc1 = torch.cat((allmc, thetas), axis=1)
    proposal = trans(transport, mc1) + allmc

    idxs = torch.sort(torch.randint(allmc.size()[0], (1024,)))[0]
    plt.scatter(
        allmc[idxs].squeeze().detach().cpu()
      , proposal[idxs].squeeze().detach().cpu()
      , color="black"
      , s=5
      , alpha=0.1
      , label="proposed transport vector"
    )

    plt.scatter(
        allmc[idxs].squeeze().detach().cpu()
      , true[idxs].squeeze().detach().cpu()
      , color="red"
      , label="true transport vector"
      , alpha=0.75
      , s=5
    )

    plt.scatter(
        allmc[idxs].squeeze().detach().cpu()
      , nom[idxs].squeeze().detach().cpu()
      , color="blue"
      , alpha=0.75
      , s=5
      , label="nominal transport vector"
    )

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.legend()

    writer.add_figure("trans", fig, global_step=epoch)

    res = torch.std(proposal - true)
    writer.add_scalar("tranport_residual", res, global_step=epoch)

    plt.close()

  data = alldata[torch.randint(ndata, (epochsize,), device=device)]
  mc = allmc[torch.randint(nmc, (10*epochsize,), device=device)]
  thetas = torch.randn((10*epochsize, nthetas), device=device)

  critdata = critic(data)
  combdata = combine(critdata)
  real = phi(combdata)

  mc1 = torch.cat((mc, thetas), axis=1)
  transmc = mc + trans(transport, mc1)
  critmc = critic(transmc)
  combmc = combine(critmc)
  fake = phi(combmc)

  if wgan:
    advloss = fake - real
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
  writer.add_scalar('phidata', real.item(), epoch)
  writer.add_scalar('phimc', fake.item(), epoch)
  writer.add_scalar('phidiff', real.item() - fake.item(), epoch)

  toptim.zero_grad()
  aoptim.zero_grad()

  for p in list(critic.parameters()) + list(phi.parameters()):
    p.data.clamp_(-0.1, 0.1)

  writer.add_scalar('learningrate', lr, epoch)

  if cycle or lrdecay:
    asched.step()
    tsched.step()

  if epoch % ncritic != 0:
    continue

  delta = trans(transport, mc1)
  distloss = torch.mean(torch.abs(delta))

  fake = phi(combine(critic(mc + delta)))
  if wgan:
    fakeloss = -fake
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
