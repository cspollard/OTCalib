import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

print("torch version:", torch.__version__)

device="cuda"
outprefix="square/"

ndata = 100000
nmc = 20*ndata
epochsize = 100000 # 2**12
nthetas = 1
nepochs = 2**18

ncritic = 10
lr = 5e-4
lam = 0
wgan = True
rmsprop = True
cycle = False # True
lrdecay = 0 # 1e-5
testthetas = torch.randn((10, 1, nthetas), device=device).repeat((1, nmc, 1))


alldata = torch.rand((ndata,), device=device)
alldata *= alldata
alldatacpu = alldata.detach().cpu().squeeze()

allmc = torch.sort(torch.rand((nmc,), device=device))[0]
allmccpu = allmc.detach().cpu().squeeze()
truecpu = allmccpu*allmccpu

alldata = alldata.view((ndata,1))
allmc = allmc.view((nmc,1))

def histcurve(bins, fills, default):
  xs = [x for x in bins for _ in (1, 2)]
  ys = [default] + [fill for fill in fills for i in (1, 2)] + [default]

  return (xs, ys)

def tonp(xs):
  return xs.cpu().detach().numpy()

def combine(seq):
  return torch.cat((torch.mean(seq, 0), torch.std(seq, 0)), axis=0)


name = \
  "deeperstiller_nongauss_batched_%d_datasamps_%d_mcsamps_%d_thetas_%d_ncritic_%.2e_lr_%0.2e_lambda" \
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

writer = SummaryWriter(outprefix + name)

transport = \
  torch.nn.Sequential(
    torch.nn.Linear(1+nthetas, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 1)
  )


def trans(nn, xs):
  # original = xs[:,0:1]
  # thetas = xs[:,1:]

  # tmp = nn(original)

  # cv = tmp[:,0:1] # central value
  # var = tmp[:,1:] # eigen variations
  # coeffs = var - cv

  # corr = torch.bmm(thetas.unsqueeze(1), coeffs.unsqueeze(2))

  # return cv + corr.squeeze(2)

  return nn(xs)


critic = \
  torch.nn.Sequential(
    torch.nn.Linear(1, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(32, 32)
  )


phi = \
  torch.nn.Sequential(
    torch.nn.Linear(64, 64)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(64, 64)
  , torch.nn.LeakyReLU(inplace=True)
  , torch.nn.Linear(64, 64)
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

  if epoch > 0 and epoch % 10 == 0:
    print("epoch", epoch)

    fig = plt.figure(figsize=(6, 6))

    thetas = torch.zeros((nmc, nthetas), device=device)
    mc1 = torch.cat((allmc, thetas), axis=1)
    nomcpu = (allmc + trans(transport, mc1)).detach().cpu().squeeze()


    variations = []
    for i in range(testthetas.size()[0]):
      mc1 = torch.cat((allmc, testthetas[i]), axis=1)
      transported = (trans(transport, mc1) + allmc).detach().cpu().squeeze()
      variations.append(transported)


    bins = [-0.05] + [x*0.05 for x in range(21)]
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
        list(map(lambda h: h.clamp(-0.01, 1.01).numpy(), variations))
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
      , color="red"
      , linestyle="dashed"
      , label="original prediction"
      , zorder=3
      )


    plt.plot(
        xs
      , htrans
      , linewidth=2
      , color="black"
      , label="transported prediction"
      , zorder=2
      )

    plt.plot(
        xs
      , htrue
      , linewidth=2
      , linestyle="dotted"
      , color="blue"
      , label="true target"
      , zorder=4
      )

    for curve in varcurves:
      plt.plot(xs, curve, color="gray", alpha=0.25, zorder=1)

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
      , label="nominal transport vector"
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
        , color="gray"
        , alpha=0.25
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

  data = alldata[torch.randint(ndata, (epochsize,), device=device)]
  # mc = allmc[torch.randint(nmc, (20*epochsize,), device=device)]
  # thetas = torch.randn((20*epochsize, nthetas), device=device)
  mc = allmc
  thetas = torch.randn((nmc, nthetas), device=device)

  real = phi(combine(critic(data)))

  mc1 = torch.cat((mc, thetas), axis=1)
  transmc = mc + trans(transport, mc1)
  fake = phi(combine(critic(transmc)))

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
