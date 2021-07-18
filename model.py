import torch
from utils import *

device = "cpu"

sigcenter = 0.1
sigwidth1 = 0.1
sigwidth2 = 0.25
sigfrac1 = 0.5
bkgcenter = -0.5
bkgwidth = 1
sigfrac = 0.5

# the number of nps needed to make his module run.
# any nps that aren't input to the networks will be set to zero.
number_nps = 1

def target(n):
  nsig = int(n * sigfrac)
  nsig1 = int(nsig * sigfrac1)
  nsig2 = nsig - nsig1

  sig1 = torch.randn((nsig1, 1), device=device)*sigwidth1 + sigcenter
  sig2 = torch.randn((nsig2, 1), device=device)*sigwidth2 + sigcenter

  sig = cat(sig1, sig2)

  bkg = torch.randn((n - nsig, 1), device=device)*bkgwidth + bkgcenter

  sig.requires_grad = True
  bkg.requires_grad = True

  return cat(sig, bkg)


def signal(thetas):
  n = thetas.size()[0]
  thissigcenter = - 0.5 + 0.1*thetas[:,:1]
  thissigwidth = 0.25 + 0.1*thetas[:,:1]
  thissigwidth = torch.clamp(thissigwidth, 0.05, 99999)
  return torch.randn((n, 1), device=device)*thissigwidth + thissigcenter


def background(thetas):
  n = thetas.size()[0]
  thisbkgcenter = bkgcenter
  thisbkgwidth = bkgwidth
  return torch.randn((n, 1), device=device)*thisbkgwidth + thisbkgcenter


def prediction(thetas):
  n = thetas.size()[0]

  nsig = int(n * sigfrac)
  sig = signal(thetas[:nsig])
  bkg = background(thetas[nsig:])

  sig.requires_grad = True
  bkg.requires_grad = True

  return (sig, bkg)


def get_thetas(n, number_thetas):
  thetas = torch.rand((n, number_nps), device=device)*4 - 2
  for i in range(number_nps):
    if i >= number_thetas:
      thetas[:,i] = 0

  return thetas
