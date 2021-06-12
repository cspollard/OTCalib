import torch
import numpy as np

# see details at https://arxiv.org/abs/1609.07152
class ICNN(torch.nn.Module):

    def __init__(self
        , nonconvex_activation
        , convex_activation
        , nonconvex_layersizes
        , convex_layersizes
        ):

        super(ICNN, self).__init__()

        assert len(nonconvex_layersizes) + 1 == len(convex_layersizes)

        self.nhidden = len(convex_layersizes) - 1

        self.g = [convex_activation for i in range(self.nhidden)]
        self.gtilde = [nonconvex_activation for i in range(self.nhidden - 1)]


        # simple matrix mul
        def W(x, y):
            return torch.nn.Linear(x, y, bias=False)

        # and full linear layer with bias
        def L(x, y):
            return torch.nn.Linear(x, y, bias=True)


        # more-or-less following the nomenclature from
        # arXiv:1609.07152

        # shorthand:
        # zsize = convex layer sizes
        # usize = nonconvex layer sizes
        zsize = convex_layersizes
        usize = nonconvex_layersizes
        ysize = zsize[0]


        Wzz = []
        Wyz = []
        Luz = []
        Luz1 = []
        Luy = []
        Luutilde = []


        for lay in range(self.nhidden):
            Wzz.append(W(zsize[lay], zsize[lay+1]))
            Wyz.append(W(ysize, zsize[lay+1]))
            Luz.append(L(usize[lay], zsize[lay]))
            Luz1.append(L(usize[lay], zsize[lay+1]))
            Luy.append(L(usize[lay], ysize))

        for lay in range(self.nhidden - 1):
            Luutilde.append(L(usize[lay], usize[lay+1]))


        self.Wzz = torch.nn.ModuleList(Wzz)
        self.Wyz = torch.nn.ModuleList(Wyz)
        self.Luz = torch.nn.ModuleList(Luz)
        self.Luz1 = torch.nn.ModuleList(Luz1)
        self.Luy = torch.nn.ModuleList(Luy)
        self.Luutilde = torch.nn.ModuleList(Luutilde)

        # the authors set the weights in the first layer to zero.
        for p in Wzz[0].parameters():
            p.data.copy_(torch.zeros_like(p.data))
            p.requires_grad = False


    def forward(self, xs, ys):
        ui = xs
        zi = torch.zeros_like(ys)

        for i in range(self.nhidden):
            zi = \
              self.g[i](
                  self.Wzz[i](zi * torch.relu(self.Luz[i](ui))) \
                + self.Wyz[i](ys * self.Luy[i](ui)) \
                + self.Luz1[i](ui)
              )

            # no need to update ui the last time through.
            if i < self.nhidden - 1:
                ui = self.gtilde[i](self.Luutilde[i](ui))

        return zi


    def enforce_convexity(self):

        # apply param = max(0, param) = relu(param) to all parameters that need to be nonnegative
        for W in self.Wzz:
            for w in W.parameters():
                w.data.copy_(torch.relu(w.data))


    def get_convexity_regularisation_term(self):

        L2_reg = 0.0

        for W in self.Wzz:
            for w in W.parameters():
                L2_reg += torch.sum(torch.square(torch.relu(-w.data)))

        return L2_reg


# need a smooth version to make sure the transport potential has a smooth gradient
# NEED TO FIND SOMETHING MORE EFFICIENT HERE
def smooth_leaky_ReLU(x, a):
    sqrtpi = np.sqrt(np.pi)
    return 0.5 * ((1 - a) * torch.exp(-torch.square(x)) + sqrtpi * x * (1 + torch.erf(x) + a * torch.erfc(x)))


# a piecewise function x0 the changeover point from f to g
def piecewise(x0, f, g):
    def h(x):
        return \
            torch.heaviside(x0-x, torch.zeros_like(x))*f(x) \
          + torch.heaviside(x-x0, torch.ones_like(x))*g(x)
    return h


# a polynomial of degree len(cs)-1 with cs as the coefficients
def poly(cs):
    def h(x):
        tot = 0
        for i in range(len(cs)):
            tot += cs[i] * x**i
        return tot
    return h


# a quadratically interpolated leaky relu function
# with b the slope below zero
# and b1 the slope above zero
# a quadratic of the form b*x + (b1 - b)/2 * x*x interpolates between t
from math import sqrt
def quad_LReLU(b, b1):
    c = (b1 - b) / 2
    pospart = piecewise(1, poly([0, b, c]), poly([-c, b+2*c]))
    return piecewise(0, poly([0, b]), pospart)
    # this is a * x below zero and a*x + b*x*x above zero
    # return piecewise(0, partial(linear, a), partial(quad, a, b))