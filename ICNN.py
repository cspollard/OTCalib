import torch
import numpy as np


def appL(input, w, b=None):
    return torch.nn.functional.linear(input, w, b)


def weight(x, y):
    ps = torch.nn.parameter.Parameter(torch.empty((y, x)))
    if x > 0 and y > 0:
        torch.nn.init.kaiming_uniform_(ps, a=np.sqrt(5))
    return ps


def bias(x):
    ps = torch.nn.parameter.Parameter(torch.empty(x))
    torch.nn.init.normal_(ps)
    return ps


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

        def id(x):
            return x

        self.g = [convex_activation for i in range(self.nhidden)]
        self.gtilde = [nonconvex_activation for i in range(self.nhidden - 1)]

        self.nonconvex_layersizes = nonconvex_layersizes
        self.convex_layersizes = convex_layersizes


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
        bz = []
        by = []
        bz1 = []

        Wuz = []
        Wuy = []
        Wuz1 = []

        Wuutilde = []
        btilde = []


        for lay in range(self.nhidden):
            Wzz.append(weight(zsize[lay], zsize[lay+1]))
            Wyz.append(weight(ysize, zsize[lay+1]))
            bz.append(bias(zsize[lay]))
            by.append(bias(ysize))
            bz1.append(bias(zsize[lay+1]))

            Wuz.append(weight(usize[lay], zsize[lay]))
            Wuy.append(weight(usize[lay], ysize))
            Wuz1.append(weight(usize[lay], zsize[lay+1]))


        for lay in range(self.nhidden - 1):
            Wuutilde.append(weight(usize[lay], usize[lay+1]))
            btilde.append(bias(usize[lay+1]))


        self.Wzz = Wzz
        self.Wyz = Wyz

        self.bz = bz
        self.by = by
        self.bz1 = bz1

        self.Wuz = Wuz
        self.Wuz1 = Wuz1
        self.Wuy = Wuy
        self.Wuutilde = Wuutilde
        self.btilde = btilde


        # enforce convexivity
        for wzz in self.Wzz:
            for p in wzz:
                p.data.copy_(p.data.abs())

        # the authors fix the weights in the first layer to zero.
        Wzz[0].data.copy_(torch.zeros_like(Wzz[0].data))
        Wzz[0].requires_grad = False


    def parameters(self):
        for p in self.parameterdict().values():
            for q in p:
                yield q

        # return iter([q for p in self.parameterdict().values() for q in p])


    def parameterdict(self):
        return \
            { "Wzz" : self.Wzz
            , "Wyz" : self.Wyz
            , "bz" : self.bz
            , "by" : self.by
            , "bz1" : self.bz1
            , "Wuz" : self.Wuz
            , "Wuz1" : self.Wuz1
            , "Wuy" : self.Wuy
            , "Wuutilde" : self.Wuutilde
            , "btilde" : self.btilde
            }


    def save(self, path):
        return torch.save(self.parameterdict(), path)

    def load(self, path):
        d = torch.load(path)

        self.Wzz = d["Wzz"]
        self.Wyz  = d["Wyz"]
        self.bz = d["bz"]
        self.by = d["by"]
        self.bz1 = d["bz1"]
        self.Wuz = d["Wuz"]
        self.Wuz1 = d["Wuz1"]
        self.Wuy = d["Wuy"]
        self.Wuutilde = d["Wuutilde"]
        self.btilde = d["btilde"]

        return d


    def forward(self, xs, ys):
        ui = xs
        zi = torch.zeros_like(ys)

        for i in range(self.nhidden):
            if self.nonconvex_layersizes[i]:
                zterm = torch.relu(zi * appL(ui, self.Wuz[i], self.bz[i]))
                yterm = ys * appL(ui, self.Wuy[i], self.by[i])
                uterm = appL(ui, self.Wuz1[i], self.bz1[i]) 
            else:
                zterm = torch.relu(zi)
                yterm = ys
                uterm = self.bz1[i]

            zi = \
              self.g[i](
                  appL(zterm, self.Wzz[i])
                + appL(yterm, self.Wyz[i])
                + uterm
              )

            if i < self.nhidden - 1 and self.nonconvex_layersizes[i]:
                # no need to update ui the last time through.
                ui = self.gtilde[i](appL(ui, self.Wuutilde[i], self.btilde[i]))

        return zi


    def enforce_convexity(self):

        # apply param = max(0, param) = relu(param) to all parameters that need to be nonnegative
        for w in self.Wzz:
            w.data.copy_(torch.relu(w.data))


    def get_convexity_regularisation_term(self):

        L2_reg = 0.0

        for w in self.Wzz:
            L2_reg += torch.sum(torch.square(torch.relu(-w.data)))

        return L2_reg



# need a smooth version to make sure the transport potential has a smooth gradient
# NEED TO FIND SOMETHING MORE EFFICIENT HERE
def smooth_leaky_ReLU(a):
    sqrtpi = np.sqrt(np.pi)
    def f(x):
        return 0.5 * ((1 - a) * torch.exp(-torch.square(x)) + sqrtpi * x * (1 + torch.erf(x) + a * torch.erfc(x)))

    return f


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
# with m0 the slope below zero
# and m1 the slope above one
# a quadratic of the form m0*x + (m1 - m0)/2 * x*x interpolates between them
def quad_LReLU(m0, m1):
    a = 0
    b = m0
    c = (m1 - m0) / 2
    pospart = piecewise(1, poly([a, b, c]), poly([-c, m1]))
    return piecewise(0, poly([0, m0]), pospart)