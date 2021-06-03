import torch, os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.special
import functools

# for plot output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# where to run
device = 'cpu'
outdir = "duality_2d"

number_samples_target = 5000
number_samples_source = 20 * number_samples_target # MC

number_dims = 2 # this is a 2d toy problem

lr_f_func = 1e-3
lr_g_func = 1e-3

batch_size = 1000
critic_updates_per_batch = 10

def generate_source_square(number_samples, device):
    xvals = 2 * torch.rand((number_samples,), device = device) - 1
    yvals = 2 * torch.rand((number_samples,), device = device) - 1
    source = torch.stack([xvals, yvals], dim = 1)
    return source

def generate_target_square(number_samples, device):
    angle = np.pi / 4
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    
    xvals = 2 * torch.rand((number_samples,), device = device) - 1
    yvals = 2 * torch.rand((number_samples,), device = device) - 1
    target = torch.stack([cosval * xvals + sinval * yvals, cosval * yvals - sinval * xvals], dim = 1)
    return target

def get_bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

def detach(obj):
    return obj.detach().numpy()

# see details at https://arxiv.org/abs/1609.07152
class ICNN(torch.nn.Module):

    def __init__(self, number_inputs, number_hidden_layers, units_per_layer, activations):

        super(ICNN, self).__init__()
        
        self.activations = activations
        
        # build all the operations
        self.bypass_ops = torch.nn.ModuleList([torch.nn.Linear(number_inputs, units_per_layer, bias = True) for cur in range(number_hidden_layers - 1)])
        self.bypass_ops += [torch.nn.Linear(number_inputs, 1, bias = True)] # the last one is special because it needs to produce a scalar output
        
        self.convex_ops = torch.nn.ModuleList([torch.nn.Linear(units_per_layer, units_per_layer, bias = False) for cur in range(number_hidden_layers - 2)])
        self.convex_ops += [torch.nn.Linear(units_per_layer, 1, bias = False)]

    def forward(self, intensor):

        outtensor = self.bypass_ops[0](intensor)

        assert len(self.bypass_ops[1:]) == len(self.convex_ops)
        assert len(self.convex_ops) == len(self.activations)
        
        for cur_bypass_op, cur_convex_op, cur_activation in zip(self.bypass_ops[1:], self.convex_ops, self.activations):

            # apply the activation to the output of the previous layer
            outtensor = cur_activation(outtensor)            

            # apply the next layer
            outtensor = cur_convex_op(outtensor) + cur_bypass_op(intensor)
            
        return outtensor

    def enforce_convexity(self):
        
        # apply param = max(0, param) = relu(param) to all parameters that need to be nonnegative
        for cur_convex_op in self.convex_ops:
            for cur_param in cur_convex_op.parameters():
                cur_param.data.copy_(torch.relu(cur_param.data))             

    def get_convexity_regularisation_term(self):

        L2_reg = 0.0
        
        for cur_convex_op in self.convex_ops:
            for cur_param in cur_convex_op.parameters():
                L2_reg += torch.sum(torch.square(torch.relu(-cur_param.data)))

        return L2_reg
                        
# compute gradient d(obj)/d(var)
def grad(obj, var):
    return torch.autograd.grad(obj, var, grad_outputs = torch.ones_like(obj), create_graph = True)[0]
                
# ---------------------------------------
# prepare and cache some information
# ---------------------------------------
generate_target = generate_target_square
generate_source = generate_source_square

# prepare the data
source_data = generate_source(number_samples_source, device = device)
target_data = generate_target(number_samples_target, device = device)
source_data.requires_grad = True

def add_target_plot(observed_data, transported_data, global_step, xlabel = "x", ylabel = "y"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    scatter = ax.scatter(x = observed_data[:, 0], y = observed_data[:, 1], marker = 'x', c = 'red', alpha = 0.3)
    ax.legend([scatter], ["data"])
    
    datahist = ax.hexbin(x = transported_data[:, 0], y = transported_data[:, 1], mincnt = 1, cmap = "plasma")
    
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    writer.add_figure("target", fig, global_step = global_step)
    plt.close()    
    
def add_network_plot(network, name, global_step, xlabel = "x", ylabel = "y"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    # prepare grid for the evaluation of the network
    xvals = np.linspace(-1.5, 1.5, 50, dtype = np.float32)
    yvals = np.linspace(-1.5, 1.5, 50, dtype = np.float32)

    xgrid, ygrid = np.meshgrid(xvals, yvals)
    vals = np.stack([xgrid.flatten(), ygrid.flatten()], axis = 1)

    # evaluate network
    zvals = detach(network(torch.from_numpy(vals)))
    zvals = np.reshape(zvals, xgrid.shape)
    zvals -= np.min(zvals)

    conts = ax.contourf(xgrid, ygrid, zvals, 100, cmap = "inferno")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_aspect(1)
    plt.colorbar(conts, cax = cax)

    plt.tight_layout()
    writer.add_figure(name, fig, global_step = global_step)
    plt.close()

def add_transport_vector_field_plot(network, name, global_step, xlabel = "x", ylabel = "y"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    # prepare grid for the evaluation of the network
    xvals = np.linspace(-1.5, 1.5, 20, dtype = np.float32)
    yvals = np.linspace(-1.5, 1.5, 20, dtype = np.float32)

    xgrid, ygrid = np.meshgrid(xvals, yvals)
    vals = np.stack([xgrid.flatten(), ygrid.flatten()], axis = 1)
    
    # compute transport field
    vals_tensor = torch.from_numpy(vals)
    vals_tensor.requires_grad = True

    transport_field_tensor = apply_transport(network, vals_tensor) - vals_tensor
    transport_field = detach(transport_field_tensor)
        
    ax.quiver(vals[:,0], vals[:,1], transport_field[:,0], transport_field[:,1], color = 'black')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # plt.tight_layout()
    writer.add_figure(name, fig, global_step = global_step)
    plt.close()    

def apply_transport(g_func_network, source):

    g_vals = g_func_network(source)
    transported = grad(g_vals, source)

    return transported

# ---------------------------------------
# this is where things happen
# ---------------------------------------

# prepare the logging output
from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
runname = os.path.join(outdir, time_suffix)
writer = SummaryWriter(runname)

# need a smooth version to make sure the transport potential has a smooth gradient
# NEED TO FIND SOMETHING MORE EFFICIENT HERE
def smooth_leaky_ReLU(x, a):
    sqrtpi = np.sqrt(np.pi)
    return 0.5 * ((a - 1) * torch.exp(-torch.square(x)) + sqrtpi * x * (1 + torch.erf(x) + a * torch.erfc(x)))

# need to have a convex and non-decreasing activation function
#activations = [lambda val: torch.pow(torch.relu(val), 2), torch.nn.LeakyReLU(0.2)]
activations = [functools.partial(smooth_leaky_ReLU, a = 0), functools.partial(smooth_leaky_ReLU, a = 0.2)]

# build the two convex functions that form the Kantorovich potential pair
f_func = ICNN(number_inputs = number_dims, number_hidden_layers = 3, units_per_layer = 50,
              activations = activations)
f_func.enforce_convexity()

g_func = ICNN(number_inputs = number_dims, number_hidden_layers = 3, units_per_layer = 50,
              activations = activations)
g_func.enforce_convexity()

# build the optimisers
f_func_optim = torch.optim.RMSprop(f_func.parameters(), lr = lr_f_func)
g_func_optim = torch.optim.RMSprop(g_func.parameters(), lr = lr_g_func)

# training loop
for batch in range(50000):

    # ------------------------------
    # make diagnostic plots
    # ------------------------------
    if not batch % 10:

        transported_data_nominal = detach(apply_transport(g_func, source_data))

        add_target_plot(observed_data = target_data, transported_data = transported_data_nominal, global_step = batch)
        add_transport_vector_field_plot(g_func, "transport_field", global_step = batch)
        
        add_network_plot(g_func, "g_func", global_step = batch)
        add_network_plot(f_func, "f_func", global_step = batch)
    
    for cur_g_update in range(critic_updates_per_batch):

        g_func_optim.zero_grad()
        
        # sample current batch
        source_data_batch = source_data[torch.randint(low = 0, high = number_samples_source,
                                                      size = (batch_size,), device = device)]
        
        # evaluate the lagrangian for g
        vals_g = g_func(source_data_batch)
        grad_g = grad(vals_g, source_data_batch)
        lag_g = torch.sum(grad_g * source_data_batch, keepdim = True, dim = 1) - f_func(grad_g)
        
        # need to maximise the lagrangian
        loss_g = torch.mean(-lag_g + g_func.get_convexity_regularisation_term()) # can use a regulariser to keep it close to convexity ...
        loss_g.backward()
        g_func_optim.step()
        #g_func.enforce_convexity() # ... or enforce convexity explicitly

    f_func_optim.zero_grad()
        
    source_data_batch = source_data[torch.randint(low = 0, high = number_samples_source,
                                                  size = (batch_size,), device = device)]
    target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
                                                  size = (batch_size,), device = device)]    

    # evaluate the lagrangian for g
    vals_g = g_func(source_data_batch)
    grad_g = grad(vals_g, source_data_batch)
    lag_g = torch.sum(grad_g * source_data_batch, keepdim = True, dim = 1) - f_func(grad_g)

    # evaluate the lagrangian for f
    lag_f = f_func(target_data_batch)

    lag_total = lag_g + lag_f
    loss_total = torch.mean(lag_total)
    loss_total.backward()
    f_func_optim.step()
    f_func.enforce_convexity()    
        
writer.close()
print("done")

