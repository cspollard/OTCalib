import torch, os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.special

# for plot output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# where to run
device = 'cpu'
outdir = "gaussian_gradient_2d"

number_samples_target = 1000
number_samples_source = 20 * number_samples_target # MC

# lr_transport = 1e-4
# lr_critic = 3e-3

lr_transport = 8e-4
lr_critic = 4e-3

critic_updates_per_batch = 10
critic_outputs = 1

batch_size = 1024
use_gradient = True

# ---------------------------------------
# utilities for now
# ---------------------------------------

def true_transport_potential_circle(x, y):
    return 0.5 * (np.square(x) + np.square(y))

def generate_source_circle(number_samples, device):
    angles = 2 * np.pi * torch.rand((number_samples,), device = device)
    radii = torch.sqrt(torch.rand((number_samples,), device = device))
    source = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim = 1)
    return source

def generate_target_circle(number_samples, device):
    angles = 2 * np.pi * torch.rand((number_samples,), device = device)
    radii = 2.0 * torch.sqrt(torch.rand((number_samples,), device = device))
    target = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim = 1)
    return target

def true_transport_potential_square(x, y):
    return 0.0 * x

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

# ---------------------------------------
# prepare and cache some information
# ---------------------------------------

# type of data to use
# true_transport_potential = true_transport_potential_circle
# generate_target = generate_target_circle
# generate_source = generate_source_circle

true_transport_potential = true_transport_potential_square
generate_target = generate_target_square
generate_source = generate_source_square

# prepare the data
source_data = generate_source(number_samples_source, device = device)
target_data = generate_target(number_samples_target, device = device)
source_data.requires_grad = True

use_wasserstein = False

# ---------------------------------------
# utilities for later
# ---------------------------------------

def add_source_plot(source_data, global_step):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.hexbin(x = source_data[:, 0], y = source_data[:, 1], mincnt = 1)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    writer.add_figure("source", fig, global_step = global_step)    
    plt.close()

def add_target_plot(observed_data, transported_data, global_step, xlabel = "x", ylabel = "y"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    datahist = ax.hexbin(x = transported_data[:, 0], y = transported_data[:, 1], mincnt = 1, cmap = "plasma")

    scatter = ax.scatter(x = observed_data[:, 0], y = observed_data[:, 1], marker = 'x', c = 'red', alpha = 0.3)
    ax.legend([scatter], ["data"])
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    writer.add_figure("target", fig, global_step = global_step)
    plt.close()    
    
def add_network_plot(network, name, global_step):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    # prepare grid for the evaluation of the network
    xvals = np.linspace(-2.0, 2.0, 50, dtype = np.float32)
    yvals = np.linspace(-2.0, 2.0, 50, dtype = np.float32)

    xgrid, ygrid = np.meshgrid(xvals, yvals)
    vals = np.stack([xgrid.flatten(), ygrid.flatten()], axis = 1)

    # evaluate network
    zvals = detach(network(torch.from_numpy(vals)))
    zvals = np.reshape(zvals, xgrid.shape)
    zvals -= np.min(zvals)

    conts = ax.contourf(xgrid, ygrid, zvals, 100)
    ax.set_aspect(1)
    plt.colorbar(conts)

    plt.tight_layout()
    writer.add_figure(name, fig, global_step = global_step)
    plt.close()

def add_transport_potential_comparison_plot_contours(true_potential, network, name, global_step, contour_values = np.linspace(0.2, 2.0, 15), xlabel = "x", ylabel = "y"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    # prepare grid for the evaluation of the network
    xvals = np.linspace(-1.2, 1.2, 50, dtype = np.float32)
    yvals = np.linspace(-1.2, 1.2, 50, dtype = np.float32)

    xgrid, ygrid = np.meshgrid(xvals, yvals)
    vals = np.stack([xgrid.flatten(), ygrid.flatten()], axis = 1)

    # evaluate network
    zvals_network = detach(network(torch.from_numpy(vals)))
    zvals_network = np.reshape(zvals_network, xgrid.shape)
    zvals_network -= np.min(zvals_network)

    # evaluate true potential
    zvals_true = true_potential(xgrid, ygrid)
    zvals_true -= np.min(zvals_true)

    cont_true = ax.contour(xgrid, ygrid, zvals_true, 10, colors = 'black', linestyles = 'solid', levels = contour_values, linewidths = 2.0)
    cont_network = ax.contour(xgrid, ygrid, zvals_network, 10, colors = 'red', linestyles = 'dashed', levels = contour_values)
    
    cont_network.collections[0].set_label("learned potential")
    cont_true.collections[0].set_label("true potential")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    ax.set_aspect(1)
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), fancybox = True, shadow = True, ncol = 2)

    # plt.tight_layout()
    writer.add_figure(name, fig, global_step = global_step)
    plt.close()    

def add_transport_potential_comparison_plot_radial(true_potential, network, name, global_step, contour_values = np.linspace(0.05, 3.0, 20), xlabel = "x", ylabel = "transport potential"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    # prepare grid for the evaluation of the network
    xvals = np.linspace(0, 1.2, 50, dtype = np.float32)
    yvals = np.zeros_like(xvals)

    vals = np.stack([xvals.flatten(), yvals.flatten()], axis = 1)

    # evaluate network
    zvals_network = detach(network(torch.from_numpy(vals)))
    zvals_network -= np.min(zvals_network)

    # evaluate true potential
    zvals_true = true_potential(xvals, yvals)
    zvals_true -= np.min(zvals_true)

    ax.plot(xvals, zvals_true, color = 'black', linestyle = 'solid', linewidth = 2.0, label = "true potential")
    ax.plot(xvals, zvals_network, color = 'red', linestyle = 'dashed', label = "learned potential")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    
    ax.set_aspect(1)
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), fancybox = True, shadow = True, ncol = 2)

    plt.tight_layout()
    writer.add_figure(name, fig, global_step = global_step)
    plt.close()    

def add_transport_vector_field_plot(network, name, global_step, xlabel = "x", ylabel = "y"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    # prepare grid for the evaluation of the network
    xvals = np.linspace(-1.2, 1.2, 20, dtype = np.float32)
    yvals = np.linspace(-1.2, 1.2, 20, dtype = np.float32)

    xgrid, ygrid = np.meshgrid(xvals, yvals)
    vals = np.stack([xgrid.flatten(), ygrid.flatten()], axis = 1)
    
    # compute transport field
    vals_tensor = torch.from_numpy(vals)
    vals_tensor.requires_grad = True
    
    transported_vals = apply_transport(network, vals_tensor)
    transport_field_tensor = transported_vals - vals_tensor
    transport_field = detach(transport_field_tensor)
    
    ax.quiver(vals[:,0], vals[:,1], transport_field[:,0], transport_field[:,1], color = 'black')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # plt.tight_layout()
    writer.add_figure(name, fig, global_step = global_step)
    plt.close()    
    
def build_layer(number_inputs, number_outputs, activation):
    return torch.nn.Sequential(torch.nn.Linear(number_inputs, number_outputs), activation())
    
def build_fully_connected(number_inputs, number_outputs, number_hidden_layers, units_per_layer, activation):

    hidden_layers = [build_layer(units_per_layer, units_per_layer, activation) for cur in range(number_hidden_layers)]
    layers = [build_layer(number_inputs, units_per_layer, activation)] + hidden_layers + [torch.nn.Linear(units_per_layer, number_outputs)]
    
    return torch.nn.Sequential(*layers)

def build_moments(inputs):
    return torch.cat((torch.mean(inputs, 0), torch.std(inputs, 0)), axis = 0)

def apply_transport(network, source):
    
    output = network(source)

    if use_gradient:
        output = torch.autograd.grad(output, source, grad_outputs = torch.ones_like(output), create_graph = True)[0]
    
    return source + output

# ---------------------------------------
# this is where things happen
# ---------------------------------------

# prepare the logging output
from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
runname = os.path.join(outdir, time_suffix)
writer = SummaryWriter(runname)

# build the transport network and start from somewhere close to the identity
number_outputs = 1 if use_gradient else 2

# transport_network = build_fully_connected(2, number_outputs, number_hidden_layers = 1, units_per_layer = 30,
#                                           activation = torch.nn.Tanh)

transport_network = build_fully_connected(2, number_outputs, number_hidden_layers = 1, units_per_layer = 30,
                                          activation = torch.nn.Tanh)


# transport_network[-1].weight.data *= 0.01
# transport_network[-1].bias.data *= 0.01
transport_network.to(device)

critic = build_fully_connected(2, critic_outputs, number_hidden_layers = 1, units_per_layer = 30,
                               activation = torch.nn.Tanh)

# critic = build_fully_connected(2, critic_outputs, number_hidden_layers = 1, units_per_layer = 30,
#                                activation = torch.nn.Tanh)

critic.to(device)

# build the optimisers
transport_optim = torch.optim.RMSprop(transport_network.parameters(), lr = lr_transport)
adversary_optim = torch.optim.RMSprop(list(critic.parameters()), lr = lr_critic)

for batch in range(50000):

    print("step {}".format(batch))
    
    for cur_adversary_update in range(critic_updates_per_batch):
    
        # sample current batch
        source_data_batch = source_data[torch.randint(low = 0, high = number_samples_source,
                                                      size = (batch_size,), device = device)]
        target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
                                                      size = (batch_size,), device = device)]
        
        # ------------------------------
        # train adversary
        # ------------------------------

        adversary_optim.zero_grad()
        transport_optim.zero_grad()
        
        critic_target = critic(target_data_batch)
        
        transported_source_data_batch = apply_transport(transport_network, source_data_batch)        
        critic_transported_source = critic(transported_source_data_batch)

        if use_wasserstein:
            # Wasserstein loss for adversary
            adv_loss = torch.mean(critic_transported_source) - torch.mean(critic_target)
        else:
            adv_loss = torch.mean(torch.binary_cross_entropy_with_logits(critic_transported_source, torch.zeros_like(critic_transported_source)) + torch.binary_cross_entropy_with_logits(critic_target, torch.ones_like(critic_target)))
            
        # critic update
        adv_loss.backward()
        adversary_optim.step()

        if use_wasserstein:
            for p in list(critic.parameters()):
                p.data.clamp_(-0.1, 0.1)
    
    # ------------------------------
    # train transport network
    # ------------------------------

    # sample current batch
    source_data_batch = source_data[torch.randint(low = 0, high = number_samples_source,
                                                  size = (batch_size,), device = device)]
    target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
                                                  size = (batch_size,), device = device)]    
    
    transport_optim.zero_grad()
    adversary_optim.zero_grad()

    # minimise the Wasserstein loss between transported source and bootstrapped target
    transported_source_data_batch = apply_transport(transport_network, source_data_batch)
    critic_output = critic(transported_source_data_batch)
    
    if use_wasserstein:
        transport_loss = -torch.mean(critic_output)
    else:
        transport_loss = torch.mean(torch.binary_cross_entropy_with_logits(critic_output, torch.ones_like(critic_output)))

    transport_loss.backward()
    transport_optim.step()

    if not batch % 10:
    
        # ------------------------------
        # make diagnostic plots
        # ------------------------------
        
        transported_data_nominal = detach(apply_transport(transport_network, source_data))

        add_network_plot(critic, name = "critic", global_step = batch)
        add_transport_vector_field_plot(transport_network, "transport_field", global_step = batch)
        
        if use_gradient:
            add_network_plot(transport_network, name = "transport_potential", global_step = batch)
            add_transport_potential_comparison_plot_contours(true_transport_potential, transport_network, name = "transport_potential_comparison_contours", global_step = batch)
            add_transport_potential_comparison_plot_radial(true_transport_potential, transport_network, name = "transport_potential_comparison_radial", global_step = batch)
            
        add_source_plot(detach(source_data), global_step = batch)
        add_target_plot(observed_data = target_data, transported_data = transported_data_nominal, global_step = batch)
            
writer.close()
print("done")
