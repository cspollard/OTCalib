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
outdir = "gaussian_gradient_1d"

number_samples_target = 5000                       # data
number_samples_source = 20 * number_samples_target # MC

lr_transport = 1e-4
lr_critic = 3e-3
critic_updates_per_batch = 10
critic_outputs = 1

batch_size = 1024
use_gradient = True

# ---------------------------------------
# utilities for now
# ---------------------------------------

def true_transport_potential(x):
    return 0.5 * np.square(x) + 2 * x

def true_transport_function(x):
    return 2 * x + 2

def generate_source(number_samples, device):
    source = torch.randn((number_samples,), device = device) - 1
    source, _ = torch.sort(source)
    return source.view((number_samples, 1))

def generate_target(number_samples, device):
    rand = torch.randn((number_samples,), device = device) - 1
    target = true_transport_function(rand)
    target, _ = torch.sort(target)
    return target.view((number_samples, 1))

def get_bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

def detach(obj):
    return obj.detach().numpy()

# ---------------------------------------
# prepare and cache some information
# ---------------------------------------

# prepare the data
source_data = generate_source(number_samples_source, device = device)
target_data = generate_target(number_samples_target, device = device)
source_data.requires_grad = True

true_transported_data = true_transport_function(source_data)

# prepare some histogram data that stays constant
bin_edges = np.linspace(-5.0, 5.0, 100)
source_data_hist, source_data_edges = np.histogram(detach(source_data), bins = bin_edges)
target_data_hist, target_data_edges = np.histogram(detach(target_data), bins = bin_edges)
true_transported_data_hist, true_transported_data_edges = np.histogram(detach(true_transported_data), bins = bin_edges)

use_wasserstein = False

# ---------------------------------------
# utilities for later
# ---------------------------------------

def add_transport_potential_comparison_plot(true_potential, network, name, writer, global_step, xlabel = "x", ylabel = "transport potential"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    xvals = np.linspace(-5, 5, 50, dtype = np.float32)
    xvals = np.expand_dims(xvals, axis = 1)
    
    vals_network = detach(network(torch.from_numpy(xvals)))
    vals_network -= np.min(vals_network)
    
    vals_true = true_potential(xvals)
    vals_true -= np.min(vals_true)

    ax.plot(xvals, vals_true, color = "black", linestyle = "solid", linewidth = 2.0, label = "true potential")
    ax.plot(xvals, vals_network, color = "red", linestyle = "dashed", label = "learned potential")

    leg = ax.legend()
    leg.get_frame().set_linewidth(0.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    writer.add_figure(name, fig, global_step = global_step)
    plt.close()

def add_transport_closure_plot(transported_data_nominal, writer, global_step, xlabel = "x"):

    def plot_histogram_from_data(ax, data, **kwargs):
        hist, edges = np.histogram(data, bins = bin_edges)
        ax.hist(get_bin_centers(edges), weights = hist / sum(hist),
                histtype = 'step', bins = edges, **kwargs)        
    
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.hist(get_bin_centers(source_data_edges), weights = source_data_hist / sum(source_data_hist),
            histtype = 'step', bins = source_data_edges, label = "MC", color = "black", ls = "dotted", lw = 2)
    ax.hist(get_bin_centers(true_transported_data_edges), weights = true_transported_data_hist / sum(true_transported_data_hist),
            histtype = 'step', bins = true_transported_data_edges, label = "truth", color = "red", ls = "dashed", lw = 2)

    plot_histogram_from_data(ax, transported_data_nominal, label = "transported MC", color = "blue", lw = 2)

    ax.scatter(get_bin_centers(target_data_edges), target_data_hist / sum(target_data_hist), color = 'black',
               label = "data")
    leg = ax.legend()
    leg.get_frame().set_linewidth(0.0)

    ax.set_xlabel(xlabel)

    writer.add_figure("transport_closure", fig, global_step = global_step)
    plt.close()

def add_transport_plot(xval, yval_nominal, writer, global_step, xlabel = "x", ylabel = "transport function"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(xval, yval_nominal, color = "red", linestyle = "dashed", label = "learned transport function")
    ax.plot(detach(source_data), detach(true_transported_data - source_data), linestyle = "solid", color = "black", lw = 2, label = "true transport function")
    
    leg = ax.legend()
    leg.get_frame().set_linewidth(0.0)    

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    writer.add_figure("transport", fig, global_step = global_step)
    plt.close()

def add_critic_plot(xval, yval, writer, global_step, xlabel = "x"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(xval, yval, color = "black", lw = 2, label = "critic")

    ax.set_xlabel(xlabel)
    
    writer.add_figure("critic", fig, global_step = global_step)
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
number_outputs = 1 if use_gradient else 1 # in 1D transport, the transport network _always_ needs to have a single output, gradients or not

transport_network = build_fully_connected(1, 1, number_hidden_layers = 1, units_per_layer = 30,
                                          activation = torch.nn.Tanh)
transport_network[-1].weight.data *= 0.01
transport_network[-1].bias.data *= 0.01
transport_network.to(device)

critic = build_fully_connected(1, critic_outputs, number_hidden_layers = 1, units_per_layer = 30,
                               activation = torch.nn.Tanh)
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
        transported_data_nominal = apply_transport(transport_network, source_data)
        critic_output_nominal = critic(transported_data_nominal)

        add_transport_potential_comparison_plot(true_transport_potential, transport_network, name = "transport_potential_comparison", writer = writer, global_step = batch)
        add_transport_closure_plot(transported_data_nominal = detach(transported_data_nominal), writer = writer, global_step = batch)
        add_transport_plot(xval = detach(source_data), yval_nominal = detach(transported_data_nominal - source_data), writer = writer, global_step = batch)
        add_critic_plot(xval = detach(source_data), yval = detach(critic_output_nominal), writer = writer, global_step = batch)

writer.close()
print("done")
