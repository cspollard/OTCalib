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
outdir = "gaussian"

number_samples_target = 5000                       # data
number_samples_source = 4 * number_samples_target # MC

lr_transport = 5e-5
lr_critic = 5e-5
critic_updates_per_batch = 10
critic_outputs = 1

number_thetas = 2
ensemble_size = 20

# ---------------------------------------
# utilities for now
# ---------------------------------------

def true_transport_vector(x):
    #return scipy.special.erfinv(x) - x
    return x * x - x

def true_transport_function(x):
    return x + true_transport_vector(x)

def generate_source(number_samples, device):
    #source = 2 * torch.rand((number_samples,), device = device) - 1 # uniform distribution on [-1, 1]
    source = torch.rand((number_samples,), device = device) # uniform distribution on [-1, 1]
    source, _ = torch.sort(source)
    return source.view((number_samples, 1))

def generate_target(number_samples, device):
    #rand = 2 * torch.rand((number_samples,), device = device) - 1 # uniform distribution on [-1, 1]
    rand = torch.rand((number_samples,), device = device) # uniform distribution on [-1, 1]
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

true_transported_data = true_transport_function(source_data)

# prepare some histogram data that stays constant
#bin_edges = np.linspace(-2.5, 2.5, 50)
bin_edges = np.linspace(0.0, 1.0, 100)
source_data_hist, source_data_edges = np.histogram(source_data, bins = bin_edges)
target_data_hist, target_data_edges = np.histogram(target_data, bins = bin_edges)
true_transported_data_hist, true_transported_data_edges = np.histogram(true_transported_data, bins = bin_edges)

use_wasserstein = True

# ---------------------------------------
# utilities for later
# ---------------------------------------

def add_transport_closure_plot(transported_data_nominal, transported_data_ensemble, writer, global_step):

    def plot_histogram_from_data(ax, data, **kwargs):
        hist, edges = np.histogram(data, bins = bin_edges)
        ax.hist(get_bin_centers(edges), weights = hist / sum(hist),
                histtype = 'step', bins = edges, **kwargs)        
    
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.hist(get_bin_centers(source_data_edges), weights = source_data_hist / sum(source_data_hist),
            histtype = 'step', bins = source_data_edges, label = "original prediction", color = "black", ls = "dotted", lw = 2)
    ax.hist(get_bin_centers(true_transported_data_edges), weights = true_transported_data_hist / sum(true_transported_data_hist),
            histtype = 'step', bins = true_transported_data_edges, label = "true target", color = "red", ls = "dashed", lw = 2)

    plot_histogram_from_data(ax, transported_data_nominal, label = "transported prediction", color = "blue", lw = 2)

    for cur_data in transported_data_ensemble:
        plot_histogram_from_data(ax, cur_data, color = "blue", lw = 1)
    
    ax.scatter(get_bin_centers(target_data_edges), target_data_hist / sum(target_data_hist), color = 'black',
               label = "observed target")
    leg = ax.legend()
    leg.get_frame().set_linewidth(0.0)    

    writer.add_figure("transport_closure", fig, global_step = global_step)
    plt.close()

def add_transport_plot(xval, yval_nominal, yval_ensemble, writer, global_step):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(xval, yval_nominal, color = "blue", lw = 2, label = "transport function")

    for cur_yval in yval_ensemble:
        ax.plot(xval, cur_yval, color = "blue", lw = 1)
    
    ax.plot(source_data, true_transported_data, color = "red", lw = 2, label = "true transport function")
    leg = ax.legend()
    leg.get_frame().set_linewidth(0.0)    

    writer.add_figure("transport", fig, global_step = global_step)
    plt.close()

def add_critic_plot(xval, yval, writer, global_step):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.plot(xval, yval, color = "black", lw = 2, label = "critic")

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

def apply_transport(network, source, thetas):

    tmp = network(source)    
    central_value = tmp[:, 0:1]    
    dual_thetas = tmp[:, 1:]    
    correction = torch.unsqueeze(torch.matmul(dual_thetas, thetas), -1)    
    transport_vector = central_value + correction
    
    return source + transport_vector

# ---------------------------------------
# this is where things happen
# ---------------------------------------

# prepare the logging output
from time import gmtime, strftime
time_suffix = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
runname = os.path.join(outdir, time_suffix)
writer = SummaryWriter(runname)

# build the transport network and start from somewhere close to the identity
transport_network = build_fully_connected(1, 1 + number_thetas, number_hidden_layers = 3, units_per_layer = 20,
                                          activation = torch.nn.LeakyReLU)
transport_network[-1].weight.data *= 0.01
transport_network[-1].bias.data *= 0.01
transport_network.to(device)

critic = build_fully_connected(1, critic_outputs, number_hidden_layers = 3, units_per_layer = 20,
                               activation = torch.nn.LeakyReLU)
critic.to(device)

phi = build_fully_connected(2 * critic_outputs, 1, number_hidden_layers = 1, units_per_layer = 20,
                            activation = torch.nn.LeakyReLU)
phi.to(device)

# build the optimisers
transport_optim = torch.optim.RMSprop(transport_network.parameters(), lr = lr_transport)
adversary_optim = torch.optim.RMSprop(list(critic.parameters()) + list(phi.parameters()), lr = lr_critic)

for batch in range(50000):

    print("step {}".format(batch))
    
    for cur_adversary_update in range(critic_updates_per_batch):
    
        # sample current batch
        source_data_batch = source_data
        target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
                                                      size = (number_samples_source,), device = device)]
        thetas_batch = torch.randn((number_thetas,), device = device)
        
        # ------------------------------
        # train adversary
        # ------------------------------

        adversary_optim.zero_grad()
        transport_optim.zero_grad()
        
        phival_target = phi(build_moments(critic(target_data_batch)))
        
        transported_source_data_batch = apply_transport(transport_network, source_data_batch, thetas_batch)        
        phival_transported_source = phi(build_moments(critic(transported_source_data_batch)))

        if use_wasserstein:
            # Wasserstein loss for adversary
            adv_loss = phival_transported_source - phival_target
        else:
            adv_loss = torch.binary_cross_entropy_with_logits(phival_transported_source, torch.zeros_like(phival_transported_source)) + torch.binary_cross_entropy_with_logits(phival_target, torch.ones_like(phival_target))
            
        # phi + critic update
        adv_loss.backward()
        adversary_optim.step()

        if use_wasserstein:
            for p in list(critic.parameters()) + list(phi.parameters()):
                p.data.clamp_(-0.1, 0.1)
    
    # ------------------------------
    # train transport network
    # ------------------------------

    # sample current batch
    source_data_batch = source_data
    target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
                                                  size = (number_samples_source,), device = device)]
    thetas_batch = torch.randn((number_thetas,), device = device)
    
    transport_optim.zero_grad()
    adversary_optim.zero_grad()

    # minimise the Wasserstein loss between transported source and bootstrapped target
    transported_source_data_batch = apply_transport(transport_network, source_data_batch, thetas_batch)
    critic_output = critic(transported_source_data_batch)
    phival_transported_source = phi(build_moments(critic_output))
    
    if use_wasserstein:
        transport_loss = -phival_transported_source
    else:
        transport_loss = torch.binary_cross_entropy_with_logits(phival_transported_source, torch.ones_like(phival_transported_source))

    transport_loss.backward()
    transport_optim.step()

    if not batch % 10:
    
        # ------------------------------
        # make diagnostic plots
        # ------------------------------
        
        # for plots, prepare the ensemble of transport functions
        transported_data_nominal = detach(apply_transport(transport_network, source_data_batch, torch.zeros((number_thetas,))))
        
        transported_data_ensemble = []
        for cur in range(ensemble_size):
            cur_thetas = torch.randn((number_thetas,), device = device)
            cur_transported_data = detach(apply_transport(transport_network, source_data_batch, cur_thetas))
            transported_data_ensemble.append(cur_transported_data)
        
        add_transport_closure_plot(transported_data_nominal = transported_data_nominal, transported_data_ensemble = transported_data_ensemble, writer = writer, global_step = batch)
        add_transport_plot(xval = detach(source_data_batch), yval_nominal = transported_data_nominal, yval_ensemble = transported_data_ensemble, writer = writer, global_step = batch)
        add_critic_plot(xval = detach(source_data_batch), yval = detach(critic_output), writer = writer, global_step = batch)

writer.close()
print("done")
