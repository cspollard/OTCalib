import torch, os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.special

# for plot output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
from sys import argv, stdout
import shutil

if len(argv) < 2:
  print("please provide a json steering file")
  exit(-1)

fconfig = open(argv[1])
config = json.load(fconfig)
fconfig.close()


outdir = config["outdir"]
device = config["device"]

number_samples_target = config["number_samples_target"]
number_samples_source = config["number_samples_source"]

number_thetas = config["number_thetas"]
batch_size = config["batch_size"]

use_gradient = config["use_gradient"]

modeljson = config["model"]

class JSONModel:
  def __init__(self, dict):
    pass

  # TODO:
  # do we need to encode the true transport function?
  # def true_transport_function(x):
  #   pass

  def generate_source(self, x, device):
    pass

  def generate_target(self, x, device):
    pass


  def plots(self):
    pass


class Circle2D (JSONModel):

# ---------------------------------------
# utilities for now
# ---------------------------------------

  def true_transport_function(self, x):
      return 1.5 * x

  def generate_source(self, number_samples, device):
      angles = 2 * np.pi * torch.rand((number_samples,), device = device)
      radii = torch.sqrt(torch.rand((number_samples,), device = device))
      source = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim = 1)
      return source

  def generate_target(self, number_samples, device):
      angles = 2 * np.pi * torch.rand((number_samples,), device = device)
      radii = 2.0 * torch.sqrt(torch.rand((number_samples,), device = device))
      target = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim = 1)
      return target


model = Circle2D(modeljson)


def get_bin_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])

def detach(obj):
    return obj.detach().cpu().numpy()

# ---------------------------------------
# prepare and cache some information
# ---------------------------------------

# prepare the data
source_data = model.generate_source(number_samples_source, device = device)
target_data = model.generate_target(number_samples_target, device = device)
source_data.requires_grad = True

# true_transported_data = model.true_transport_function(source_data)

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
    
    writer.add_figure("source", fig, global_step = global_step)
    plt.close()

def add_target_plot(observed_data, transported_data, global_step):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    ax.hexbin(x = transported_data[:, 0], y = transported_data[:, 1], mincnt = 1)

    scatter = ax.scatter(x = observed_data[:, 0], y = observed_data[:, 1], marker = 'x', c = 'red')
    ax.legend([scatter], ["target"])
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    
    writer.add_figure("target", fig, global_step = global_step)
    plt.close()    
    
def add_network_plot(network, name, global_step):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    # prepare grid for the evaluation of the critic
    xvals = np.linspace(-2.0, 2.0, 50, dtype = np.float32)
    yvals = np.linspace(-2.0, 2.0, 50, dtype = np.float32)

    xgrid, ygrid = np.meshgrid(xvals, yvals)
    vals = np.stack([xgrid.flatten(), ygrid.flatten()], axis = 1)

    # evaluate critic
    zvals = detach(network(torch.from_numpy(vals).to(device)))
    zvals = np.reshape(zvals, xgrid.shape)

    conts = ax.contourf(xgrid, ygrid, zvals, 100)

    ax.set_aspect(1)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    
    plt.colorbar(conts)
    
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

def apply_transport(network, source, thetas):
    
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

# always keep a copy of the steering file
shutil.copyfile(argv[1], outdir + "/" + time_suffix + ".json")
writer = SummaryWriter(runname)

# build the transport network and start from somewhere close to the identity
number_outputs = 1 if use_gradient else 2

transport_network = build_fully_connected(2, number_outputs, number_hidden_layers = 1, units_per_layer = 30,
                                          activation = torch.nn.Tanh)
transport_network[-1].weight.data *= 0.01
transport_network[-1].bias.data *= 0.01
transport_network.to(device)

critic = build_fully_connected(2, config["critic_outputs"], number_hidden_layers = 1, units_per_layer = 30,
                               activation = torch.nn.Tanh)
critic.to(device)

# build the optimisers
transport_optim = torch.optim.RMSprop(transport_network.parameters(), lr = config["lr_transport"])
adversary_optim = torch.optim.RMSprop(list(critic.parameters()), lr = config["lr_critic"])

for batch in range(50000):

    print("step {}".format(batch))
    
    for cur_adversary_update in range(config["critic_updates_per_batch"]):
    
        # sample current batch
        # source_data_batch = source_data
        # target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
        #                                               size = (number_samples_source,), device = device)]
        source_data_batch = source_data[torch.randint(low = 0, high = number_samples_source,
                                                      size = (batch_size,), device = device)]
        target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
                                                      size = (batch_size,), device = device)]
        thetas_batch = torch.randn((number_thetas,), device = device)
        
        # ------------------------------
        # train adversary
        # ------------------------------

        adversary_optim.zero_grad()
        transport_optim.zero_grad()
        
        critic_target = critic(target_data_batch)
        
        transported_source_data_batch = apply_transport(transport_network, source_data_batch, thetas_batch)        
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
    # source_data_batch = source_data
    # target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
    #                                               size = (number_samples_source,), device = device)]
    source_data_batch = source_data[torch.randint(low = 0, high = number_samples_source,
                                                  size = (batch_size,), device = device)]
    target_data_batch = target_data[torch.randint(low = 0, high = number_samples_target,
                                                  size = (batch_size,), device = device)]    
    thetas_batch = torch.randn((number_thetas,), device = device)
    
    transport_optim.zero_grad()
    adversary_optim.zero_grad()

    # minimise the Wasserstein loss between transported source and bootstrapped target
    transported_source_data_batch = apply_transport(transport_network, source_data_batch, thetas_batch)
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
        
        transported_data_nominal = detach(apply_transport(transport_network, source_data, torch.zeros((number_thetas,))))

        add_network_plot(critic, name = "critic", global_step = batch)
        if use_gradient:
            add_network_plot(transport_network, name = "transport_potential", global_step = batch)
        add_source_plot(detach(source_data), global_step = batch)
        add_target_plot(observed_data = detach(target_data), transported_data = transported_data_nominal, global_step = batch
          )
            
writer.close()
print("done")