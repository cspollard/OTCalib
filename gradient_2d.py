import torch, os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.special

# for plot output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# where to run
device = 'cpu'
outdir = "gradient_2d_manifold"

number_samples_target = 5000
number_samples_source = 20 * number_samples_target # MC

lr_transport = 7e-4
lr_critic = 3e-3

# lr_transport = 4e-4
# lr_critic = 2e-3

# critic_updates_per_batch = 10
critic_updates_per_batch = 20

critic_outputs = 1

batch_size = 250
#batch_size = 5
use_gradient = True

# ds^2 = A^2 dr^2 + r^2 dphi^2
A = 5
eps = 1e-2

# ---------------------------------------
# utilities for now
# ---------------------------------------

def true_transport_potential_circle(x, y):
    return 0.5 * (np.square(x) + np.square(y))

def generate_source_circle(number_samples, device):
    angles = 2 * np.pi * torch.rand((number_samples,), device = device)
    radii = torch.sqrt(torch.rand((number_samples,), device = device)) + eps
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
    xvals = 2 * torch.rand((number_samples,), device = device) + 1
    yvals = 2 * torch.rand((number_samples,), device = device) - 1
    source = torch.stack([xvals, yvals], dim = 1)
    return source

def generate_target_square(number_samples, device):
    angle = 0.0
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    
    xvals = 2 * torch.rand((number_samples,), device = device) - 1
    yvals = 2 * torch.rand((number_samples,), device = device) + 1
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

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)

    plt.tight_layout()
    writer.add_figure("source", fig, global_step = global_step)    
    plt.close()

def add_target_plot(observed_data, transported_data, global_step, xlabel = "x", ylabel = "y"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    datahist = ax.hexbin(x = transported_data[:, 0], y = transported_data[:, 1], mincnt = 1, cmap = "plasma")

    scatter = ax.scatter(x = observed_data[:, 0], y = observed_data[:, 1], marker = 'x', c = 'red', alpha = 0.3)
    ax.legend([scatter], ["data"])
    
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
    xvals = np.linspace(-3.5, 3.5, 50, dtype = np.float32)
    yvals = np.linspace(-3.5, 3.5, 50, dtype = np.float32)

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

def add_transport_potential_comparison_plot_contours(true_potential, network, name, global_step, contour_values = np.linspace(0.05, 2.0, 15), xlabel = "x", ylabel = "y"):

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    # prepare grid for the evaluation of the network
    xvals = np.linspace(-3.2, 3.2, 50, dtype = np.float32)
    yvals = np.linspace(-3.2, 3.2, 50, dtype = np.float32)

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
    xvals = np.linspace(-3.2, 3.2, 20, dtype = np.float32)
    yvals = np.linspace(-3.2, 3.2, 20, dtype = np.float32)

    xgrid, ygrid = np.meshgrid(xvals, yvals)
    vals = np.stack([xgrid.flatten(), ygrid.flatten()], axis = 1)
    
    # compute transport field
    vals_tensor = torch.from_numpy(vals)
    vals_tensor.requires_grad = True
    
    transport_field_tensor = compute_tangent_vector_x_y(network, vals_tensor)
    transport_field = detach(transport_field_tensor)
    
    ax.quiver(vals[:,0], vals[:,1], transport_field[:,0], transport_field[:,1], color = 'black')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # plt.tight_layout()
    writer.add_figure(name, fig, global_step = global_step)
    plt.close()    

def add_geodesic_plot(network, name, global_step, xlabel = "x", ylabel = "y"):

    def plot_geodesic(geodesic, ax, color = "black"):
        ax.scatter(geodesic[0][0], geodesic[0][1], color = color)
        ax.scatter(geodesic[-1][0], geodesic[-1][1], color = color)
        ax.plot(geodesic[:,0], geodesic[:,1], color = color)
    
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)

    points = [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
    colors = ["red", "green", "blue", "orange"]

    for point, color in zip(points, colors):
        geodesic_tensor = generate_geodesic(transport_network, point)
        geodesic = detach(geodesic_tensor)
        plot_geodesic(geodesic, ax, color)

    ax.set_xlim((-3.2, 3.2))
    ax.set_ylim((-3.2, 3.2))

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

def compute_tangent_vector_r_phi(network, source):

    # print("apply_transport")
    
    source_x = source[:,0]
    source_y = source[:,1]

    # print("x = {}".format(source_x))
    # print("y = {}".format(source_y))    

    source_phi = torch.atan2(source_y, source_x)
    source_r = torch.sqrt(torch.square(source_x) + torch.square(source_y))

    # print("phi = {}".format(source_phi))
    # print("r = {}".format(source_r))    
    
    output = network(source)

    # print("phi = {}".format(output))
    
    if use_gradient:
        output = torch.autograd.grad(output, source, grad_outputs = torch.ones_like(output), create_graph = True)[0]

    grad_x = output[:,0]
    grad_y = output[:,1]

    # print("grad_x = {}".format(grad_x))
    # print("grad_y = {}".format(grad_y))
    
    # r-component of tangent velocity
    vel_r = 1.0 / (A ** 2) * (torch.cos(source_phi) * grad_x + torch.sin(source_phi) * grad_y)
    vel_phi = 1.0 / source_r * (-torch.sin(source_phi) * grad_x + torch.cos(source_phi) * grad_y)

    # print("vel_r = {}".format(vel_r))
    # print("vel_phi = {}".format(vel_phi))
    
    return vel_r, vel_phi, source_r, source_phi

def compute_tangent_vector_x_y(network, source):

    # first compute r / phi components of tangent vector
    vel_r, vel_phi, source_r, source_phi = compute_tangent_vector_r_phi(network, source)

    # convert them into the x / y components
    vel_x = torch.cos(source_phi) * vel_r - source_r * torch.sin(source_phi) * vel_phi
    vel_y = torch.sin(source_phi) * vel_r + source_r * torch.cos(source_phi) * vel_phi
    
    vel = torch.cat([torch.unsqueeze(vel_x, dim = 1), torch.unsqueeze(vel_y, dim = 1)], dim = 1)

    return vel

def generate_geodesic(network, source_point):

    source_point = np.array([source_point], dtype = np.float32)
    source_point_tensor = torch.from_numpy(source_point)
    source_point_tensor.requires_grad = True
    
    # compute velocity tangent vector
    vel_r, vel_phi, source_r, source_phi = compute_tangent_vector_r_phi(network, source_point_tensor)
    
    # norm of velocity tangent vector
    vel_norm = torch.sqrt((A ** 2) * torch.square(vel_r) + torch.square(source_r * vel_phi)) + eps

    # normalise velocity tangent vector
    vel_r_norm = vel_r / vel_norm
    vel_phi_norm = vel_phi / vel_norm

    # compute constants that define the geodesic
    L = vel_phi_norm * torch.square(source_r)
    C1 = torch.sign(vel_r) * torch.sqrt(torch.square(source_r) - torch.square(L))
    C2 = source_phi - A * torch.atan2(C1, L)
    
    s_vals = vel_norm * torch.linspace(0.0, 1.0, 50)
    
    # evaluate the geodesic at the right point
    r_target = torch.sqrt(torch.square(L) + torch.square(C1 + s_vals / A))
    phi_target = C2 + A * torch.atan2(A * C1 + s_vals, A * L)

    x_target = r_target * torch.cos(phi_target)
    y_target = r_target * torch.sin(phi_target)

    target = torch.cat([torch.unsqueeze(x_target, dim = 1), torch.unsqueeze(y_target, dim = 1)], dim = 1)

    return target

def apply_transport(network, source):
    
    # compute velocity tangent vector
    vel_r, vel_phi, source_r, source_phi = compute_tangent_vector_r_phi(network, source)

    vel_xy = compute_tangent_vector_x_y(network, source)
    
    # norm of velocity tangent vector
    vel_norm = torch.sqrt((A ** 2) * torch.square(vel_r) + torch.square(source_r * vel_phi)) + eps

    # normalise velocity tangent vector
    vel_r_norm = vel_r / vel_norm
    vel_phi_norm = vel_phi / vel_norm

    # compute constants that define the geodesic
    L = vel_phi_norm * torch.square(source_r)
    C1 = torch.sign(vel_r) * torch.sqrt(torch.square(source_r) - torch.square(L))
    C2 = source_phi - A * torch.atan2(C1, L)

    # evaluate the geodesic at the right point
    r_target = torch.sqrt(torch.square(L) + torch.square(C1 + vel_norm / A))
    phi_target = C2 + A * torch.atan2(A * C1 + vel_norm, A * L)

    # print("r_source = {}".format(source_r))
    # print("r_target = {}".format(r_target))

    # print("phi_source = {}".format(source_phi))
    # print("phi_target = {}".format(phi_target))
    
    x_target = r_target * torch.cos(phi_target)
    y_target = r_target * torch.sin(phi_target)

    # print("x_target = {}".format(x_target))
    # print("y_target = {}".format(y_target))
    
    target = torch.cat([torch.unsqueeze(x_target, dim = 1), torch.unsqueeze(y_target, dim = 1)], dim = 1)

    # print("source = {}".format(source))
    # print("target = {}".format(target))

    # target_alt = source + vel_xy

    # diff = target - target_alt
    # print("diff = {}".format(diff))

    # if torch.any(torch.isnan(diff)):
    #     import sys
    #     sys.exit(1)
    
    return target

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

transport_network = build_fully_connected(2, number_outputs, number_hidden_layers = 4, units_per_layer = 50,
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

        # vel_r, vel_phi = compute_tangent_vector_r_phi(transport_network, source_data_batch)

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

            add_geodesic_plot(transport_network, name = "geodesics", global_step = batch)
            add_network_plot(transport_network, name = "transport_potential", global_step = batch)
            # add_transport_potential_comparison_plot_contours(true_transport_potential, transport_network, name = "transport_potential_comparison_contours", global_step = batch)
            # add_transport_potential_comparison_plot_radial(true_transport_potential, transport_network, name = "transport_potential_comparison_radial", global_step = batch)
            
        add_source_plot(detach(source_data), global_step = batch)
        add_target_plot(observed_data = target_data, transported_data = transported_data_nominal, global_step = batch)
            
writer.close()
print("done")
