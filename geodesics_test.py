import torch, os
import numpy as np

# for plot output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = 'cpu'

def detach(obj):
    return obj.detach().numpy()

def generate_geodesic(source_point, tangent_vector):

    source_point = np.array([source_point], dtype = np.float32)
    source_point_tensor = torch.from_numpy(source_point)
    source_point_tensor.requires_grad = True

    tangent_vector = np.array([tangent_vector], dtype = np.float32)
    tangent_vector_tensor = torch.from_numpy(tangent_vector)
    tangent_vector_tensor.requires_grad = True
    
    # compute velocity tangent vector
    vel_r, vel_phi, source_r, source_phi = compute_tangent_vector_r_phi(source_point_tensor, tangent_vector_tensor)
    
    # norm of velocity tangent vector
    vel_norm = torch.sqrt((A ** 2) * torch.square(vel_r) + torch.square(source_r * vel_phi)) + eps

    # normalise velocity tangent vector
    vel_r_norm = vel_r / vel_norm
    vel_phi_norm = vel_phi / vel_norm

    # compute constants that define the geodesic
    L = vel_phi_norm * torch.square(source_r)
    C1 = torch.sign(vel_r) * torch.sqrt(torch.square(source_r) - torch.square(L))
    C2 = source_phi - A * torch.atan(C1 / (L + eps))
    
    s_vals = vel_norm * torch.linspace(0.0, 1.0, 100)
    
    # evaluate the geodesic at the right point
    r_target = torch.sqrt(torch.square(L) + torch.square(C1 + s_vals / A))
    phi_target = C2 + A * torch.atan((A * C1 + s_vals) / (A * L + eps))

    # print(A * C1 + s_vals)
    # print(phi_target)
    print("--------")
    print("x = {}".format(A * L))
    print("y = {}".format(A * C1 + s_vals))
    print("atan2 = {}".format(torch.atan2(A * C1 + s_vals, A * L)))
    print("--------")

    x_target = r_target * torch.cos(phi_target)
    y_target = r_target * torch.sin(phi_target)

    target = torch.cat([torch.unsqueeze(x_target, dim = 1), torch.unsqueeze(y_target, dim = 1)], dim = 1)

    return target

def compute_tangent_vector_r_phi(source, tangent_vector):

    # print("apply_transport")
    
    source_x = source[:,0]
    source_y = source[:,1]

    # print("x = {}".format(source_x))
    # print("y = {}".format(source_y))    

    source_phi = torch.atan2(source_y, source_x)
    source_r = torch.sqrt(torch.square(source_x) + torch.square(source_y))

    # print("phi = {}".format(source_phi))
    # print("r = {}".format(source_r))    

    grad_x = tangent_vector[:,0]
    grad_y = tangent_vector[:,1]

    # print("grad_x = {}".format(grad_x))
    # print("grad_y = {}".format(grad_y))
    
    # r-component of tangent velocity
    vel_r = 1.0 / (A ** 2) * (torch.cos(source_phi) * grad_x + torch.sin(source_phi) * grad_y)
    vel_phi = 1.0 / source_r * (-torch.sin(source_phi) * grad_x + torch.cos(source_phi) * grad_y)

    # print("vel_r = {}".format(vel_r))
    # print("vel_phi = {}".format(vel_phi))
    
    return vel_r, vel_phi, source_r, source_phi

def compute_tangent_vector_x_y(source, tangent_vector):

    # first compute r / phi components of tangent vector
    vel_r, vel_phi, source_r, source_phi = compute_tangent_vector_r_phi(source, tangent_vector)

    # convert them into the x / y components
    vel_x = torch.cos(source_phi) * vel_r - source_r * torch.sin(source_phi) * vel_phi
    vel_y = torch.sin(source_phi) * vel_r + source_r * torch.cos(source_phi) * vel_phi
    
    vel = torch.cat([torch.unsqueeze(vel_x, dim = 1), torch.unsqueeze(vel_y, dim = 1)], dim = 1)

    return vel

def add_geodesic_plot(outpath, xlabel = "x", ylabel = "y"):

    def plot_geodesic(geodesic, ax, color = "black"):
        ax.scatter(geodesic[0][0], geodesic[0][1], color = color)
        ax.scatter(geodesic[-1][0], geodesic[-1][1], color = color)
        ax.plot(geodesic[:,0], geodesic[:,1], color = color)

        arrow_ind = int(len(geodesic) / 2)
        ax.annotate('',
                    xy = (geodesic[arrow_ind + 1][0], geodesic[arrow_ind + 1][1]),
                    xytext = (geodesic[arrow_ind][0], geodesic[arrow_ind][1]),
                    arrowprops = {'arrowstyle': '->', 'lw': 3, 'color': color},
                    va = 'center')

    def plot_transport_vector_field(ax, tangent_vector):
        xvals = np.linspace(-2, 2, 20, dtype = np.float32)
        yvals = np.linspace(-2, 2, 20, dtype = np.float32)

        xgrid, ygrid = np.meshgrid(xvals, yvals)
        vals = np.stack([xgrid.flatten(), ygrid.flatten()], axis = 1)
    
        # compute transport field
        vals_tensor = torch.from_numpy(vals)
        vals_tensor.requires_grad = True

        tangent_vector = np.array([tangent_vector], dtype = np.float32)
        tangent_vector_tensor = torch.from_numpy(tangent_vector)
        tangent_vector_tensor.requires_grad = True    
        
        transport_field_tensor = compute_tangent_vector_x_y(vals_tensor, tangent_vector_tensor)
        transport_field = detach(transport_field_tensor)
    
        ax.quiver(vals[:,0], vals[:,1], transport_field[:,0], transport_field[:,1], color = 'black')

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)    

    tangent_vector = [3.5, 1.5]
    points = [[-0.2, 0.1], [-0.4, 0.1], [-0.6, 0.1], [-0.8, 0.1], [-1.0, 0.1], [-1.5, 1.0]]
    points = [[0.2, 0.1], [0.4, 0.1], [0.6, 0.1], [0.8, 0.1], [1.0, 0.1], [1.5, 1.0]]
    points = [[0.1, 0.2], [0.1, 0.4], [0.1, 0.6], [0.1, 0.8], [0.1, 1.0], [1.0, 1.5]]
    points = [[0.1, -0.2], [0.1, -0.4], [0.1, -0.6], [0.1, -0.8], [0.1, -1.0], [1.0, -1.5]]
    colors = ["red", "green", "blue", "orange", "black", "gray"]

    plot_transport_vector_field(ax, tangent_vector)
    
    for point, color in zip(points, colors):
        geodesic_tensor = generate_geodesic(point, tangent_vector)
        geodesic = detach(geodesic_tensor)
        plot_geodesic(geodesic, ax, color)

    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))

    fig.savefig(outpath)

    plt.close()    


# -----------------------------------
# this is where things happen
# -----------------------------------

A = 5
eps = 1e-2
outdir = "geodesics_test"

if not os.path.exists(outdir):
    os.makedirs(outdir)

outpath = os.path.join(outdir, "geo.pdf")

add_geodesic_plot(outpath)
