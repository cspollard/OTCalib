import torch
import numpy as np

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


# need a smooth version to make sure the transport potential has a smooth gradient
# NEED TO FIND SOMETHING MORE EFFICIENT HERE
def smooth_leaky_ReLU(x, a):
    sqrtpi = np.sqrt(np.pi)
    return 0.5 * ((a - 1) * torch.exp(-torch.square(x)) + sqrtpi * x * (1 + torch.erf(x) + a * torch.erfc(x)))

