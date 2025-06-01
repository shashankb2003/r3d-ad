import torch
import torch.nn.functional as F
from torch.nn import Module
import numpy as np
import copy
import math
from .common import *


class ConsistencyPoint(Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        
        # Create EMA target network
        self.target_net = copy.deepcopy(net)
        
        # Freeze target network (no gradients)
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        # Official adaptive scheduling parameters (from consistency training paper)
        self.training_step = 1
        self.s0 = 2.0                    # Initial discretization steps
        self.s1 = 4000.0  # Target discretization steps at end of training
        self.mu0 = 0.95                  # EMA decay rate at beginning of model training
        self.K = 40000                  # Total number of training iterations (default, will be updated)

    def set_total_training_steps(self, total_steps):
        """Set total training steps K for adaptive scheduling"""
        self.K = total_steps

    def get_adaptive_num_steps(self):
        """Get adaptive number of steps N(k) using official formula"""
        k = float(self.training_step)
        K = float(self.K)
        s0 = self.s0
        s1 = self.s1
        
        if k >= K:
            return int(s1)
        

        ratio = k / K
        inner_term = ratio * ((s1 + 1)**2 - s0**2) + s0**2
        N_k = math.ceil(math.sqrt(max(0, inner_term)) - 1) + 1

        # Ensure it's within valid bounds
        N_k = max(int(s0), min(N_k, int(s1)))
        
        return N_k
    
    def get_adaptive_ema_rate(self):
        """Get adaptive EMA rate μ(k) using official formula"""
        N_k = float(self.get_adaptive_num_steps())
        s0 = self.s0
        mu0 = self.mu0
        
        # Official formula: μ(k) = exp(s0 * log(μ0) / N(k))
        # Ensure N_k > 0 to avoid division by zero
        if N_k <= 0:
            N_k = s0
            
        mu_k = math.exp(s0 * math.log(mu0) / N_k)
        
        # Clamp to reasonable bounds [0.5, 0.999]
        mu_k = max(0.5, min(0.999, mu_k))
        
        return mu_k

    def get_scalings(self, sigma,sigma_data=0.5,sigma_min=0.002):
        c_skip = sigma_data**2 / (
            (sigma - sigma_min) ** 2 + sigma_data**2
        )
        c_out = (
            (sigma - sigma_min)
            * sigma_data
            / (sigma**2 + sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

   
    def consistency_function(self, x, sigma, context, use_target=False, x_raw=None):
        """The consistency function F_θ or F_θ⁻ with architectural boundary condition"""
        if x_raw is None:
            x_raw = x
        
        # Choose which network to use
        network = self.target_net if use_target else self.net
        
        # Always get network prediction to maintain gradient flow
        c_skip,c_out,c_in=self.get_scalings(sigma)
        # print("c_skip: ",c_skip.shape," c_out: ",c_out.shape," c_in: ",c_in.shape)
        # print("context dim: ",context.shape)
        f_theta = network(c_in[:, None, None]*x, beta=sigma, context=context)
        # output = torch.where(sigma[0] == 0.002, x_raw, f_theta)
        output=c_skip[:, None, None]*x_raw+c_out[:, None, None]*f_theta
        return output

    def update_target_network(self, ema_rate=None):
        """Update target network with adaptive EMA rate"""
        if ema_rate is None:
            ema_rate = self.get_adaptive_ema_rate()
        
        with torch.no_grad():
            for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
                target_param.data.mul_(ema_rate).add_(param.data, alpha=1 - ema_rate)

    def step_training(self):
        """Increment training step counter for adaptive scheduling"""
        self.training_step += 1
        
    # def _karras_schedule(self, N, eps, T, rho):
    #     """Generate Karras schedule using the boundary function"""
    #     return torch.tensor([
    #         (T ** (1 / rho) + (i)/ (N - 1) * 
    #             (eps ** (1 / rho) - T ** (1 / rho))) ** rho
    #         for i in range(N)
    #     ], dtype=torch.float32)
    def append_dims(self,x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]
    def get_loss(self, x_0, context, t=None, x_raw=None,sigma_max=80.0,sigma_min=0.002,rho=7):
        """Consistency training loss with adaptive Karras scheduling"""
        batch_size, _, point_dim = x_0.size()
        
        # Get adaptive number of steps for this training iteration
        N_k = self.get_adaptive_num_steps()
        
        # Generate Karras sigmas for current adaptive steps
        # karras_boundaries = self._karras_schedule(N_k, 0.002, 80, 7)
        
        # if t is None:
        #     # Sample timesteps from adaptive range [1, N_k-1] inclusive
        #     t_indices = torch.randint(0, N_k-1, (batch_size,))
        # else:
        #     # Ensure provided t is within adaptive range
        #     t_indices = torch.clamp(torch.tensor(t), 0, N_k-1)
            
        # # Get sigma values from Karras schedule
        # sigma_t = karras_boundaries[t_indices].to(x_0.device)
        
        # # For t-1, handle boundary case
        # prev_timestep_indices = t_indices + 1
        # sigma_t_minus_1 = karras_boundaries[prev_timestep_indices].to(x_0.device)
        indices = torch.randint(
            0, N_k - 1, (batch_size,), device=x_0.device
        )
        t = sigma_max ** (1 / rho) + indices / (N_k - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
        )
        t = t**rho

        t2 = sigma_max ** (1 / rho) + (indices + 1) / (N_k - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
        )
        t2 = t2**rho
        # print("Indices: ",indices," Steps: ",N_k)
        # Add noise to clean data using Karras sigmas
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * self.append_dims(t, batch_size)

        d = (x_t - x_0) / self.append_dims(t, batch_size)
        x_t2 = x_t + d * self.append_dims(t2 - t, batch_size)
        x_t2 = x_t2.detach()        
        # print("Timestep for online network is: ",t)
        # print("Timestep for target network is: ",t2)
        # Consistency loss: F_θ(x_t, σ_t) should equal F_θ⁻(x_{t-1}, σ_{t-1})
        # Target from EMA network (no gradients)
        with torch.no_grad():
            target = self.consistency_function(x_t2, t2, context, use_target=True, x_raw=x_raw)
        # Prediction from main network (with gradients)
        prediction = self.consistency_function(x_t, t, context, use_target=False, x_raw=x_raw)
        
        loss = F.mse_loss(prediction, target, reduction="mean")
        return loss

    # def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
    #     """Single-step sampling using main network"""
    #     batch_size = context.size(0)
        
    #     # Start from noise with maximum sigma from Karras schedule
    #     z = torch.randn([batch_size, num_points, point_dim]).to(context.device)*80.0
    #     sigma_init = torch.full((batch_size,),80.0, device=context.device)
    #     x_0 = self.consistency_function(z,sigma_init, context, use_target=False)
    #     noise_levels=[40.0,20.0,10.0,5.0]
    #     for t in noise_levels:
    #         z = torch.randn([batch_size, num_points, point_dim]).to(context.device)
    #         x_T = x_0 + math.sqrt(t**2 - 0.002**2) * z

    #         sigma_t = torch.full((batch_size,), t, device=context.device)
    #         x_0 = self.consistency_function(x_T, sigma_t, context, use_target=False)
        
    #     if ret_traj:
    #         return {4000: x_T, 0: x_0}
    #     else:
    #         return x_0 

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False,eps=0.002,T=80.0,rho=7.0,N = 151,ts=[0,67,150]):
        """Standard multistep consistency sampling following Algorithm 1"""
    
        batch_size = context.size(0)
        x = torch.randn([batch_size, num_points, point_dim]).to(context.device) * T

        s_in = x.new_ones([batch_size])
        for i in range(len(ts)-1):
            t= (T ** (1 / rho) + (ts[i])/ (N - 1) * 
                    (eps ** (1 / rho) - T ** (1 / rho))) ** rho
            x0=self.consistency_function(x,t * s_in,context, use_target=False)
            t_next= (T ** (1 / rho) + (ts[i+1])/ (N - 1) * 
                    (eps ** (1 / rho) - T ** (1 / rho))) ** rho
            t_next =np.clip(t_next,eps,T)
            # t_next=torch.full((batch_size,),t_next, device=context.device)
            x = x0 + torch.randn_like(x) * math.sqrt(t_next**2 - eps**2)
        return x

