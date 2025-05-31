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
        self.training_step = 0
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

    

   
    def consistency_function(self, x, sigma, context, use_target=False, x_raw=None):
        """The consistency function F_θ or F_θ⁻ with architectural boundary condition"""
        if x_raw is None:
            x_raw = x
        
        # Choose which network to use
        network = self.target_net if use_target else self.net
        
        # Always get network prediction to maintain gradient flow
        f_theta = network(x, beta=sigma, context=context)
        output = torch.where(sigma[0] == 0.002, x_raw, f_theta)

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
        
    def _karras_schedule(self, N, eps, T, rho):
        """Generate Karras schedule using the boundary function"""
        return torch.tensor([
            (T ** (1 / rho) + (i)/ (N - 1) * 
                (eps ** (1 / rho) - T ** (1 / rho))) ** rho
            for i in range(N)
        ], dtype=torch.float32)

    def get_loss(self, x_0, context, t=None, x_raw=None):
        """Consistency training loss with adaptive Karras scheduling"""
        batch_size, _, point_dim = x_0.size()
        
        # Get adaptive number of steps for this training iteration
        N_k = self.get_adaptive_num_steps()
        
        # Generate Karras sigmas for current adaptive steps
        karras_boundaries = self._karras_schedule(N_k, 0.002, 80, 7)
        
        if t is None:
            # Sample timesteps from adaptive range [1, N_k-1] inclusive
            t_indices = torch.randint(1, N_k, (batch_size,))
        else:
            # Ensure provided t is within adaptive range
            t_indices = torch.clamp(torch.tensor(t), 1, N_k-1)
            
        # Get sigma values from Karras schedule
        sigma_t = karras_boundaries[t_indices].to(x_0.device)
        
        # For t-1, handle boundary case
        t_minus_1_indices = t_indices - 1
        sigma_t_minus_1 = karras_boundaries[t_minus_1_indices].to(x_0.device)
        
        # Add noise to clean data using Karras sigmas
        noise = torch.randn_like(x_0)
        x_t = x_0 + sigma_t.view(-1, 1, 1) * noise
        x_t_minus_1 = x_0 + sigma_t_minus_1.view(-1, 1, 1) * noise
        
        # Consistency loss: F_θ(x_t, σ_t) should equal F_θ⁻(x_{t-1}, σ_{t-1})
        # Target from EMA network (no gradients)
        with torch.no_grad():
            target = self.consistency_function(x_t_minus_1, sigma_t_minus_1, context, use_target=True, x_raw=x_raw)
        
        # Prediction from main network (with gradients)
        prediction = self.consistency_function(x_t, sigma_t, context, use_target=False, x_raw=x_raw)
        
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
        x_copy=x
        for i in range(len(ts)-1):
            t= (T ** (1 / rho) + (ts[i])/ (N - 1) * 
                    (eps ** (1 / rho) - T ** (1 / rho))) ** rho
            t=torch.full((batch_size,),t, device=context.device)
            x0=self.consistency_function(x,t,context, use_target=False)
            t_next= (T ** (1 / rho) + (ts[i+1])/ (N - 1) * 
                    (eps ** (1 / rho) - T ** (1 / rho))) ** rho
            t_next =np.clip(t_next,eps,T)
            # t_next=torch.full((batch_size,),t_next, device=context.device)
            x = x0 + torch.randn_like(x) * math.sqrt(t_next**2 - eps**2)
        if ret_traj:
            return {4000:x_copy,0:x}
        else:
            return x

