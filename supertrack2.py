# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gym
import numpy as np
import torch
import random
import time
from torch import nn
from torch.distributions import MultivariateNormal
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch import autograd

print("My numpy version is: ", np.__version__) # version 1.23.1 is required

from my_lib_t import *
device = 'cuda'

def global_means_std():
    w_inputs, p_inputs = [], []
    for episode in agent.memory:  
        for frame in episode:  
            l_state_s = agent.env.to_local(frame[0]['state_s']) 
            l_state_k = agent.env.to_local(frame[0]['state_k'])

            l_state_s = torch.cat([torch.tensor(l_state_s[0], dtype=torch.float32).flatten(), torch.tensor(l_state_s[1], dtype=torch.float32).flatten(), torch.tensor(l_state_s[2], dtype=torch.float32).flatten(), torch.tensor(l_state_s[3], dtype=torch.float32).flatten(), torch.tensor(l_state_s[4], dtype=torch.float32), torch.tensor(l_state_s[5], dtype=torch.float32)])
            l_state_k = torch.cat([torch.tensor(l_state_k[0], dtype=torch.float32).flatten(), torch.tensor(l_state_k[1], dtype=torch.float32).flatten(), torch.tensor(l_state_k[2], dtype=torch.float32).flatten(), torch.tensor(l_state_k[3], dtype=torch.float32).flatten(), torch.tensor(l_state_k[4], dtype=torch.float32), torch.tensor(l_state_k[5], dtype=torch.float32)])

            p_inputs.append(torch.cat((l_state_s, l_state_k)))

            ti = torch.tensor(frame[1], dtype=torch.float32) 
            w_inputs.append(torch.cat((l_state_s, ti.flatten())))
    
    p_mean = torch.mean(torch.stack(p_inputs), dim=0).to(device)
    w_mean = torch.mean(torch.stack(w_inputs), dim=0).to(device)

    p_std = torch.std(torch.stack(p_inputs), dim=0, unbiased=False).to(device) # unbiased=False cuz polulation not sample 
    w_std = torch.std(torch.stack(w_inputs), dim=0, unbiased=False).to(device)
    
    return p_mean, p_std, w_mean, w_std


# Neural Network Definition
class FeedForwardNN(nn.Module):           
    def __init__(self, in_dim, out_dim, tanh=False):
        super(FeedForwardNN, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        if tanh==True:
            self.model = nn.Sequential(
                nn.Linear(in_dim, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, out_dim),
                nn.Tanh() # output normalization
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_dim, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, 1024),
                nn.ELU(),
                nn.Linear(1024, out_dim)
            )
        torch.manual_seed(14) #! stochasticity
        random.seed(14) #! stochasticity
        self.model.apply(self.init_weights) # weight initialization
        # self.mean, self.std = None, None

    #     for idx, layer in enumerate(self.model):
    #         if isinstance(layer, (nn.Linear, nn.ELU, nn.Tanh)):  # You can refine this
    #             layer.register_forward_hook(self.make_hook(f"layer_{idx}"))
                
    # def make_hook(self, name):
    #     def hook(module, input, output):
    #         self.activations[name] = output.detach()
    #     return hook

    def forward(self, obs):  # expects shape [batch_size, input_dim]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = self.bn(obs) # normalize each feature
        # print('FORWARD',obs[0])

        # time.sleep(1000)
        # print('222222##############', obs.mean())
        # print('222222@@@@@@@@@@@@@@', obs.std())
        # if self.mean is not None:
            # print("____#NORMALIZED#____")
            # obs = (obs - self.mean) / (self.std + 1e-10) # normalize
        # time.sleep(10000)
        return self.model(obs)
    
    @staticmethod
    def init_weights(m):
        """Use Kaiming Uniform init Also known as He initialization"""
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=0.01)
            # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # 'relu' also works well for ELU
            # torch.nn.init.uniform_(m.weight, a=-50, b=50)
            # if m.bias is not None:
            #     nn.init.zeros_(m.bias)

# Define the DQN agent class
class SUPERTRACK:
    def __init__(self, env):
        self.env = env
        # self.obs_size = env.observation_space.shape[0] # 390 values -> 572
        # self.action_size = env.action_space.shape[0] # 48 values  -> 
        # self.obs_size = 286
        # self.action_size = 63

        self.memory = deque() #maxlen=10

        self._init_hyperparameters()

        self.policy = FeedForwardNN(710, 63, tanh=True).to(device)  # 21 * 3
        self.policy_optim = torch.optim.RAdam(self.policy.parameters(), lr=self.lr_p)
        
        self.world = FeedForwardNN(439, 132).to(device) #355+84, 66+66
        self.world_optim = torch.optim.RAdam(self.world.parameters(), lr=self.lr_w)

        self.w_mean, self.w_std, self.p_mean, self.p_std = None, None, None, None
        # self.cov_var = torch.full(size=(63,), fill_value=self.noise_scale) 
        # self.cov_mat = torch.diag(self.cov_var) # (63,63) - reuslts in matrix with fill_value as diagonals (everythitng else is 0)
        # print('\ncov_var',self.cov_var)
        # print('\ncov_mat',self.cov_mat.shape)

    def _init_hyperparameters(self):
        self.lr_w = 0.001 # optimal lr 0.0001
        self.lr_p = 0.001 # optimal lr 0.0001
        # self.num_episodes = 1000
        # self.n_updates_per_iteration = 200 # well we dont need update weights over the same N_steps(Batch samples), as our method is sample efficient.

        self.min_ep_len = 48
        self.max_ep_len = 512
        self.offset_scale = 2.094 #120 not 2.094 rad | CLOSE to 0 will almost produce identity rot -> pure kin rot PD
        self.noise_scale = 0.01 #! stochasticity
        self.wl = 0.01
        self.wl_s = 0.02 #0.02
        self.w_loss_history = []
        self.p_loss_history = []

        self.dt=1./60.


    def learn(self, total_timesteps, batch_size_w, batch_size_p): # learn
        #train world model | policy model
        self.t_so_far = 0
        # pbar = tqdm(total = total_timesteps)
        while self.t_so_far < total_timesteps: # k training iteration
            # for _ in range(self.n_updates_per_iteration): # do Multiple Updates on One mini-Batch Data
            self.t_so_far += 1
            # pbar.update(1)
            #------------------------dynamic lr------------------------------------
            # frac = (self.t_so_far - 1.0) / total_timesteps
            # new_lr_p = self.lr_p * (1.0 - frac)
            # new_lr_p = max(new_lr_p, 1e-10)
            # new_lr_w = self.lr_w * (1.0 - frac)
            # new_lr_w = max(new_lr_w, 1e-10)
            # print('NEW_LR_NEW_LR_NEW_LR_NEW_LR',new_lr_p)
            # self.world_optim.param_groups[0]["lr"] = new_lr_w
            # self.policy_optim.param_groups[0]["lr"] = new_lr_p
            #------------------------------------------------------------
            # fill buffer 
            # self.gater_data() # buff = [[[S,K+1,T],[S,K+1,T]..], [[S,K+1,T],[S,K+1,T]..], ]
            # start_w = time.time() 
            # self.world_optim.zero_grad()
            # self.policy_optim.zero_grad()

            # self.world.train()
            # for param in self.world.parameters():
            #         param.requires_grad =True
            # self.train_world_model_real(batch_size=batch_size_w, dt=self.dt) # Get average loss for ONE mini-batch
            # self.world_optim.zero_grad() # clear gradients from last iteration

            # for name, param in self.world.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}")
            # time.sleep(1000)
            # self.train_world_model_MB_CUDA(batch_size=batch_size_w)
            # self.train_world_model_MB_accum(batch_size=self.batch_size)
            # self.train_world_model(dt=self.dt)

            # print(f"WORLD TIME: {time.time()-start_w:.6f} seconds")
            # start_p = time.time() 
            
            # self.train_policy_model(dt=self.dt)
            # self.train_policy_model_accum()

            # self.world.eval()
            # for param in self.world.parameters():
            #         param.requires_grad =False
            # with autograd.detect_anomaly(): #!!!!
            self.train_policy_model_real(batch_size=batch_size_p, dt=self.dt)
            # self.policy_optim.zero_grad()

            # time.sleep(10000)
            # for name, param in self.policy.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}")
            # self.train_policy_model_MB_CUDA(batch_size=self.batch_size)
            # print(f"POLICY TIME: {time.time()-start_p:.6f} seconds")
            # p_loses.append(p_l)
            if self.t_so_far % 100 == 0:
                print('iter=',self.t_so_far) #'LR=',new_lr_p)

        # pbar.close()


    def act(self, w_states): # sample polcy 
        #* transform to local space
        l_state_s = self.env.to_local(w_states['state_s'])
        l_state_k = self.env.to_local(w_states['state_k'])
        # print(l_state_s[0])
        # print('####P####\n', l_state_s[0])
        # print('####V####\n', l_state_s[1])
        # print('####R####\n', l_state_s[2])
        # print('####RV####\n', l_state_s[3])
        # print('####H####\n', l_state_s[4])
        # print('####U####\n', l_state_s[5])
        # w_states_t = {'pos': torch.tensor(w_states['state_s']['pos'], dtype=torch.float32),'vel': torch.tensor(w_states['state_s']['vel'], dtype=torch.float32), 'rot': torch.tensor(w_states['state_s']['rot'], dtype=torch.float32),'rot_vel': torch.tensor(w_states['state_s']['rot_vel'], dtype=torch.float32)}
        # l_state_s_t = to_local_t(w_states_t)
        # print('@@@@P@@@@\n',l_state_s_t[0])
        # print('@@@@V@@@@\n',l_state_s_t[1])
        # print('@@@@R@@@@\n',l_state_s_t[2])
        # print('@@@@RV@@@@\n',l_state_s_t[3])
        # print('@@@@H@@@@\n',l_state_s_t[4])
        # print('@@@@U@@@@\n',l_state_s_t[5])
        #* normalize/scale features to stabilize training
        l_state_s = torch.cat([torch.tensor(l_state_s[0], dtype=torch.float32).flatten(), torch.tensor(l_state_s[1], dtype=torch.float32).flatten(), torch.tensor(l_state_s[2], dtype=torch.float32).flatten(), torch.tensor(l_state_s[3], dtype=torch.float32).flatten(), torch.tensor(l_state_s[4], dtype=torch.float32), torch.tensor(l_state_s[5], dtype=torch.float32)])
        l_state_k = torch.cat([torch.tensor(l_state_k[0], dtype=torch.float32).flatten(), torch.tensor(l_state_k[1], dtype=torch.float32).flatten(), torch.tensor(l_state_k[2], dtype=torch.float32).flatten(), torch.tensor(l_state_k[3], dtype=torch.float32).flatten(), torch.tensor(l_state_k[4], dtype=torch.float32), torch.tensor(l_state_k[5], dtype=torch.float32)])
        si_ki = torch.cat((l_state_s, l_state_k)).to(device)
        # print(si_ki)
        # l_si_pos, l_si_vel, l_si_rot, l_si_ang, l_si_height, l_si_up = to_local_git(torch.tensor(w_states['state_s']['pos']).unsqueeze(0).to(device),torch.tensor(w_states['state_s']['vel']).unsqueeze(0).to(device),torch.tensor(w_states['state_s']['rot']).unsqueeze(0).to(device),torch.tensor(w_states['state_s']['rot_vel']).unsqueeze(0).to(device), 1, device)
        # l_ki_pos, l_ki_vel, l_ki_rot, l_ki_ang, l_ki_height, l_ki_up = to_local_git(torch.tensor(w_states['state_k']['pos']).unsqueeze(0).to(device),torch.tensor(w_states['state_k']['vel']).unsqueeze(0).to(device),torch.tensor(w_states['state_k']['rot']).unsqueeze(0).to(device),torch.tensor(w_states['state_k']['rot_vel']).unsqueeze(0).to(device), 1, device)
        # print('POS@@@',l_ki_pos.shape,'\n','VEL@@@',l_ki_vel.shape,'\n','ROT@@@',l_ki_rot.shape,'\n','ROT_VEL@@@',l_ki_ang.shape,'\n','HEIGHT@@@',l_ki_height.shape,'\n','UP@@@',l_ki_up.shape,'\n') 
        # time.sleep(10000)
        # si_l = torch.cat([si_l[0].flatten(), si_l[1].flatten(), si_l[2].flatten(), si_l[3].flatten(), si_l[4], si_l[5]])
        # ki_l = torch.cat([ki_l[0].flatten(), ki_l[1].flatten(), ki_l[2].flatten(), ki_l[3].flatten(), ki_l[4], ki_l[5]])
        # si_ki = torch.cat((si_l, ki_l))
        # si_ki.requires_grad =True
        # si_ki = torch.cat([
        #     l_si_pos.reshape(1, -1),
        #     l_si_vel.reshape(1, -1),
        #     l_si_rot.reshape(1, -1),
        #     l_si_ang.reshape(1, -1),
        #     l_si_height.reshape(1, -1),
        #     l_si_up.reshape(1, -1),

        #     l_ki_pos.reshape(1, -1),
        #     l_ki_vel.reshape(1, -1),
        #     l_ki_rot.reshape(1, -1),
        #     l_ki_ang.reshape(1, -1),
        #     l_ki_height.reshape(1, -1),
        #     l_ki_up.reshape(1, -1)
        # ], dim=1) 
        # print('GITTTTTTTTTTTTTTTTT',l_ki_rot)
        # print('\nSIM STATE',(l_state_s))
        # print('\nKIN STATE',(l_state_k))
        # time.sleep(100000)
        # if self.w_mean is not None:
        #     # print("____#NORMALIZED#____")
        #     si_ki = (si_ki - self.p_mean) / (self.p_std + 1e-10)
        with torch.no_grad(): # no need for gradient 
            o = self.policy.forward(si_ki.unsqueeze(0))
            # noise = torch.randn_like(o) # Gaussian noise (zero mean, unit variance) 
            # o = o + self.noise_scale * noise
            # dist = MultivariateNormal(mean, self.cov_mat)
            # action = dist.sample()
            # torch.cuda.synchronize()
            # log_prob = dist.log_prob(action)
            # print(o)
        # print("TIME: {:.6f} seconds".format(time.time() - start))
        return np.asarray(o.to('cpu'))










    # def train_policy_model_MB_CUDA(self, batch_size, Nw=32, dt=1./60.):
    #     """
    #     Mini-batch training for the policy model
    #     """
    #     rand_eps = random.choices(self.memory, k=batch_size)  # Select batch_size random episodes
    #     batch_windows = []
    #     for rand_ep in rand_eps:
    #         start_idx = random.randint(0, len(rand_ep) - Nw)
    #         rand_window = rand_ep[start_idx : start_idx + Nw]
    #         batch_windows.append(rand_window) # [ Win[Fr[],Fr[],..],Win[[],[],..]...
    #     # print("\n###########", np.array(batch_windows).shape) # (32, 32, 3)
    #     #* Set initial predicted state
    #     batch_si = [{key: torch.tensor(win[0][0]['state_s'][key], device=device, dtype=torch.float32)
    #             for key in win[0][0]['state_s']} for win in batch_windows]  # (32, ) of P0's {'pos':...}
    #     # print("\n###########", np.array(batch_si).shape) 
    #     # print("\n###########", batch_si[0])
    #     # time.sleep(10000)
    #     Pred_states = [[] for _ in range(batch_size)]
    #     os_batch = [[] for _ in range(batch_size)]
    #     #* Predict P over a window of 𝑁Π frame
    #     for i in range(Nw-1): # 32 states in one go 32 times
    #         batch_ki = [{key: torch.tensor(win[i][0]['state_k'][key], device=device, dtype=torch.float32)
    #                     for key in win[i][0]['state_k']} for win in batch_windows]
    #         batch_k_rot = [torch.tensor(win[i][2], device=device, dtype=torch.float32) for win in batch_windows]
            
    #         si_l_batch = [flt_stat_t(to_local_t(si), self.s_means, self.s_stds) for si in batch_si]
    #         ki_l_batch = [flt_stat_t(to_local_t(ki), self.k_means, self.k_stds) for ki in batch_ki]

    #         si_ki_batch = [torch.cat((si_l, ki_l)) for si_l, ki_l in zip(si_l_batch, ki_l_batch)]
    #         si_ki_batch = torch.stack(si_ki_batch)  # Stack tensors for batch processing
    #         si_ki_batch.requires_grad = True
    #         #* Predict PD offset
    #         offsets = self.policy.forward(si_ki_batch)
    #         [os.append(offset) for os, offset in zip(os_batch, offsets)]
    #         #* Add noise to offset
    #         noise = torch.randn_like(offsets)  # Gaussian noise (zero mean, unit variance) 
    #         actions = offsets + self.noise_scale * noise
    #         # print("\n###########", actions.shape) # ([32, 63])
    #         # time.sleep(10000)
    #         #* Compute PD target
    #         action_vecs = actions.view(batch_size, 21, 3)
    #         exp_batch = torch.stack([quat_exp_t(o * self.offset_scale/2) for o in action_vecs.view(-1, 3)]).view(batch_size, 21, 4) # view(-1, 3)= (unifies all under in one) deletes batch demention for easier calcualtion, and adds it back 
    #         ti_batch = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp_batch, batch_k_rot)])      
    #         # print("\n###########", exp_batch.shape) # torch.Size([32, 22, 4]) 
    #         # time.sleep(10000)
    #         #* Pass through world model
    #         # ti_batch = actions.flatten(start_dim=1)  # Flatten each action set for world model input
    #         ti_batch = [(ti - self.s_target_mean) / (agent.s_target_std + 1e-10) for ti in ti_batch] # normalize
    #         # print("\n###########", ti_batch) 
    #         # time.sleep(10000)
    #         si_ti_batch = [torch.cat((si_l, ti.flatten())) for si_l, ti in zip(si_l_batch, ti_batch)] #
    #         si_ti_batch = torch.stack(si_ti_batch) 
    #         # print("\n###########", si_ti_batch.shape) # ([32, 439])
    #         # print("\n###########", si_ti_batch)
    #         # time.sleep(10000)
    #         #* Predict rigid body accelerations
    #         pred = self.world.forward(si_ti_batch)
    #         pos_a = pred[:, :66].reshape(batch_size, 22, 3)
    #         rot_a = pred[:, 66:].reshape(batch_size, 22, 3)
    #         #* Convert accelerations to world space
    #         pos_wa = torch.stack([quaternion_apply(si['rot'][0], v) for si, v in zip(batch_si, pos_a)])
    #         rot_wa = torch.stack([quaternion_apply(si['rot'][0], v) for si, v in zip(batch_si, rot_a)])
    #         #* Integrate rigid body accelerations to get final predicted state
    #         pred_vel = dt * pos_wa + torch.stack([si['vel'] for si in batch_si])
    #         pred_rot_vel = dt * rot_wa + torch.stack([si['rot_vel'] for si in batch_si])
    #         pred_pos = dt * pred_vel + torch.stack([si['pos'] for si in batch_si])
    #         exp = torch.stack([quat_exp_t(v * dt / 2) for v in pred_rot_vel.view(-1, 3)]).view(batch_size, 22, 4)
    #         pred_rot = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, torch.stack([si['rot'] for si in batch_si]))])

    #         for b in range(batch_size):
    #             Pred_states[b].append({'pos': pred_pos[b], 'vel': pred_vel[b], 'rot': pred_rot[b], 'rot_vel': pred_rot_vel[b]})
    #             batch_si[b] = {'pos': pred_pos[b].detach(), 'vel': pred_vel[b].detach(), 'rot': pred_rot[b].detach(), 'rot_vel': pred_rot_vel[b].detach()} 
    #     #* Compute losses over mini-batch
    #     window_loss = 0
    #     # print('\nOFFSET', len(os_batch[0][0])) #(32 batches, 31 frames, 63)
    #     for b in range(batch_size):
    #             for i in range(len(Pred_states[b])):
    #                 ki = batch_windows[b][i+1][0]['state_k']
    #                 ki = {key: torch.tensor(ki[key], device=device, dtype=torch.float32) for key in ki}
    #                 pi = Pred_states[b][i]
    #                 #* Compute Local Spaces
    #                 pi_l = to_local_t(pi)
    #                 ki_l = to_local_t(ki)
    #                 #* Compute losses in Local Space
    #                 pos_loss = self.wl * torch.sum(torch.abs(pi_l[0] - ki_l[0]))
    #                 vel_loss = self.wl * torch.sum(torch.abs(pi_l[1] - ki_l[1]))
    #                 rot_loss = self.wl * torch.sum(torch.abs(pi_l[2] - ki_l[2]))
    #                 rot_vel_loss = self.wl * torch.sum(torch.abs(pi_l[3] - ki_l[3]))
    #                 height_loss = self.wl * torch.sum(torch.abs(pi_l[4] - ki_l[4]))
    #                 up_loss = self.wl * torch.sum(torch.abs(pi_l[5] - ki_l[5]))

    #                 o2_loss = self.wl_s * torch.sum(torch.square(os_batch[b][i]))
    #                 o_loss = self.wl_s * torch.sum(torch.abs(os_batch[b][i]))

    #                 window_loss += pos_loss + vel_loss + rot_loss + rot_vel_loss + height_loss + up_loss + o2_loss + o_loss
        
    #     # Normalize loss across batch
    #     batch_loss = window_loss / batch_size
    #     print('total_p_loss#########    ', batch_loss)
    #     # print("@@@@@@@@@    ", window_loss.grad_fn)
    #     # print("@@@@@@@@@@@@@@    ",window_loss.grad_fn.next_functions)
    #     #* Update network paramete    
    #     self.policy_optim.zero_grad()
    #     window_loss.backward() 
    #     # for name, param in self.world.named_parameters():
    #     #     if param.grad is not None:
    #     #         print(f"{name}: {param.grad.norm().item()}")
    #     nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #     self.policy_optim.step()
    #     self.p_loss_history.append(batch_loss.detach().item())





    # def train_policy_model_accum(self, Nw=32, dt=1./60.):
    #     """
    #     one window at a time
    #     """
    #     rand_ep = random.choice(self.memory)
    #     start_idx = random.randint(0, len(rand_ep) - Nw)
    #     rand_window = rand_ep[start_idx : start_idx + Nw] # rand 32 frames in order [[FR states, s_targets],[FR states, s_targets]..]
    #     #* Set initial predicted state
    #     si = rand_window[0][0]['state_s'] # P0   [ME[EP[FR states, s_targets], [], []..
    #     si = {'pos': torch.tensor(si['pos'], dtype=torch.float32), 'vel': torch.tensor(si['vel'], dtype=torch.float32), 'rot': torch.tensor(si['rot'], dtype=torch.float32), 'rot_vel': torch.tensor(si['rot_vel'], dtype=torch.float32)}
    #     # k_rot = torch.tensor(rand_window[0][2])
    #     Pred_states = []
    #     os = []
    #     #* Predict P over a window of 𝑁Π frame
    #     for i in range(Nw-1):
    #         ki = rand_window[i][0]['state_k'] 
    #         ki = {'pos': torch.tensor(ki['pos'], dtype=torch.float32), 'vel': torch.tensor(ki['vel'], dtype=torch.float32), 'rot': torch.tensor(ki['rot'], dtype=torch.float32), 'rot_vel': torch.tensor(ki['rot_vel'], dtype=torch.float32)}
    #         k_rot = torch.tensor(rand_window[i][2], dtype=torch.float32)
            
    #         si_l = to_local_t(si)
    #         ki_l = to_local_t(ki)
    #         si_l = flt_stat_t(si_l, self.s_means, self.s_stds) 
    #         ki_l = flt_stat_t(ki_l, self.k_means, self.k_stds)
    #         si_ki = torch.cat((si_l, ki_l))
    #         si_ki.requires_grad =True
    #         #* Predict PD offset
    #         offsets = self.policy.forward(si_ki)
    #         os.append(offsets)
    #         #* Add noise to offset
    #         noise = torch.randn_like(offsets) # Gaussian noise (zero mean, unit variance) 
    #         action = offsets + self.noise_scale * noise
    #         # print("\n###########", action, '\n\n') 
    #         # print("\n###########", offsets) 
    #         # time.sleep(10000)
    #         #* Compute PD target
    #         action_vecs = action.reshape(21, 3)  
    #         exp = torch.stack([quat_exp_t(o * self.offset_scale/2) for o in action_vecs])
    #         ti = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, k_rot)]) # final pd targets 
    #         #* Pass through world model
    #         # print("\n###########", ti) 
    #         # time.sleep(10000)
    #         ti = (ti - self.s_target_mean) / (agent.s_target_std + 1e-10) # normalise
    #         si_ti = torch.cat((si_l, ti.flatten()))
    #         # Predict rigid body accelerations
    #         pred = self.world.forward(si_ti)
    #         pos_a = pred[:66].reshape(22, 3)
    #         rot_a = pred[66:].reshape(22, 3)
    #         #* Convert accelerations to world space
    #         pos_wa = torch.stack([quaternion_apply(si['rot'][0], v) for v in pos_a])
    #         rot_wa = torch.stack([quaternion_apply(si['rot'][0], v) for v in rot_a])
    #         #* Integrate rigid body accelerations to get final predicted state
    #         pred_vel = dt * pos_wa + si['vel']
    #         pred_rot_vel = dt * rot_wa + si['rot_vel']
    #         pred_pos = dt * pred_vel + si['pos']     
    #         exp = torch.stack([quat_exp_t(v * dt/2) for v in pred_rot_vel])
    #         pred_rot = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, si['rot'])])
            
    #         pred_state = {'pos': pred_pos, 'vel': pred_vel, 'rot': pred_rot, 'rot_vel': pred_rot_vel}
    #         si = {'pos': pred_pos.detach(), 'vel': pred_vel.detach(), 'rot': pred_rot.detach(), 'rot_vel': pred_rot_vel.detach()}
    #         Pred_states.append(pred_state) 
    #     self.policy_optim.zero_grad()
    #     # print('\nOFFSET',len(os)) #(31, )
    #     for i in range(len(Pred_states)):
    #         ki = rand_window[i+1][0]['state_k']
    #         ki = {'pos': torch.tensor(ki['pos'], dtype=torch.float32), 'vel': torch.tensor(ki['vel'], dtype=torch.float32), 'rot': torch.tensor(ki['rot'], dtype=torch.float32), 'rot_vel': torch.tensor(ki['rot_vel'], dtype=torch.float32)}
    #         pi = Pred_states[i]
    #         #* Compute Local Spaces
    #         pi_l = to_local_t(pi)
    #         ki_l = to_local_t(ki)
    #         #* Compute losses in Local Space
    #         # [local_pos, local_vel, local_rot, local_rot_vel, local_height, up_vec]
    #         pos_diff = torch.abs(pi_l[0] - ki_l[0]) # absolute difference between each pair of corresponding elements. [[x, x, x],[]..]
    #         pos_loss = self.wl*torch.sum(pos_diff) # sums along the axis to give the total L norm.

    #         vel_diff = torch.abs(pi_l[1] - ki_l[1]) 
    #         vel_loss = self.wl*torch.sum(vel_diff)
            
    #         rot_diff = torch.abs(pi_l[2] - ki_l[2])
    #         rot_loss = self.wl*torch.sum(rot_diff)

    #         rot_vel_diff = torch.abs(pi_l[3] - ki_l[3]) 
    #         rot_vel_loss = self.wl*torch.sum(rot_vel_diff)

    #         height_diff = torch.abs(pi_l[4] - ki_l[4]) 
    #         height_loss = self.wl*torch.sum(height_diff)

    #         up_diff = torch.abs(pi_l[5] - ki_l[5]) 
    #         up_loss = self.wl*torch.sum(up_diff)

    #         o2_loss = self.wl_s*torch.sum(torch.square(os[i]))  # Compute squared Euclidean norm / l2 norm
    #         o_loss = self.wl_s*torch.sum(torch.abs(os[i])) # l1 norm
    #         # print(f"Position Loss: {pos_loss.item()}, Velocity Loss: {vel_loss.item()}, Rotation Loss: {rot_loss.item()}, Offset Loss: {o_loss.item()}")
    #         frame_loss = pos_loss + vel_loss + rot_loss + rot_vel_loss + height_loss + up_loss + o2_loss + o_loss
    #         frame_loss.backward()
    #         if (i+1)%Nw-1 == 0:
    #             print('frame_loss#########    ',frame_loss)
    #     # print('#########    ')
    #     #* Update network paramete    
    #     nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #     self.policy_optim.step()





    # def train_policy_model_MB(self, batch_size, Nw=32, dt=1./60.):
    #     """
    #     Mini-batch training for the policy model
    #     """
    #     rand_eps = random.choices(self.memory, k=batch_size)  # Select batch_size random episodes
    #     batch_windows = []
    #     for rand_ep in rand_eps:
    #         start_idx = random.randint(0, len(rand_ep) - Nw)
    #         rand_window = rand_ep[start_idx : start_idx + Nw]
    #         batch_windows.append(rand_window) # [ Win[Fr[],Fr[],..],Win[[],[],..]...
    #     # print("\n###########", np.array(batch_windows).shape) # (32, 32, 3)
    #     #* Set initial predicted state
    #     batch_si = [{key: torch.tensor(win[0][0]['state_s'][key], dtype=torch.float32)
    #             for key in win[0][0]['state_s']} for win in batch_windows]  # (32, ) of P0's {'pos':...}
    #     # print("\n###########", np.array(batch_si).shape) 
    #     # print("\n###########", batch_si[0])
    #     # time.sleep(10000)
    #     Pred_states = [[] for _ in range(batch_size)]
    #     os_batch = [[] for _ in range(batch_size)]
    #     #* Predict P over a window of 𝑁Π frame
    #     for i in range(Nw-1): # 32 states in one go 32 times
    #         batch_ki = [{key: torch.tensor(win[i][0]['state_k'][key], dtype=torch.float32)
    #                     for key in win[i][0]['state_k']} for win in batch_windows]
    #         batch_k_rot = [torch.tensor(win[i][2]) for win in batch_windows]
            
    #         si_l_batch = [flt_stat_t(to_local_t(si), self.s_means, self.s_stds) for si in batch_si]
    #         ki_l_batch = [flt_stat_t(to_local_t(ki), self.k_means, self.k_stds) for ki in batch_ki]

    #         si_ki_batch = [torch.cat((si_l, ki_l)) for si_l, ki_l in zip(si_l_batch, ki_l_batch)]
    #         si_ki_batch = torch.stack(si_ki_batch)  # Stack tensors for batch processing
    #         si_ki_batch.requires_grad = True
    #         #* Predict PD offset
    #         # print("\nPOLICY_INPUT\n", si_ki_batch)
    #         offsets = self.policy.forward(si_ki_batch)
    #         [os.append(offset) for os, offset in zip(os_batch, offsets)]
    #         #* Add noise to offset
    #         noise = torch.randn_like(offsets)  # Gaussian noise (zero mean, unit variance) 
    #         actions = offsets + self.noise_scale * noise
    #         # print("\n###########", actions.shape) # ([32, 63])
    #         # time.sleep(10000)
    #         #* Compute PD target
    #         action_vecs = actions.view(batch_size, 21, 3)
    #         exp_batch = torch.stack([quat_exp_t(o * self.offset_scale / 2) for o in action_vecs.view(-1, 3)]).view(batch_size, 21, 4) # view(-1, 3)= (unifies all under in one) deletes batch demention for easier calcualtion, and adds it back 
    #         ti_batch = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp_batch, batch_k_rot)])      
    #         # print("\n###########", exp_batch.shape) # torch.Size([32, 22, 4]) 
    #         # time.sleep(10000)
    #         #* Pass through world model
    #         # ti_batch = actions.flatten(start_dim=1)  # Flatten each action set for world model input
    #         ti_batch = [(ti - self.s_target_mean) / (agent.s_target_std + 1e-10) for ti in ti_batch] # normalize
    #         # print("\n###########", ti_batch) 
    #         # time.sleep(10000)
    #         si_ti_batch = [torch.cat((si_l, ti.flatten())) for si_l, ti in zip(si_l_batch, ti_batch)] #
    #         si_ti_batch = torch.stack(si_ti_batch) 
    #         # print("\n###########", si_ti_batch.shape) # ([32, 439])
    #         # print("\n###########", si_ti_batch)
    #         # time.sleep(10000)
    #         #* Predict rigid body accelerations
    #         # print("\nWORD_INPUT\n", si_ti_batch)
    #         pred = self.world.forward(si_ti_batch)
    #         pos_a = pred[:, :66].reshape(batch_size, 22, 3)
    #         rot_a = pred[:, 66:].reshape(batch_size, 22, 3)
    #         #* Convert accelerations to world space
    #         pos_wa = torch.stack([quaternion_apply(si['rot'][0], v) for si, v in zip(batch_si, pos_a)])
    #         rot_wa = torch.stack([quaternion_apply(si['rot'][0], v) for si, v in zip(batch_si, rot_a)])
    #         #* Integrate rigid body accelerations to get final predicted state
    #         pred_vel = dt * pos_wa + torch.stack([si['vel'] for si in batch_si])
    #         pred_rot_vel = dt * rot_wa + torch.stack([si['rot_vel'] for si in batch_si])
    #         pred_pos = dt * pred_vel + torch.stack([si['pos'] for si in batch_si])
    #         exp = torch.stack([quat_exp_t(v * dt / 2) for v in pred_rot_vel.view(-1, 3)]).view(batch_size, 22, 4)
    #         pred_rot = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, torch.stack([si['rot'] for si in batch_si]))])

    #         for b in range(batch_size):
    #             Pred_states[b].append({'pos': pred_pos[b], 'vel': pred_vel[b], 'rot': pred_rot[b], 'rot_vel': pred_rot_vel[b]})
    #             batch_si[b] = {'pos': pred_pos[b].detach(), 'vel': pred_vel[b].detach(), 'rot': pred_rot[b].detach(), 'rot_vel': pred_rot_vel[b].detach()} 
    #     #* Compute losses over mini-batch
    #     window_loss = 0
    #     # print('\nOFFSET', len(os_batch[0][0])) #(32 batches, 31 frames, 63)
    #     for b in range(batch_size):
    #             for i in range(len(Pred_states[b])):
    #                 ki = batch_windows[b][i+1][0]['state_k']
    #                 ki = {key: torch.tensor(ki[key], dtype=torch.float32) for key in ki}
    #                 pi = Pred_states[b][i]
    #                 #* Compute Local Spaces
    #                 pi_l = to_local_t(pi)
    #                 ki_l = to_local_t(ki)
    #                 #* Compute losses in Local Space
    #                 pos_loss = self.wl * torch.sum(torch.abs(pi_l[0] - ki_l[0]))
    #                 vel_loss = self.wl * torch.sum(torch.abs(pi_l[1] - ki_l[1]))
    #                 rot_loss = self.wl * torch.sum(torch.abs(pi_l[2] - ki_l[2]))
    #                 rot_vel_loss = self.wl * torch.sum(torch.abs(pi_l[3] - ki_l[3]))
    #                 height_loss = self.wl * torch.sum(torch.abs(pi_l[4] - ki_l[4]))
    #                 up_loss = self.wl * torch.sum(torch.abs(pi_l[5] - ki_l[5]))

    #                 o2_loss = self.wl_s * torch.sum(torch.square(os_batch[b][i]))
    #                 o_loss = self.wl_s * torch.sum(torch.abs(os_batch[b][i]))

    #                 window_loss += pos_loss + vel_loss + rot_loss + rot_vel_loss + height_loss + up_loss + o2_loss + o_loss
        
    #     # Normalize loss across batch
    #     batch_loss = window_loss / batch_size
    #     print('total_p_loss#########    ', batch_loss)
    #     # print("@@@@@@@@@    ", window_loss.grad_fn)
    #     # print("@@@@@@@@@@@@@@    ",window_loss.grad_fn.next_functions)
    #     #* Update network paramete    
    #     self.policy_optim.zero_grad()
    #     window_loss.backward() 
    #     for name, param in self.policy.named_parameters():
    #         if param.grad is not None:
    #             print(f"{name}: {param.grad.norm().item()}")
    #     nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #     self.policy_optim.step()
    #     self.p_loss_history.append(batch_loss.detach().item())






    def train_policy_model_real(self, batch_size, dt, Nw=32):
        """
        real minibatch training
        """
        rand_eps = random.choices(self.memory, k=batch_size) 
        batch_windows = []
        for rand_ep in rand_eps:
            start_idx = random.randint(0, len(rand_ep) - Nw)
            rand_window = rand_ep[start_idx : start_idx + Nw]  
            batch_windows.append(rand_window)
        #* Set initial predicted state
        s_pos = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_s']['pos'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)
        # print(pos.shape) #torch.Size([32, 8, 22, 3])
        s_vel = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_s']['vel'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)

        s_rot = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_s']['rot'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)

        s_ang = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_s']['rot_vel'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device) 

        k_pos = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_k']['pos'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)
        # print(pos.shape) #torch.Size([32, 8, 22, 3])
        k_vel = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_k']['vel'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device) 

        k_rot = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_k']['rot'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)  

        k_ang = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_k']['rot_vel'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device) 


        k_t = torch.stack([  # kin current rotations 
            torch.stack([
                torch.tensor(frame[2], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)

        si_pos = s_pos[:, 0, :, :] #torch.Size([32, 22, 3])
        si_vel = s_vel[:, 0, :, :] 
        si_rot = s_rot[:, 0, :, :]
        si_ang = s_ang[:, 0, :, :]

        loss = 0
        #* Predict P over a window of 𝑁Π frame
        for i in range(Nw-1):
            # ki = rand_window[i][0]['state_k'] 
            # ki = {'pos': torch.tensor(ki['pos'], dtype=torch.float32), 'vel': torch.tensor(ki['vel'], dtype=torch.float32), 'rot': torch.tensor(ki['rot'], dtype=torch.float32), 'rot_vel': torch.tensor(ki['rot_vel'], dtype=torch.float32)}
            # k_rot = torch.tensor(rand_window[i][2], dtype=torch.float32)
            ki_pos = k_pos[:, i, :, :] #torch.Size([32, 22, 3])
            ki_vel = k_vel[:, i, :, :] 
            ki_rot = k_rot[:, i, :, :]
            ki_ang = k_ang[:, i, :, :]
            # print('ki_pos',ki_pos[0])
            # print('ki_vel',ki_vel[0])
            # print('ki_rot',ki_rot[0])
            # print('ki_ang',ki_ang[0])
            # si_l = to_local_t(si)
            # ki_l = to_local_t(ki)
            l_si_pos, l_si_vel, l_si_rot, l_si_ang, l_si_height, l_si_up = to_local_git(si_pos,si_vel,si_rot,si_ang, batch_size, device)
            l_ki_pos, l_ki_vel, l_ki_rot, l_ki_ang, l_ki_height, l_ki_up = to_local_git(ki_pos,ki_vel,ki_rot,ki_ang, batch_size, device)
            # print('POS@@@',l_ki_pos.shape,'\n','VEL@@@',l_ki_vel.shape,'\n','ROT@@@',l_ki_rot.shape,'\n','ROT_VEL@@@',l_ki_ang.shape,'\n','HEIGHT@@@',l_ki_height.shape,'\n','UP@@@',l_ki_up.shape,'\n') 
            # time.sleep(10000)
            # si_l = torch.cat([si_l[0].flatten(), si_l[1].flatten(), si_l[2].flatten(), si_l[3].flatten(), si_l[4], si_l[5]])
            # ki_l = torch.cat([ki_l[0].flatten(), ki_l[1].flatten(), ki_l[2].flatten(), ki_l[3].flatten(), ki_l[4], ki_l[5]])
            # si_ki = torch.cat((si_l, ki_l))
            # si_ki.requires_grad =True
            si_ki = torch.cat([
                l_si_pos.reshape(batch_size, -1),
                l_si_vel.reshape(batch_size, -1),
                l_si_rot.reshape(batch_size, -1),
                l_si_ang.reshape(batch_size, -1),
                l_si_height,
                l_si_up,

                l_ki_pos.reshape(batch_size, -1),
                l_ki_vel.reshape(batch_size, -1),
                l_ki_rot.reshape(batch_size, -1),
                l_ki_ang.reshape(batch_size, -1),
                l_ki_height,
                l_ki_up
            ], dim=1) 
            # print(l_si_pos.reshape(batch_size, -1))
            # print(l_si_vel.reshape(batch_size, -1))
            # print(l_si_rot.reshape(batch_size, -1))
            # print(l_si_ang.reshape(batch_size, -1))
            # print(l_si_height)
            # print(l_si_up)
            # print('####################\n\n\n\n')

            # print(si_ki.shape)
            # time.sleep(10000)
            # print("\n######INPUT#####", si_ki) # normalized inputs?
            # print("M/S:", torch.mean(si_ki),'\n',torch.std(si_ki))
            # if use_batchnorm == False:
                # print("____#NORMALIZED#____")
                # si_ki = (si_ki - self.p_mean) / (self.p_std + 1e-10)
            #* Predict PD offset
            # print("\n\n######INPUT#####", si_ki) 
            o = self.policy.forward(si_ki)
            # print("\n######OUTPUT#####", o[10]) #!
            # time.sleep(10000)
            # print("M/S:", torch.mean(offsets),'\n',torch.std(offsets))
            # os.append(offsets)
            #* Add noise to offset
            noise = torch.randn_like(o) # Gaussian noise (zero mean, unit variance) 
            o_hat = o + self.noise_scale * noise
            # print("\n###########", o_hat[10], '\n\n') 
            # print("\n###########", offsets) 
            #* Compute PD target
            o_hat = o_hat.reshape(batch_size, 21, 3)  
            exp = quat_exp_t(o_hat * self.offset_scale/2) #*
            ti = quaternion_raw_multiply(exp, k_t[:, i, :, :]) # final pd targets
            # print('AAAAAAAAAAAAAAAAAA',o_hat)
            # print("PRED ROT", ti[0],'\n') #!!!
            # print("TARGET ROT", k_t[:, i, :, :][0],'\n') #!!!

            #* Pass through world model
            # print("\n###########", ti) 
            # time.sleep(10000)
            # ti = (ti - self.s_target_mean) / (agent.s_target_std + 1e-10) # normalise
            # si_ti = torch.cat((si_l, ti.flatten()))
            si_ti = torch.cat([
                l_si_pos.reshape(batch_size, -1),
                l_si_vel.reshape(batch_size, -1),
                l_si_rot.reshape(batch_size, -1),
                l_si_ang.reshape(batch_size, -1),
                l_si_height,
                l_si_up,

                ti.reshape(batch_size, -1)
            ], dim=1) 
            # if use_batchnorm == False:
                # print("____#NORMALIZED#____")
                # si_ti = (si_ti - self.w_mean) / (self.w_std + 1e-10)
            # print("\n@@@@W_INPUT@@@@\n", si_ti[10])
            #* Predict rigid body accelerations
            pred = self.world.forward(si_ti)
            # print("\n@@@@OUTPUT@@@@", pred[10]) # normalized inputs?
            # if self.t_so_far==2 and i==1:time.sleep(10000)
            # time.sleep(10000)
            #* Convert accelerations to world space
            pos_a = pred[:, :66].reshape(batch_size, 22, 3) # (32, 22, 3)
            # print('AAA',pos_a[10])
            rot_a = pred[:, 66:].reshape(batch_size, 22, 3)
            #* Convert accelerations to world space
            pos_wa = quaternion_apply(si_rot[:, 0:1, :], pos_a)
            # print(pos_wa[10],'\n\n\n')
            rot_wa = quaternion_apply(si_rot[:, 0:1, :], rot_a)
            #* Integrate rigid body accelerations to get final predicted state
            # print('\n\n\nV',si_vel[0])
            # print('\nA',si_ang[0])
            # print('\nP',si_pos[0])
            # print('\nR',si_rot[0])
            si_vel = dt * pos_wa + si_vel
            si_ang = dt * rot_wa + si_ang
            si_pos = dt * si_vel + si_pos
            exp = quat_exp_t(si_ang * dt / 2)
            si_rot = quaternion_raw_multiply(exp, si_rot)
            # print(l_si_height)
            # print(l_si_up)
            # print('@@@@@@@@@@@@@@@@@@@@\n\n\n\n')
            # print('POS@@@',si_pos.shape,'\n','VEL@@@',si_vel.shape,'\n','ROT@@@',si_rot.shape,'\n','ROT_VEL@@@',si_ang.shape,'\n') 
            # time.sleep(10000)
            #* Compute Local Spaces
            l_si_pos, l_si_vel, l_si_rot, l_si_ang, l_si_height, l_si_up = to_local_git(si_pos,si_vel,si_rot,si_ang, batch_size, device)
            # l_ki_pos, l_ki_vel, l_ki_rot, l_ki_ang, l_ki_height, l_ki_up = to_local_git(k_pos,k_vel,k_rot,k_ang, batch_size, device)
            # print('POS@@@',l_si_pos[0][:6],'\n','VEL@@@',l_si_vel[0][:6],'\n','ROT@@@',l_si_rot[0][:6],'\n','ROT_VEL@@@',l_si_ang[0][:6],'\n','HEIGHT@@@',l_si_height[0][:6],'\n','UP@@@',l_si_up[0][:6],'\n')
            # print('POS@@@',l_si_pos.shape,'\n','VEL@@@',l_si_vel.shape,'\n','ROT@@@',l_si_rot.shape,'\n','ROT_VEL@@@',l_si_ang.shape,'\n','HEIGHT@@@',l_si_height.shape,'\n','UP@@@',l_si_up.shape,'\n') 
            # time.sleep(10000)
            #* Compute losses
            # print('losses\n\n', i)
            loss +=     0.1*torch.mean( torch.sum( torch.abs(l_si_pos - l_ki_pos), dim = -1)) # same as manual
            # print('PRED_POS',l_si_pos[0][:5])
            # print('TARGET_POS', l_ki_pos[0][:5])
            # print(loss,'\n')
            loss +=     0.1*torch.mean( torch.sum( torch.abs(l_si_vel - l_ki_vel), dim = -1))
            # print('PRED_vel',l_si_vel[0][:5])
            # print('TARGET_vel', l_ki_vel[0][:5])
            # print(loss,'\n')
            loss +=     0.1*torch.mean( torch.sum( torch.abs(l_si_rot - l_ki_rot), dim = -1))
            # print('PRED_rot',l_si_rot[0][:5])
            # print('TARGET_rot', l_ki_rot[0][:5])
            # print(loss,'r\n')
            loss +=     0.1*torch.mean( torch.sum( torch.abs(l_si_ang - l_ki_ang), dim = -1))
            # print('PRED_ang',l_si_ang[0][:5])
            # print('TARGET_ang', l_ki_ang[0][:5])
            # print(loss,'\n')
            loss +=     0.1*torch.mean( torch.sum( torch.abs(l_si_height - l_ki_height), dim = -1))
            # print('PRED_height',l_si_height[0][:5])
            # print('TARGET_height', l_ki_height[0][:5])            
            # print(loss,'\n')
            loss += 	0.1*torch.mean( torch.sum( torch.abs(l_si_up - l_ki_up), dim = -1))
            # print('PRED_up',l_si_up[0][:5])
            # print('TARGET_up', l_ki_up[0][:5])
            # print(loss,'\n')
            loss +=     0.02*torch.mean( torch.sum( torch.abs(o), dim = -1))
            # print('PRED_offset_L1',o[0][:5]) # penalize large PD offsets
            # print(loss,'\n')
            loss +=     0.02*torch.mean( torch.sum( torch.square(o), dim = -1))
            # print(loss,'\n\n')
            # if self.t_so_far==2 and i==2:time.sleep(10000)
        print('total_p_loss#########        ', loss/batch_size)       
        #* Update network parameters
        self.policy_optim.zero_grad() 
        # self.world_optim.zero_grad() #! useless as req_grads is set to false
        loss.backward()

        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm= 25.0) #!CLIP
        # self.plot_grad_flow2(self.policy.named_parameters())
        # print(pred.grad_fn)
        # for name, param in self.policy.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad.norm().item()}")

        # total_norm = 0.0
        # for p in self.policy.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f"[Grad Norm] Current total norm = {total_norm:.4f}") # L2 norm of all gradients across the model. avarage value 
        # print("\n######OUTPUT#####", o[10][:15])

        if self.t_so_far % 10 == 0:
            total_norm = 0.0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"[Grad Norm] Current total norm = {total_norm:.4f}") # L2 norm of all gradients across the model. avarage value 

        self.policy_optim.step()

        if self.t_so_far % 10 == 0:
            self.p_loss_history.append(loss.detach().item())
            # print('APPPPPPPPPPPPPPPPPPPPPPENDED')








    # def train_policy_model(self, dt, Nw=32): #*ALGORITHM 2
    #     """
    #     one window at a time
    #     """
    #     rand_ep = random.choices(self.memory, k=1) 
    #     start_idx = random.randint(0, len(rand_ep[0]) - Nw) #!
    #     rand_window = rand_ep[0][start_idx : start_idx + Nw] # rand 32 frames in order [[FR states, s_targets],[FR states, s_targets]..]
    #     #* Set initial predicted state
    #     si = rand_window[0][0]['state_s'] # P0   [ME[EP[FR states, s_targets, k_rot], [], []..
    #     si = {'pos': torch.tensor(si['pos'], dtype=torch.float32), 'vel': torch.tensor(si['vel'], dtype=torch.float32), 'rot': torch.tensor(si['rot'], dtype=torch.float32), 'rot_vel': torch.tensor(si['rot_vel'], dtype=torch.float32)}
    #     # k_rot = torch.tensor(rand_window[0][2])
    #     Pred_states = []
    #     os = []
    #     #* Predict P over a window of 𝑁Π frame
    #     for i in range(Nw-1):
    #         ki = rand_window[i][0]['state_k'] 
    #         ki = {'pos': torch.tensor(ki['pos'], dtype=torch.float32), 'vel': torch.tensor(ki['vel'], dtype=torch.float32), 'rot': torch.tensor(ki['rot'], dtype=torch.float32), 'rot_vel': torch.tensor(ki['rot_vel'], dtype=torch.float32)}
    #         k_rot = torch.tensor(rand_window[i][2], dtype=torch.float32)
            
    #         si_l = to_local_t(si)
    #         ki_l = to_local_t(ki)
    #         # si_l = flt_stat_t(si_l, self.s_means, self.s_stds) 
    #         # ki_l = flt_stat_t(ki_l, self.k_means, self.k_stds)
    #         si_l = torch.cat([si_l[0].flatten(), si_l[1].flatten(), si_l[2].flatten(), si_l[3].flatten(), si_l[4], si_l[5]])
    #         ki_l = torch.cat([ki_l[0].flatten(), ki_l[1].flatten(), ki_l[2].flatten(), ki_l[3].flatten(), ki_l[4], ki_l[5]])
    #         si_ki = torch.cat((si_l, ki_l))
    #         si_ki.requires_grad =True
    #         # print("\n######INPUT#####", si_ki) # normalized inputs?
    #         # print("M/S:", torch.mean(si_ki),'\n',torch.std(si_ki))
    #         #* Predict PD offset
    #         offsets = self.policy.forward(si_ki)
    #         # print("\n######OUTPUT#####", offsets) # normalized inputs?
    #         # print("M/S:", torch.mean(offsets),'\n',torch.std(offsets))
    #         os.append(offsets)
    #         #* Add noise to offset
    #         noise = torch.randn_like(offsets) # Gaussian noise (zero mean, unit variance) 
    #         action = offsets + self.noise_scale * noise
    #         # print("\n###########", action, '\n\n') 
    #         # print("\n###########", offsets) 
    #         # time.sleep(10000)
    #         #* Compute PD target
    #         action_vecs = action.reshape(21, 3)  
    #         exp = torch.stack([quat_exp_t(o * self.offset_scale/2) for o in action_vecs])
    #         ti = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, k_rot)]) # final pd targets 
    #         #* Pass through world model
    #         # print("\n###########", ti) 
    #         # time.sleep(10000)
    #         # ti = (ti - self.s_target_mean) / (agent.s_target_std + 1e-10) # normalise
    #         si_ti = torch.cat((si_l, ti.flatten()))
    #         # Predict rigid body accelerations
    #         # if self.w_mean is not None:
    #         #     # print("____#NORMALIZED#____")
    #         #     si_ti = (si_ti - self.w_mean) / (self.w_std + 1e-10)
    #         pred = self.world.forward(si_ti)
    #         pos_a = pred[:66].reshape(22, 3)
    #         rot_a = pred[66:].reshape(22, 3)
    #         #* Convert accelerations to world space
    #         pos_wa = torch.stack([quaternion_apply(si['rot'][0], v) for v in pos_a])
    #         rot_wa = torch.stack([quaternion_apply(si['rot'][0], v) for v in rot_a])
    #         #* Integrate rigid body accelerations to get final predicted state
    #         pred_vel = dt * pos_wa + si['vel']
    #         pred_rot_vel = dt * rot_wa + si['rot_vel']
    #         pred_pos = dt * pred_vel + si['pos']     
    #         exp = torch.stack([quat_exp_t(v * dt/2) for v in pred_rot_vel])
    #         pred_rot = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, si['rot'])])
            
    #         pred_state = {'pos': pred_pos, 'vel': pred_vel, 'rot': pred_rot, 'rot_vel': pred_rot_vel}
    #         si = {'pos': pred_pos.detach(), 'vel': pred_vel.detach(), 'rot': pred_rot.detach(), 'rot_vel': pred_rot_vel.detach()}
    #         Pred_states.append(pred_state) 
    #     window_loss = 0
    #     # print('\nOFFSET',len(os)) #(31, )
    #     for i in range(len(Pred_states)):
    #         ki = rand_window[i][0]['state_k'] # shouldnt be a i+1
    #         ki = {'pos': torch.tensor(ki['pos'], dtype=torch.float32), 'vel': torch.tensor(ki['vel'], dtype=torch.float32), 'rot': torch.tensor(ki['rot'], dtype=torch.float32), 'rot_vel': torch.tensor(ki['rot_vel'], dtype=torch.float32)}
    #         pi = Pred_states[i]
    #         #* Compute Local Spaces
    #         pi_l = to_local_t(pi)
    #         ki_l = to_local_t(ki)
    #         # print('POS@@@',pi_l[0][:6],'\n','VEL@@@',pi_l[1][:6],'\n','ROT@@@',pi_l[2][:6],'\n','ROT_VEL@@@',pi_l[3][:6],'\n','HEIGHT@@@',pi_l[4][:6],'\n','UP@@@',pi_l[5][:6],'\n')
    #         # test0 = self.env.to_local(ki)
    #         # print('\n\nLOCAL ',i, test0[5][1])
    #         # print('\n\nT-LOCAL-T',i, ki_l[5][1]) # identical
    #         # time.sleep(10000) 
    #         #* Compute losses in Local Space
    #         # [local_pos, local_vel, local_rot, local_rot_vel, local_height, up_vec]
    #         pos_diff = torch.abs(pi_l[0] - ki_l[0]) # absolute difference between each pair of corresponding elements. [[x, x, x],[]..]
    #         pos_loss = self.wl*torch.sum(pos_diff) # sums along the axis to give the total L norm.
    #         # loss = self.wl*torch.mean( torch.sum( torch.abs(pi_l[0] - ki_l[0]), dim = -1)) #!
    #         # print("\n#####pos_loss#####", loss)
    #         # time.sleep(10000)
    #         vel_diff = torch.abs(pi_l[1] - ki_l[1]) 
    #         vel_loss = self.wl*torch.sum(vel_diff)
    #         # print("\n#####vel_loss#####", vel_loss)
    #         rot_diff = torch.abs(pi_l[2] - ki_l[2])
    #         rot_loss = self.wl*torch.sum(rot_diff)
    #         # print("\n#####rot_loss#####", rot_loss)
    #         rot_vel_diff = torch.abs(pi_l[3] - ki_l[3]) 
    #         rot_vel_loss = self.wl*torch.sum(rot_vel_diff)
    #         # print("\n#####rot_vel_loss#####", rot_vel_loss)
    #         height_diff = torch.abs(pi_l[4] - ki_l[4]) 
    #         height_loss = self.wl*torch.sum(height_diff)
    #         # print("\n#####height_loss#####", height_loss)
    #         up_diff = torch.abs(pi_l[5] - ki_l[5]) 
    #         up_loss = self.wl*torch.sum(up_diff)
    #         # print("\n#####up_loss#####", up_loss)
    #         o2_loss = self.wl_s*torch.sum(torch.square(os[i]))  # Compute squared Euclidean norm / l2 norm
    #         o_loss = self.wl_s*torch.sum(torch.abs(os[i])) # l1 norm
    #         # print("\n#####o2_loss#####", o2_loss)
    #         # print("\n#####o_loss#####", o_loss)
    #         # print("\nO2_LOSS", o2_loss.grad_fn)
    #         # print("\nO_LOSS", o_loss.grad_fn)
    #         # print("\nUp", up_loss.grad_fn)
    #         # time.sleep(10000)
    #         # print(f"Position Loss: {pos_loss.item()}, Velocity Loss: {vel_loss.item()}, Rotation Loss: {rot_loss.item()}, Offset Loss: {o_loss.item()}")

    #         window_loss += pos_loss + vel_loss + rot_loss + rot_vel_loss + height_loss + up_loss + o2_loss + o_loss
    #     #* Update network paramete    
    #     print('total_p_loss#########    ',window_loss)
    #     # print("@@@@@@@@@    ", window_loss.grad_fn)
    #     # print("@@@@@@@@@@@@@@    ",window_loss.grad_fn.next_functions)
    #     self.policy_optim.zero_grad()
    #     window_loss.backward() 
    #     # for name, param in self.policy.named_parameters():
    #     #     if param.grad is not None:
    #     #         print(f"{name}: {param.grad.norm().item()}")
    #     nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #     self.policy_optim.step()

















    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.show()

    def plot_grad_flow2(self, named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.show()















    # def train_world_model(self, dt, Nw=8): #*ALGORITHM 1
    #     """
    #     one window at a time
    #     """
    #     # sample a window of random trajectories from buffer #!check MB
    #     # rand_ep = random.choice(self.memory)
    #     rand_ep = random.choices(self.memory, k=1) 
    #     start_idx = random.randint(0, len(rand_ep[0]) - Nw) #!
    #     rand_window = rand_ep[0][start_idx : start_idx + Nw] # (8, 3) rand 8 frames in order [[FR states, s_targets, k_rot],[FR states, s_targets, k_rot]..] #!
    #     #* Set initial predicted state
    #     pi = rand_window[0][0]['state_s'] # P0   [ME[EP[FR states[state_s, state_k], s_targets, k_rot], [], []..
    #     # print("\n######PI#####",pi)
    #     pi = {'pos': torch.tensor(pi['pos'], dtype=torch.float32),'vel': torch.tensor(pi['vel'], dtype=torch.float32), 'rot': torch.tensor(pi['rot'], dtype=torch.float32),'rot_vel': torch.tensor(pi['rot_vel'], dtype=torch.float32)} # not needed?
    #     Pred_states = []
    #     #* Predict P over a window of 𝑁W frames
    #     for i in range(Nw-1): 
    #         #-----------------------------------------------------------
    #         # if pi['pos'].grad_fn is not None: 
    #             # print('@@@@@@@@@@@@@@@')
    #         #     print(pi['pos'].grad_fn.next_functions,'\n')
    #         #     print(pi['pos'].grad_fn.next_functions[0][0].next_functions,'\n')
    #         #     print(pi['pos'].grad_fn.next_functions[0][0].next_functions[0][0].next_functions,'\n') 
    #         #     print(pi['pos'].grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions,'\n')   
    #         #     print(pi['pos'].grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)  
    #         #     print('\n\n')
    #         # for name, param in self.world.named_parameters():
    #         #     if param.requires_grad:
    #         #         print(f"{name}.grad:\n{param.grad}")
    #         # print('POS###',pi['pos'][:6],'\n','VEL###',pi['vel'][:6],'\n','ROT###',pi['rot'][:6],'\n','ROT_VEL###',pi['rot_vel'][:6],'\n')
    #         pi_l = to_local_t(pi) #!###############
    #         # print('POS@@@',pi_l[0][:6],'\n','VEL@@@',pi_l[1][:6],'\n','ROT@@@',pi_l[2][:6],'\n','ROT_VEL@@@',pi_l[3][:6],'\n','HEIGHT@@@',pi_l[4][:6],'\n','UP@@@',pi_l[5][:6],'\n')
    #         # time.sleep(10000)
    #         # print('@@@@@@@@@@@@@@@', pi['pos'].grad_fn)
    #         # pi_l = flt_stat_t(pi_l, self.s_means, self.s_stds)
    #         pi_l = torch.cat([pi_l[0].flatten(), pi_l[1].flatten(), pi_l[2].flatten(), pi_l[3].flatten(), pi_l[4], pi_l[5]])
    #         # print(pi_l.shape)
    #         # print("\n#####PI######", pi_l)
    #         # time.sleep(10000)
    #         ti = torch.tensor(rand_window[i][1], dtype=torch.float32) 
    #         # print("\n####TI######", ti)
    #         # ti = (ti - self.s_target_mean) / (agent.s_target_std + 1e-10) # normalize
    #         # print("\n####TI_NORM######", ti)
    #         pi_ti = torch.cat((pi_l, ti.flatten()))
    #         # pi_ti.requires_grad =True
    #         #-----------------------------------------------------------
    #         # print("\n######INPUT#####") 
    #         # if self.w_mean is not None:
    #         #     # print("____#NORMALIZED#____")
    #         #     pi_ti = (pi_ti - self.w_mean) / (self.w_std + 1e-10)
    #         #* Predict rigid body accelerations
    #         # print("\nINPUT###", pi_ti[:30]) # normalized inputs? #!#########
    #         # print("M/S OUT:", torch.mean(obs.detach()),'\n',torch.std(obs.detach()))
    #         # print(self.w_mean[:10])
    #         # print(self.std[:50],'\n\n\n')
    #         pred = self.world.forward(pi_ti) # 66+66 compute local rigid body positional and rotational accelerations
    #         # for name, act in self.world.activations.items():
    #         #     mean = act.mean().item()
    #         #     std = act.std().item()
    #         #     print(f"{name}: shape={act.shape}, mean={mean:.4f}, std={std:.4f}")
    #         # print('\n')
    #         # print("\nOUTPUT@@@", pred[:50]) # normalized inputs? #!#########
    #         # print("M/S OUT:", torch.mean(pred.detach()),'\n',torch.std(pred.detach()))
    #         # time.sleep(10000)
    #         pos_a = pred[:66].reshape(22, 3)
    #         rot_a = pred[66:].reshape(22, 3)
    #         # print("acceleration p:", pos_a,'\n\n', pi['vel'])
    #         #* Convert accelerations to world space
    #         pos_wa = torch.stack([quaternion_apply(pi['rot'][0], v) for v in pos_a])
    #         rot_wa = torch.stack([quaternion_apply(pi['rot'][0], v) for v in rot_a])
    #         #* Integrate rigid body accelerations to get final predicted state  
    #         # print('POS###',pi['pos'][:6],'\n','VEL###',pi['vel'][:6],'\n','ROT###',pi['rot'][:6],'\n','ROT_VEL###',pi['rot_vel'][:6],'\n')
    #         pred_vel = dt * pos_wa + pi['vel']
    #         pred_rot_vel = dt * rot_wa + pi['rot_vel']
    #         pred_pos = dt * pred_vel + pi['pos']     
    #         exp = torch.stack([quat_exp_t(v * dt/2) for v in pred_rot_vel])
    #         pred_rot = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, pi['rot'])])
    #         print(pred_vel[:2])
    #         print(pred_rot_vel[:2])
    #         print(pred_pos[:2])
    #         print(pred_rot[:2])
    #         # print("\nSAMPLE", pred_vel)     
    #         # print("\n###########", pos_wa)
    #         time.sleep(10000)
    #         pi = {'pos': pred_pos, 'vel': pred_vel, 'rot': pred_rot, 'rot_vel': pred_rot_vel}
    #         Pred_states.append(pi) # (8, )
    #         #pi = {'pos': pred_pos.detach(), 'vel': pred_vel.detach(), 'rot': pred_rot.detach(), 'rot_vel': pred_rot_vel.detach()}
    #     # print(np.array(Pred_states).shape) (7, )
    #     #* Compute losses
    #     window_loss = 0
    #     for i in range(len(Pred_states)):
    #         si = rand_window[i+1][0]['state_s']
    #         pi = Pred_states[i]
    #         # print('__SI__\n',si['rot_vel'],'\n')
    #         # print('__PI__\n',pi['rot_vel'],'\n')
    #         pos_diff = torch.abs(torch.tensor(si['pos']) - pi['pos']) # absolute difference between each pair of corresponding elements. [[x, x, x],[]..]
    #         pos_loss = self.wl*torch.sum(pos_diff) # sums along the axis to give the total L norm.
    #         # print("\n######pos_diff#####", pos_diff)
    #         # print("\n#####pos_loss#####", pos_loss)
    #         # print("\n#####SIM VEL#####", si['vel'])
    #         # print("\n#####PRED VEL#####", pi['vel'])
    #         # print("\n#####PRED VEL#####", pi['vel'])
    #         # print('##################pos_loss',pos_loss.grad_fn)
    #         vel_diff = torch.abs(torch.tensor(si['vel']) - pi['vel']) # with grad
    #         vel_loss = self.wl*torch.sum(vel_diff)
    #         # print("\n######vel_diff#####", vel_diff)
    #         # print("\n######vel_loss#####", vel_loss)
    #         # print('##################vel_loss',vel_loss.grad_fn)
    #         rot_diff = torch.stack([torch.abs(quat_log_t(quaternion_raw_multiply(q1, quaternion_inverse(torch.tensor(q0))) )) for q0, q1 in zip(si['rot'], pi['rot'])]) # q1 * inverse(q0) = diff
    #         rot_loss = self.wl*torch.sum(rot_diff)
    #         # print("\n######rot_diff#####", rot_diff)
    #         # print("\n#####rot_loss######", rot_loss)
    #         # print('##################rot_loss',rot_loss.grad_fn)
    #         rot_vel_diff = torch.abs(torch.tensor(si['rot_vel']) - pi['rot_vel']) # with grad
    #         rot_vel_loss = self.wl*torch.sum(rot_vel_diff)
    #         # print("\n######rot_vel_diff#####", rot_vel_diff)
    #         # print("\n#####rot_vel_loss######", rot_vel_loss)
    #         # print('##################rot_vel_loss',rot_vel_loss.grad_fn)
    #         window_loss += pos_loss + vel_loss + rot_loss + rot_vel_loss
    #     # pos_diff = self.wl*torch.sum([torch.abs(torch.tensor(si['pos']) - pi['pos']) for q1, q0 in zip(exp, pi['rot'])])
        
    #     #* Update network parameters
    #     print('total_w_loss#########    ',window_loss)
    #     # print("@@@@@@@@@    ", window_loss.grad_fn)
    #     # print("@@@@@@@@@@@@@@    ",window_loss.grad_fn.next_functions)
    #     self.world_optim.zero_grad()
    #     window_loss.backward() # retain_graph=True?
    #     # nn.utils.clip_grad_norm_(self.world.parameters(), self.max_grad_norm)
    #     # self.plot_grad_flow2(self.world.named_parameters()) #!#########
    #     # for name, param in self.world.named_parameters():
    #     #     if param.grad is not None:
    #     #         print(f"{name}: {param.grad.norm().item()}")
        

    #     self.world_optim.step()
    #     self.w_loss_history.append(window_loss.detach().item())






    def train_world_model_real(self, batch_size, dt, Nw=8):
        """
        real minibatch training
        """
        # Sample a batch of random trajectories    
        rand_eps = random.choices(self.memory, k=batch_size) 
        batch_windows = []
        for rand_ep in rand_eps:
            start_idx = random.randint(0, len(rand_ep) - Nw)
            rand_window = rand_ep[start_idx : start_idx + Nw]  
            batch_windows.append(rand_window)
        # print(np.array(batch_windows).shape) (batch_size, 8, 3)
        #* Set initial state
        # batch_s_states = [[{key: torch.tensor(wind[i][0]['state_s'][key], dtype=torch.float32) 
        #             for key in wind[i][0]['state_s']} for i in range(Nw)] 
        #             for wind in batch_windows] # (32, 8) where 8 - state_s of each frame as tensor
        
        #batch_windows[0][0][0]['state_s']['pos']
                #1wind|1fr| |states 
        #[32, 8, 21, 3]
        #[0-32w, 0-8f, 0-21
        # for i in range(500):
        s_pos = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_s']['pos'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)
        # print(pos.shape) #torch.Size([32, 8, 22, 3])
        s_vel = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_s']['vel'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)

        s_rot = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_s']['rot'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)

        s_ang = torch.stack([
            torch.stack([
                torch.tensor(frame[0]['state_s']['rot_vel'], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)

        s_t = torch.stack([ #torch.Size([32, 8, 21, 4])
            torch.stack([
                torch.tensor(frame[1], dtype=torch.float32)
                for frame in wind
            ]) for wind in batch_windows
        ]).to(device)
        # print(ti.shape)
        si_pos = s_pos[:, 0, :, :] #torch.Size([32, 22, 3])
        si_vel = s_vel[:, 0, :, :] 
        si_rot = s_rot[:, 0, :, :]
        si_ang = s_ang[:, 0, :, :]
        

        # print(s_pos)
        loss = 0
        for i in range(Nw-1):
        # #* Predict rigid body accelerations
            l_si_pos, l_si_vel, l_si_rot, l_si_ang, l_si_height, l_si_up = to_local_git(si_pos,si_vel,si_rot,si_ang, batch_size, device)
            # print('POS@@@',l_pos.shape,'\n','VEL@@@',l_vel.shape,'\n','ROT@@@',l_rot.shape,'\n','ROT_VEL@@@',l_ang.shape,'\n','HEIGHT@@@',l_height.shape,'\n','UP@@@',l_up.shape,'\n') 
            # ROT_VEL@@@ torch.Size([1, 22, 3])
            # HEIGHT@@@ torch.Size([1, 22])
            # UP@@@ torch.Size([1, 3])
            s_ti = s_t[:, i, :, :]
            # print(s_ti.shape)
            si_ti = torch.cat([
                l_si_pos.reshape(batch_size, -1),
                l_si_vel.reshape(batch_size, -1),
                l_si_rot.reshape(batch_size, -1),
                l_si_ang.reshape(batch_size, -1),
                l_si_height,
                l_si_up,
                s_ti.reshape(batch_size, -1)
            ], dim=1) 
            # print(pi_ti.shape)
            # if use_batchnorm == False:
                # print("____#NORMALIZED#____")
                # si_ti = (si_ti - self.w_mean) / (self.w_std + 1e-10)
            # print('##########SI_TI',si_ti0)
            # print('##############', si_ti0.mean())
            # print('@@@@@@@@@@@@@@', si_ti0.std())
            #* Predict rigid body accelerations
            # print('##########INPUT',si_ti[0][:5])
            pred = self.world.forward(si_ti)
            # print('##########PRED',pred[0])
            #* Convert accelerations to world space
            pos_a = pred[:, :66].reshape(batch_size, 22, 3) # (32, 22, 3)
            # print(pos_a)
            rot_a = pred[:, 66:].reshape(batch_size, 22, 3)

            pos_wa = quaternion_apply(si_rot[:, 0:1, :], pos_a)
            # print(pos_a)
            rot_wa = quaternion_apply(si_rot[:, 0:1, :], rot_a)
            #* Integrate rigid body accelerations to get final predicted state  
            # s_pos = s_pos + self.dtime*s_vel #[B, body, 3]
            si_vel = dt * pos_wa + si_vel
            si_ang = dt * rot_wa + si_ang
            si_pos = dt * si_vel + si_pos
            exp = quat_exp_t(si_ang * dt / 2)
            si_rot = quaternion_raw_multiply(exp, si_rot)
            #* Compute losses
            # print('W_losses\n\n')
            loss += self.wl*torch.mean( torch.sum( torch.abs(s_pos[:, i+1, :, :]-si_pos), dim = -1)) 
            # print('III  ',i)
            # print('\n\n',si_pos.shape) # torch.Size([1024, 22, 3])
            # print('PRED_POS',si_pos[0][:5])
            # print('TARGET_POS', s_pos[:, i+1, :, :][0][:5])
            # print(loss,'\n')
            loss += self.wl*torch.mean( torch.sum( torch.abs(s_vel[:, i+1, :, :]-si_vel), dim = -1)) 
            # print('PRED_vel',si_vel[0][:5])
            # print('TARGET_vel', s_vel[:, i+1, :, :][0][:5])
            # print(loss,'\n')
            loss += self.wl*torch.mean( torch.sum( torch.abs(2.0 * quat_log_t(quaternion_raw_multiply (si_rot, quaternion_inverse(s_rot[:, i+1, :, :]) ) ) ), dim = -1) )
            # print('PRED_rot',si_rot[0][:5])
            # print('TARGET_rot', s_rot[:, i+1, :, :][0][:5])
            # print(loss,'\n')
            loss += self.wl*torch.mean( torch.sum( torch.abs(s_ang[:, i+1, :, :]-si_ang), dim = -1)) 
            # print('PRED_ang',si_ang[0][:5])
            # print('TARGET_ang', s_ang[:, i+1, :, :][0][:5])
            # print(loss,'\n')
            # time.sleep(10000)
        print('total_w_loss#########    ', loss/batch_size)  
        #* Update network parameters
        self.world_optim.zero_grad()
        loss.backward()

        # nn.utils.clip_grad_norm_(self.world.parameters(), max_norm=0.1) #!CLIP
        # total_norm = 0.0
        # for p in self.world.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f"[Grad Norm] Current total norm = {total_norm:.4f}") # L2 norm of all gradients across the model. avarage value 

        # self.plot_grad_flow2(self.world.named_parameters())
        # for name, param in self.world.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad.norm().item()}")
        self.world_optim.step()

        if self.t_so_far % 10 == 0:
            total_norm = 0.0
            for p in self.world.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"[Grad Norm] Current total norm = {total_norm:.4f}") # L2 norm of all gradients across the model. avarage value 

        if self.t_so_far % 10 == 0:
            self.w_loss_history.append(loss.detach().item())
            # print('APPPPPPPPPPPPPPPPPPPPPPENDED')




    # def train_world_model_MB(self, batch_size, dt, Nw=8):
    #     """
    #     Mini-batch training for the world model.
    #     """
    #     # Sample a batch of random trajectories    
    #     rand_eps = random.choices(self.memory, k=batch_size) 
    #     batch_windows = []
    #     for rand_ep in rand_eps:
    #         start_idx = random.randint(0, len(rand_ep) - Nw)
    #         rand_window = rand_ep[start_idx : start_idx + Nw]  
    #         batch_windows.append(rand_window) # [ Win[Fr[],Fr[],..],Win[[],[],..]...
    #     # print("\n###########", np.array(batch_windows).shape) # (32, 8, 3)
    #     #* Set initial state
    #     batch_s_states = [[{key: torch.tensor(wind[i][0]['state_s'][key], dtype=torch.float32) 
    #                         for key in wind[i][0]['state_s']} for i in range(Nw)] 
    #                         for wind in batch_windows] # (32, 8) where 8 - state_s of each frame as tensor
    #     # print("\n###########", batch_windows[0][0][0]['state_s']['pos'])
    #     # print("\n###########", batch_windows[0][0])
    #     pi_batch = [{key: batch_s_states[b][0][key] for key in batch_s_states[b][0]} for b in range(batch_size)] #(32, ) of P0's {'pos':...}
    #     # print("\n###########", pi_batch[0])
    #     # time.sleep(10000)
    #     Pred_states = [[] for _ in range(batch_size)]  # (32, 0)
    #     #* Predict P over a window of 𝑁w frames
    #     for i in range(Nw-1): # 32 states in one go 8 times    (32 windows - 8 frames)
    #         # pi_l_batch = [flt(self.env.to_local(pi)) for pi in pi_batch]
    #         pi_l_batch = [(to_local_t(pi)) for pi in pi_batch] 
    #         pi_l_batch = [torch.cat([pi_l[0].flatten(), pi_l[1].flatten(), pi_l[2].flatten(), pi_l[3].flatten(), pi_l[4], pi_l[5]]) for pi_l in pi_l_batch]
    #         # print("\n###########", (pi_l_batch[0]))
    #         # time.sleep(10000) 
    #         ti_batch = [torch.tensor(batch_windows[w][i][1], dtype=torch.float32) for w in range(batch_size)]
    #         # print("\n###########", (ti_batch))
    #         # time.sleep(10000) 
    #         # ti_batch = [(ti - self.s_target_mean) / (agent.s_target_std + 1e-10) for ti in ti_batch] # normalize
    #         # print("\n#####s_target_mean######", ti_batch)
    #         # time.sleep(10000)
    #         pi_ti_batch = torch.stack([torch.cat((pi_l, ti.flatten())) for pi_l, ti in zip(pi_l_batch, ti_batch)])
    #         # pi_ti_batch.requires_grad = True
    #         # print("\n###########", pi_ti_batch.shape) # normalized inputs?
    #         # print("Mean of normalized tensor:", torch.mean(pi_ti_batch[0]))
    #         # print("Standard deviation of normalized tensor:", torch.std(pi_ti_batch[0]))  
    #         # time.sleep(10000)
    #         if self.w_mean is not None:
    #             # print("____#NORMALIZED#____")
    #             pi_ti_batch = torch.stack([(pi_ti - self.w_mean) / (self.w_std + 1e-10) for pi_ti in pi_ti_batch])
    #         #* Predict rigid body accelerations
    #         # print("\nINPUT###", pi_ti_batch[0][:10]) # normalized inputs? #!#########
    #         # print("M/S OUT:", torch.mean(pi_ti_batch.detach()),'\n',torch.std(pi_ti_batch.detach()))
    #         # print(self.w_mean[:10])
    #         # print(self.std[:50],'\n\n\n')
    #         pred = self.world.forward(pi_ti_batch)  # (batch_size, 132)
    #         # print("\nOUTPUT@@@", pred[0][:50]) # ([16, 132]) #!#########
    #         # print("M/S OUT:", torch.mean(pred.detach()),'\n',torch.std(pred.detach()))
    #         # time.sleep(10000)
    #         pos_a = pred[:, :66].reshape(batch_size, 22, 3) # (32, 22, 3)
    #         rot_a = pred[:, 66:].reshape(batch_size, 22, 3)
    #         # print("\n###########", pos_a.shape)
    #         # time.sleep(10000)
    #         #* Convert accelerations to world space
    #         pos_wa = torch.stack([quaternion_apply(pi['rot'][0], v) for pi, v in zip(pi_batch, pos_a)]) # (32, 22, 3)
    #         rot_wa = torch.stack([quaternion_apply(pi['rot'][0], v) for pi, v in zip(pi_batch, rot_a)])
    #         # print("\n###########", pos_wa.shape)
    #         # time.sleep(10000)
    #         #* Integrate rigid body accelerations to get final predicted state  
    #         pred_vel = dt * pos_wa + torch.stack([pi['vel'] for pi in pi_batch]) # 
    #         pred_rot_vel = dt * rot_wa + torch.stack([pi['rot_vel'] for pi in pi_batch])
    #         pred_pos = dt * pred_vel + torch.stack([pi['pos'] for pi in pi_batch])
    #         exp = torch.stack([quat_exp_t(v * dt / 2) for v in pred_rot_vel.view(-1, 3)]).view(batch_size, 22, 4) # view(-1, 3) = (batch_size * 22, 3) for esier calc; (batch_size, 22, 4) brings batch dim back
    #         pred_rot = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, torch.stack([pi['rot'] for pi in pi_batch]))])
    #         # print("\nBATCH", pred_vel[0])            
    #         # print("\n###########", exp.shape) # torch.Size([32, 22, 4]) 
    #         # time.sleep(10000)
    #         for w in range(batch_size):
    #             Pred_states[w].append({'pos': pred_pos[w], 'vel': pred_vel[w], 'rot': pred_rot[w], 'rot_vel': pred_rot_vel[w]})  # save predictions  
    #             pi_batch[w] = {'pos': pred_pos[w], 'vel': pred_vel[w], 'rot': pred_rot[w], 'rot_vel': pred_rot_vel[w]}
    #     # print("\n###########", np.array(Pred_states).shape) # (32, {}*7)
    #     # time.sleep(10000)
    #     #* Compute losses over mini-batch
    #     window_loss = 0
    #     for w in range(batch_size):
    #         for i in range(len(Pred_states[w])):
    #             si = batch_s_states[w][i+1]
    #             pi = Pred_states[w][i]

    #             pos_loss = self.wl * torch.sum(torch.abs(si['pos'] - pi['pos']))
    #             vel_loss = self.wl * torch.sum(torch.abs(si['vel'] - pi['vel']))
    #             rot_diff = torch.stack([torch.abs(quat_log_approx_t(quaternion_raw_multiply(q1, quaternion_inverse(q0)))) for q0, q1 in zip(si['rot'], pi['rot'])])
    #             rot_loss = self.wl * torch.sum(rot_diff)
    #             rot_vel_loss = self.wl * torch.sum(torch.abs(si['rot_vel'] - pi['rot_vel']))

    #             window_loss += pos_loss + vel_loss + rot_loss + rot_vel_loss

    #     # Normalize loss across batch
    #     batch_loss = window_loss/batch_size # so that loss does not grow with the batch size
    #     print('total_w_loss#########    ',batch_loss)
    #     # print("@@@@@@@@@    ", batch_loss.grad_fn)
    #     # print("@@@@@@@@@@@@@@    ",batch_loss.grad_fn.next_functions)
    #     #* Update network parameters
    #     self.world_optim.zero_grad()
    #     batch_loss.backward()
    #     self.plot_grad_flow2(self.world.named_parameters())
    #     for name, param in self.world.named_parameters():
    #         if param.grad is not None:
    #             print(f"{name}: {param.grad.norm().item()}")
    #     # nn.utils.clip_grad_norm_(self.world.parameters(), max_norm=self.max_grad_norm)
    #     self.world_optim.step()

    #     self.w_loss_history.append(batch_loss.detach().item())






    # def train_world_model_MB_accum(self, batch_size, Nw=8, dt=1./60.):
    #     """
    #     Mini-batch training for the world model.
    #     """
    #     # Sample a batch of random trajectories
    #     rand_eps = random.choices(self.memory, k=batch_size) 
    #     batch_windows = []
    #     for rand_ep in rand_eps:
    #         start_idx = random.randint(0, len(rand_ep) - Nw)
    #         rand_window = rand_ep[start_idx : start_idx + Nw]  
    #         batch_windows.append(rand_window) # [ Win[Fr[],Fr[],..],Win[[],[],..]...
    #     # print("\n###########", np.array(batch_windows).shape) # (32, 8, 3)
    #     #* Set initial state
    #     batch_s_states = [[{key: torch.tensor(wind[i][0]['state_s'][key], dtype=torch.float32) 
    #                         for key in wind[i][0]['state_s']} for i in range(Nw)] 
    #                         for wind in batch_windows] # (32, 8) where 8 - state_s of each frame as tensor
    #     # print("\n###########", batch_windows[0][0][0]['state_s']['pos'])
    #     # print("\n###########", batch_windows[0][0])
    #     pi_batch = [{key: batch_s_states[b][0][key] for key in batch_s_states[b][0]} for b in range(batch_size)] #(32, ) of P0's {'pos':...}
    #     # print("\n###########", pi_batch[0])
    #     # time.sleep(10000)
    #     Pred_states = [[] for _ in range(batch_size)]  # (32, 0)
    #     #* Predict P over a window of 𝑁w frames
    #     for i in range(Nw-1): # 32 states in one go 8 times
    #         # pi_l_batch = [flt(self.env.to_local(pi)) for pi in pi_batch]
    #         pi_l_batch = [flt_stat_t(to_local_t(pi), self.s_means, self.s_stds) for pi in pi_batch] 
    #         # print("\n###########", (pi_l_batch[0]))
    #         # time.sleep(10000) 
    #         ti_batch = [torch.tensor(batch_windows[w][i][1], dtype=torch.float32) for w in range(batch_size)]
    #         # print("\n###########", (ti_batch))
    #         # time.sleep(10000) 
    #         ti_batch = [(ti - self.s_target_mean) / (agent.s_target_std + 1e-10) for ti in ti_batch] # normalize
    #         # print("\n###########", self.s_target_mean)
    #         # time.sleep(10000)
    #         pi_ti_batch = [torch.cat((pi_l, ti.flatten())) for pi_l, ti in zip(pi_l_batch, ti_batch)]
    #         pi_ti_batch = torch.stack(pi_ti_batch)  # Stack tensors for batch processing
    #         pi_ti_batch.requires_grad = True
    #         # print("\n###########", pi_ti_batch.shape) # normalized inputs?
    #         # print("Mean of normalized tensor:", torch.mean(pi_ti_batch[0]))
    #         # print("Standard deviation of normalized tensor:", torch.std(pi_ti_batch[0]))  
    #         # time.sleep(10000)
    #         #* Predict rigid body accelerations
    #         pred = self.world.forward(pi_ti_batch)  # (batch_size, 132)
    #         pos_a = pred[:, :66].reshape(batch_size, 22, 3) # (32, 22, 3)
    #         rot_a = pred[:, 66:].reshape(batch_size, 22, 3)
    #         # print("\n###########", pos_a.shape)
    #         # time.sleep(10000)
    #         #* Convert accelerations to world space
    #         pos_wa = torch.stack([quaternion_apply(pi['rot'][0], v) for pi, v in zip(pi_batch, pos_a)]) # (32, 22, 3)
    #         rot_wa = torch.stack([quaternion_apply(pi['rot'][0], v) for pi, v in zip(pi_batch, rot_a)])
    #         # print("\n###########", pos_wa.shape)
    #         # time.sleep(10000)
    #         #* Integrate rigid body accelerations to get final predicted state  
    #         pred_vel = dt * pos_wa + torch.stack([pi['vel'] for pi in pi_batch]) # 
    #         pred_rot_vel = dt * rot_wa + torch.stack([pi['rot_vel'] for pi in pi_batch])
    #         pred_pos = dt * pred_vel + torch.stack([pi['pos'] for pi in pi_batch])
    #         exp = torch.stack([quat_exp_t(v * dt / 2) for v in pred_rot_vel.view(-1, 3)]).view(batch_size, 22, 4) # view(-1, 3) = (batch_size * 22, 3) for esier calc; (batch_size, 22, 4) brings batch dim back
    #         pred_rot = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, torch.stack([pi['rot'] for pi in pi_batch]))])
    #         # print("\n###########", pred_rot_vel.shape) # torch.Size([32, 22, 3])             
    #         # print("\n###########", exp.shape) # torch.Size([32, 22, 4]) 
    #         # time.sleep(10000)
    #         for b in range(batch_size):
    #             Pred_states[b].append({'pos': pred_pos[b], 'vel': pred_vel[b], 'rot': pred_rot[b], 'rot_vel': pred_rot_vel[b]})  # save predictions  
    #             pi_batch[b] = {'pos': pred_pos[b].detach(), 'vel': pred_vel[b].detach(), 'rot': pred_rot[b].detach(), 'rot_vel': pred_rot_vel[b].detach()}
    #     # print("\n###########", np.array(Pred_states).shape) # (32, {}*7)
    #     # time.sleep(10000)
    #     #______________________________________inefficient accum__________________________________________________________
    #     #* Compute losses over mini-batch
    #     self.world_optim.zero_grad()
    #     for b in range(batch_size):
    #         window_loss = 0
    #         for i in range(len(Pred_states[b])):
    #             si = batch_s_states[b][i+1]
    #             pi = Pred_states[b][i]

    #             pos_loss = self.wl * torch.sum(torch.abs(si['pos'] - pi['pos']))
    #             vel_loss = self.wl * torch.sum(torch.abs(si['vel'] - pi['vel']))
    #             rot_diff = torch.stack([torch.abs(quat_log_approx_t(quaternion_raw_multiply(q1, quaternion_inverse(q0)))) for q0, q1 in zip(si['rot'], pi['rot'])])
    #             rot_loss = self.wl * torch.sum(rot_diff)
    #             rot_vel_loss = self.wl * torch.sum(torch.abs(si['rot_vel'] - pi['rot_vel']))
    #             # print("@@@@@@@@@    ",  pi['pos'].grad_fn, b, i)
    #             window_loss += pos_loss + vel_loss + rot_loss + rot_vel_loss

    #         window_loss.backward(retain_graph=True)
    #     print('frame_loss#########    ',window_loss)
    #     #* Update network parameters
    #     nn.utils.clip_grad_norm_(self.world.parameters(), max_norm=self.max_grad_norm)
    #     self.world_optim.step()
    #     #________________________________________________________________________________________________



        
    # def train_world_model_MB_CUDA(self, batch_size, Nw=8, dt=1./60.):
    #         """
    #         Mini-batch training for the world model with CUDA support.
    #         """
    #         # device = torch.device("cuda" if torch.cuda.is_available() else print('CUDA IS NOT AVALIABLE'))

    #         # Sample a batch of random trajectories
    #         rand_eps = random.choices(self.memory, k=batch_size) 
    #         batch_windows = []
    #         for rand_ep in rand_eps:
    #             start_idx = random.randint(0, len(rand_ep) - Nw)
    #             rand_window = rand_ep[start_idx : start_idx + Nw]  
    #             batch_windows.append(rand_window) # [ Win[Fr[states, s_targets, k_rot],Fr[],..],Win[[],[],..]...
    #         # print("\n###########", np.array(batch_windows).shape) # (32, 8, 3)
    #         # time.sleep(10000)
    #         #* Set initial state
    #         batch_s_states = [[{key: torch.tensor(wind[i][0]['state_s'][key], device=device, dtype=torch.float32) 
    #                             for key in wind[i][0]['state_s']} for i in range(Nw)] 
    #                             for wind in batch_windows] # (32, 8) where 8 - state_s of each frame as tensor
    #         # ti_batch = [[torch.tensor(wind[i][1], device=device, dtype=torch.float32) for i in range(Nw)] for wind in batch_windows]
    #         # print("\n###########", batch_windows[0][0][0]['state_s']['pos'])
    #         # print("\n###########", len(ti_batch[0]))
    #         pi_batch = [{key: batch_s_states[b][0][key] for key in batch_s_states[b][0]} for b in range(batch_size)] #(32, ) of P0's {'pos':...} array of dictionaries
    #         # ti_batch = [[torch.tensor(frame[1].flatten(), dtype=torch.float32, device=device) for frame in wind[:-1]] for wind in batch_windows]
    #         # print("\n###########", pi_batch[0:2])
    #         # time.sleep(10000)
    #         Pred_states = [[] for _ in range(batch_size)]  # (32, 0) 
    #         #* Predict P over a window of 𝑁w frames
    #         for i in range(Nw-1): # 32 states in one go 8 times
    #             # print("\n###########", pi_batch[0:2])
    #             pi_l_batch = torch.stack([flt_stat_t(to_local_t(pi), self.s_means, self.s_stds) for pi in pi_batch]) 
    #             # ti_batch = [ti_batch[w][i] for w in range(batch_size)]
    #             ti_batch = torch.stack([torch.tensor(batch_windows[w][i][1], dtype=torch.float32) for w in range(batch_size)]).to(device) 
    #             # print("\n###########", len(ti_batch[0]))
    #             # time.sleep(10000)
    #             # pi_l_batch = [pi_l.to(device) for pi_l in pi_l_batch]
    #             # ti_batch = [ti.to(device) for ti in ti_batch]
    #             ti_batch = (ti_batch - self.s_target_mean) / (agent.s_target_std + 1e-10) # normalize 
    #             # time.sleep(10000)
    #             pi_ti_batch = torch.cat((pi_l_batch, ti_batch.view(batch_size, -1)), dim=1) 
    #             # pi_ti_batch = torch.stack(pi_ti_batch)  # Stack tensors for batch processing
    #             pi_ti_batch.requires_grad = True
    #             print("\n###########", pi_ti_batch.shape)
    #             # time.sleep(10000)
    #             #* Predict rigid body accelerations
    #             pred = self.world.forward(pi_ti_batch)  # (batch_size, 132) 
    #             pos_a = pred[:, :66].reshape(batch_size, 22, 3) # (32, 22, 3)
    #             rot_a = pred[:, 66:].reshape(batch_size, 22, 3)
    #             # print("\n###########", pos_a.device)
    #             # time.sleep(10000)
    #             #* Convert accelerations to world space
    #             pos_wa = torch.stack([quaternion_apply(pi_batch[w]['rot'][0], v) for w, v in enumerate(pos_a)]) # (32, 22, 3) 
    #             rot_wa = torch.stack([quaternion_apply(pi_batch[w]['rot'][0], v) for w, v in enumerate(rot_a)])
    #             # print("\n###########", pos_wa.shape)
    #             # time.sleep(10000)
    #             #* Integrate rigid body accelerations to get final predicted state  
    #             pred_vel = dt * pos_wa + torch.stack([pi['vel'] for pi in pi_batch]) # 
    #             pred_rot_vel = dt * rot_wa + torch.stack([pi['rot_vel'] for pi in pi_batch]) 
    #             pred_pos = dt * pred_vel + torch.stack([pi['pos'] for pi in pi_batch]) 
    #             exp = torch.stack([quat_exp_t(v * dt / 2) for v in pred_rot_vel.view(-1, 3)]).view(batch_size, 22, 4) # view(-1, 3) = (batch_size * 22, 3) ... -> (batch_size, 22, 4) 
    #             pred_rot = torch.stack([quaternion_raw_multiply(q1, q0) for q1, q0 in zip(exp, torch.stack([pi['rot'] for pi in pi_batch]))]) 
    #             # print("\n###########", pred_rot_vel.shape) # torch.Size([32, 22, 3])             
    #             # print("\n###########", exp.shape) # torch.Size([32, 22, 4]) 
    #             # time.sleep(10000)
    #             for b in range(batch_size): 
    #                 Pred_states[b].append({'pos': pred_pos[b], 'vel': pred_vel[b], 'rot': pred_rot[b], 'rot_vel': pred_rot_vel[b]})  # save predictions  
    #                 pi_batch[b] = {'pos': pred_pos[b].detach(), 'vel': pred_vel[b].detach(), 'rot': pred_rot[b].detach(), 'rot_vel': pred_rot_vel[b].detach()}
    #         # print("\n###########", np.array(Pred_states).shape) # (32, {}*7)
    #         # time.sleep(10000)
    #         #* Compute losses over mini-batch
    #         window_loss = 0
    #         for b in range(batch_size): 
    #             for i in range(len(Pred_states[b])):
    #                 si = batch_s_states[b][i+1]
    #                 pi = Pred_states[b][i]

    #                 pos_loss = self.wl * torch.sum(torch.abs(si['pos'] - pi['pos']))
    #                 vel_loss = self.wl * torch.sum(torch.abs(si['vel'] - pi['vel']))
    #                 rot_diff = torch.stack([torch.abs(quat_log_approx_t(quaternion_raw_multiply(q1, quaternion_inverse(q0)))) for q0, q1 in zip(si['rot'], pi['rot'])]) #!##########
    #                 rot_loss = self.wl * torch.sum(rot_diff)
    #                 rot_vel_loss = self.wl * torch.sum(torch.abs(si['rot_vel'] - pi['rot_vel']))

    #                 window_loss += pos_loss + vel_loss + rot_loss + rot_vel_loss

    #         # si_pos = torch.stack([batch_s_states[b][i+1]['pos'] for b in range(batch_size) for i in range(Nw-1)])
    #         # pi_pos = torch.stack([Pred_states[b][i]['pos'] for b in range(batch_size) for i in range(Nw-1)])
    #         # pos_loss = self.wl * torch.sum(torch.abs(si_pos - pi_pos))

    #         # si_vel = torch.stack([batch_s_states[b][i+1]['vel'] for b in range(batch_size) for i in range(Nw-1)])
    #         # pi_vel = torch.stack([Pred_states[b][i]['vel'] for b in range(batch_size) for i in range(Nw-1)])
    #         # vel_loss = self.wl * torch.sum(torch.abs(si_vel - pi_vel))

    #         # si_rot = torch.stack([batch_s_states[b][i+1]['rot'] for b in range(batch_size) for i in range(Nw-1)])
    #         # pi_rot = torch.stack([Pred_states[b][i]['rot'] for b in range(batch_size) for i in range(Nw-1)])
    #         # rot_loss = self.wl * torch.sum(torch.abs(quat_log_approx_t(quaternion_raw_multiply(si_rot, quaternion_inverse(pi_rot)))))

    #         # si_rot_vel = torch.stack([batch_s_states[b][i+1]['rot_vel'] for b in range(batch_size) for i in range(Nw-1)])
    #         # pi_rot_vel = torch.stack([Pred_states[b][i]['rot_vel'] for b in range(batch_size) for i in range(Nw-1)])
    #         # rot_vel_loss = self.wl * torch.sum(torch.abs(si_rot_vel - pi_rot_vel))

    #         # window_loss = pos_loss + vel_loss + rot_loss + rot_vel_loss
    #         # batch_loss = window_loss / batch_size




    #         # Normalize loss across batch
    #         batch_loss = window_loss/batch_size # so that loss does not grow with the batch size
    #         # print("\n###########", batch_loss.device)
    #         # time.sleep(10000)
    #         print('total_w_loss#########    ',batch_loss)
    #         # print("@@@@@@@@@    ", batch_loss.grad_fn)
    #         # print("@@@@@@@@@@@@@@    ",batch_loss.grad_fn.next_functions)
    #         #* Update network parameters
    #         self.world_optim.zero_grad()
    #         batch_loss.backward()
    #         # for name, param in self.world.named_parameters():
    #         #     if param.grad is not None:
    #         #         print(f"{name}: {param.grad.norm().item()}")
    #         #         print(f"{name}: {param.grad.device}")
    #         print('\n',torch.cuda.memory_allocated())
    #         print(torch.cuda.memory_reserved())
    #         nn.utils.clip_grad_norm_(self.world.parameters(), max_norm=self.max_grad_norm)
    #         self.world_optim.step()

    #         self.w_loss_history.append(batch_loss.detach().cpu().item())
















    def gater_data(self, n): # 1 fill buffer | sim rollout
        samples = 0
        i=0
        while samples < n: # fill up the buffer with episodes
            print('\nSAMLES=',samples)
            ep = [] # [[S,K+1,T],[S,K+1,T]..]
            states, info = self.env.reset() # get initial local state and state_k
            done = False
            episode_start = time.time() 
            # print("STATES_STATES_STATES_STATES_STATES_STATES",states)
            # time.sleep(1000)
            for _ in range(self.max_ep_len): # simulate and gather episode data
                #* sample policy for action
                o_hat = self.act(states) # ndarray
                o_hat = o_hat.reshape(21, 3)  
                # norm = np.linalg.norm((o_hat * self.offset_scale), axis=-1, keepdims=True)
                # print(norm)
                ti = quat_exp(o_hat * self.offset_scale / 2) #*
                # print(ti,'\n\n####')
                # time.sleep(1000)
                action = {'rot': ti}               

                next_states, _, done, info = self.env.step(action) #* apply action| retrive newe state
                s_targets = info.get("s_targets") # get new targets
                k_rot = info.get("k_rot") 
                ep.append([states, s_targets, k_rot]) # append S, K, T, t to episode (everything is nparrays)
                states = next_states

                if done: 
                    i=i+1          
                    break
                    
            # print('LEN##',len(ep))
            if (len(ep) >= self.min_ep_len): 
                samples+=len(ep)
                self.memory.append(ep) 
                mean_time = (time.time() - episode_start) / len(ep)
                MT_STAT.append(len(ep))
                print(f"MEAN TIME: {mean_time:.6f} seconds, i: {i}")
            

    def save(self, path):
        torch.save({
            'world_state_dict': self.world.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'world_optim_state_dict': self.world_optim.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'p_bn': self.policy.bn.state_dict(),
            'w_bn': self.world.bn.state_dict(),
            'w_loss_history': self.w_loss_history,
            'p_loss_history': self.p_loss_history,
            # 'hyperparameters': self._get_hyperparameters()  # Save any additional hyperparameters
            # Save agent-specific variables
            # 's_means': self.s_means,
            # 's_stds': self.s_stds,
            # 'k_means': self.k_means,
            # 'k_stds': self.k_stds,
            # 's_target_mean': self.s_target_mean,
            # 's_target_std': self.s_target_std

            # 'p_mean': self.p_mean,
            # 'p_std': self.p_std,
            # 'w_mean': self.w_mean,
            # 'w_std': self.w_std
        }, path)
        print("DATA IS SAVED TO A FILE")

    def save_m(self, path):
        torch.save({
            "memory" : self.memory
        }, path)
        print("MEMORY IS SAVED TO A FILE")
    def load(self, path, hist=True):
        # Load the state of the world, policy, and optimizers
        checkpoint = torch.load(path)
        self.world.load_state_dict(checkpoint['world_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict']) #!
        self.world_optim.load_state_dict(checkpoint['world_optim_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict']) #!
        self.policy.bn.load_state_dict(checkpoint['p_bn']) #!
        self.world.bn.load_state_dict(checkpoint['w_bn'])

        # statistics
        # self.p_mean = checkpoint['p_mean'].to(device)
        # self.p_std = checkpoint['p_std'].to(device)
        # self.w_mean = checkpoint['w_mean'].to(device)
        # self.w_std = checkpoint['w_std'].to(device)
        
        if hist == True:
            self.w_loss_history = checkpoint['w_loss_history']
            self.p_loss_history = checkpoint['p_loss_history']
        print("PARAMETERS_ARE_LOADED")

    def load_m(self, path):
        # Load the state of the world, policy, and optimizers
        checkpoint = torch.load(path)
        self.memory = checkpoint['memory']
        print("MEMORY_IS_LOADED")
        # self._set_hyperparameters(checkpoint['hyperparameters'])  # Load any additional hyperparameters


    # def _get_hyperparameters(self):
    # # Return the hyperparameters as a dictionaryf
    #     return {
    #     'timesteps_per_batch': self.timesteps_per_batch,
    #     'max_timesteps_per_episode': self.max_timesteps_per_episode,
    #     'gamma': self.gamma,          
    #     'n_updates_per_iteration': self.n_updates_per_iteration,
    #     'lr': self.lr,
    #     'clip': self.clip,
    # }

    # def _set_hyperparameters(self, hyperparameters):
    #     # Set the hyperparameters from a dictionary
    #     self.timesteps_per_batch = hyperparameters['timesteps_per_batch']
    #     self.max_timesteps_per_episode = hyperparameters['max_timesteps_per_episode']
    #     self.gamma = hyperparameters['gamma']
    #     self.n_updates_per_iteration = hyperparameters['n_updates_per_iteration']
    #     self.lr = hyperparameters['lr']
    #     self.clip = hyperparameters['clip']

import supertrack


env = gym.make("SuperTrack-v0", render_mode="human")
# env = gym.make("SuperTrack-v0")
agent = SUPERTRACK(env)



# agent.gater_data()
# agent.load('supertrack_test.pt')  # load without memory
# agent.load('supertrack_v02.pt')
# agent.w_loss_history = []
# agent.gater_data()
# agent.load_m('supertrack_v1_10mem.pt')  # load without memory
# agent.policy.mean, agent.policy.std, agent.world.mean, agent.world.std = global_means_std()

# print(agent.s_means)
# time.sleep(1000)
# agent.learn(1)
# agent.gater_data(mode='evaluate') # use statistics 
torch.set_printoptions(threshold=torch.inf, precision=6)


agent.load_m('supertrack_60test1000.pt') #!
# agent.load_m('supertrack_60test1000_2.pt') #!  
# agent.load_m('supertrack_stage0_buff50.pt') 
# agent.load_m('supertrack_stage0_buff50_2.pt')
# agent.p_mean, agent.p_std, agent.w_mean, agent.w_std = global_means_std()
# print("MANUAL\n\n",agent.w_mean)
# print("STD\n\n",agent.w_std)
# print('##########MEAN_STD',agent.p_mean,'\n\n\n #############', agent.p_std)

# agent.memory.clear() # clear buffer
# agent.world.train()
# agent.load('supertrack_W45000.pt', hist=False) #!
# agent.load('supertrack_P5000.pt', hist=False)

# agent.load('supertrack_W_test.pt', hist=False)
agent.load('supertrack_P_test.pt', hist=True)
# print("BATCHNORM\n\n",agent.policy.bn.running_mean)
# time.sleep(10000)
#*-----
# # offsets = 0.
agent.world.eval()
agent.policy.eval()
MT_STAT=[]

agent.gater_data(2000)
# print(MT_STAT)
# agent.world.eval()
# for param in agent.world.parameters():
#         param.requires_grad =False
#*-----
# print("BATCHNORM_W\n\n",agent.world.bn.running_mean) #! 
# print("BATCHNORM_P\n\n",agent.policy.bn.running_mean) #!
# agent.learn(400, batch_size_w=256, batch_size_p=256) #1024*8-timesteps_per_batch

# for i in range(1):
    # agent.gater_data()
    # agent.p_mean, agent.p_std, agent.w_mean, agent.w_std = global_means_std()
    # agent.learn(10000, batch_size_w=1024, batch_size_p=512) # world 32 mb
    # agent.memory.clear() # clear buffer


#STEPS
# agent.save_m('supertrack_stage0_buff.pt') # 1.9/15000      -I
# agent.save('supertrack_stage0_W.pt') # 1000
# agent.save('supertrack_stage0_W1.pt') # 1000
# agent.save('supertrack_stage0_W5.pt') # 4000
# agent.save('supertrack_stage0_W6.pt') # 1000
# agent.save_m('supertrack_stage0_buff50.pt') # 50000 +5mocaps
# agent.save('supertrack_stage0_W7.pt') # 1000
# agent.save('supertrack_stage0_W9.pt') # 4000
# agent.save('supertrack_stage0_W10.pt') # 10000
# agent.save('supertrack_stage0_W11.pt') # 10000
# agent.save_m('supertrack_buff1.pt') #! 1.9/15000  
# agent.save('supertrack_test_buff1.pt') # 50/1buff TESTED
# agent.save('supertrack_test_buff1_E.pt') # EVALUATED TESTS
# agent.save('supertrack_test_buff1_E50.pt') # EVALUATED TESTS
# agent.save('supertrack_stage0_W12.pt') # 10000

# agent.save('supertrack_stage0_W13.pt') # 1000
# agent.save('supertrack_stage0_W13_clip.pt') # 0.1
# agent.save('supertrack_stage0_W13_512.pt') 
# agent.save('supertrack_stage0_W13_1024_01.pt') 

# agent.save_m('supertrack_stage0_buff50_2.pt') #! 50000

#*----
# agent.save('supertrack_stage0_W15_1024.pt') #15000 buff1-2-1 +
# agent.save('supertrack_stage0_P15_1024.pt') 
# agent.save('supertrack_stage0_W16_1024.pt')
# agent.save('supertrack_stage0_P16_1024.pt')
#*---
# agent.save_m('supertrack_buff50_1bvh.pt') #!
# agent.save('supertrack_W100000_1024.pt') #*
# agent.save('supertrack_P25000_1024.pt') #! is fucked
# agent.save('supertrack_W10000_1024_2.pt') # 10000 buff50_1 + 10000 supertrack_stage0_buff50_2
# agent.save('supertrack_P10000_1024_2.pt')
# agent.save('supertrack_P10000_1024_3.pt') #10000
# agent.save_m('supertrack_buff50_2bvh.pt') #! with kin vel data

# agent.save('supertrack_W10000_1024_02.pt') #* new + kin vel data 10+10
# agent.save('supertrack_P10000_1024_01.pt') #* new 

# agent.save_m('supertrack_buff50.pt') #! 
# agent.save('supertrack_W45000.pt')

# agent.save_m('supertrack_60test1000_2.pt') 
# agent.save('supertrack_W_test.pt')
# agent.save('supertrack_P_test.pt')

# print(np.array(agent.memory[1]).shape) # \ (10episodes, 95-97frames,3) 


