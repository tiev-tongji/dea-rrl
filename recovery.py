from model import DeterministicPolicy, GaussianPolicy
from utils import get_device

import torch
import os

class Recovery(object):
    def __init__(self, num_inputs, action_space, safety_critic, args):
        # General Settings
        self.device = get_device()

        # IQL
        self.use_iql    = args.use_iql_recovery
        self.clip_score = args.clip_score_recovery
        self.iql_beta   = args.iql_beta_recovery
        self.iql_awr    = args.iql_awr_recovery

        # Critic Settings
        self.critic = safety_critic

        # Policy Settings
        if args.recovery_policy == "Gaussian":
            self.alpha = args.alpha
            self.automatic_entropy_tuning = args.automatic_entropy_tuning
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs,
                action_space.shape[0],
                args.hidden_size,
                action_space).to(self.device)
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=args.lr)

    def update_parameters(self, memory, batch_size):
        ''' Update Recovery Policy '''
        state_batch, action_batch, _, _, _ = memory.sample(
            batch_size=batch_size, pos_fraction=0)

        state_batch = torch.FloatTensor(state_batch).to(self.device)

        if self.use_iql:
            # Use AWR(advantage weighted regression)
            if self.iql_awr:
                if True:
                    # Sample actions from data
                    actions = torch.FloatTensor(action_batch).to(self.device)
                    log_pi = self.policy.logprob(state_batch, actions)
                else:
                    # Sample actions from policy
                    actions, log_pi, _ = self.policy.sample(state_batch)
                    
                with torch.no_grad():
                    qf = torch.max(*self.critic.critic(state_batch, actions))
                    vf = self.critic.value(state_batch)
                    exp_adv = torch.exp(self.iql_beta * (vf - qf))
                    # if self.clip_score is not None:
                    #     exp_adv = torch.clamp(exp_adv, max=self.clip_score)
                policy_loss = (-log_pi * exp_adv).mean()
            # Use APG(advantage policy gradient)
            else:
                pi, log_pi, _ = self.policy.sample(state_batch)
                qf = torch.max(*self.critic.critic(state_batch, pi))
                vf = self.critic.value(state_batch)
                policy_loss = (self.alpha * log_pi + (qf - vf) * 10).mean()
        else:
            pi, log_pi, _ = self.policy.sample(state_batch)
            qf = self.critic(state_batch, pi)
            policy_loss = (self.alpha * log_pi + qf).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval is False:
            action, _, _ = self.policy.sample(state, eval)
        else:
            _, _, action = self.policy.sample(state, eval)
        return action.detach().cpu().numpy()[0]

    def select_actions(self, states, eval=False):
        states = torch.FloatTensor(states).to(self.device)
        if eval is False:
            actions, _, _ = self.policy.sample(states, eval)
        else:
            _, _, actions = self.policy.sample(states, eval)
        return actions.detach().cpu().numpy()

    def get_value(self, states, actions=None):
        '''
            Arguments:
                states, actions --> list of states and list of corresponding 
                actions to get Q_risk values for
            Returns: Q_risk(states, actions)
        '''
        with torch.no_grad():
            if actions == None:
                _, _, actions = self.policy.sample(states)
            return self.critic.get_value(states, actions)

    def get_state_dict(self):
        save_dict = {'recovery_policy_state_dict': self.policy.state_dict(),
                     'recovery_policy_optimizer_state_dict': self.policy_optimizer.state_dict()}
        if self.critic is not None:
            save_dict.update(self.critic.get_state_dict())
        return save_dict

    # Save model parameters
    def save_checkpoint(self, ckpt_path=None):
        print('Saving recovery models to {}'.format(ckpt_path))
        save_dict = self.get_state_dict()
        torch.save(save_dict, ckpt_path)

    def load_state_dict(self, checkpoint):
        if self.critic is not None:
            self.critic.load_state_dict(checkpoint)
        self.policy.load_state_dict(checkpoint['recovery_policy_state_dict'])
        self.policy_optimizer.load_state_dict(
            checkpoint['recovery_policy_optimizer_state_dict'])

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading recovery models from {}'.format(ckpt_path))
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(checkpoint)

            if evaluate:
                self.eval()
            else:
                self.train()

    def train(self):
        self.policy.train()
        if self.critic is not None:
            self.critic.train()

    def eval(self):
        self.policy.eval()
        if self.critic is not None:
            self.critic.eval()
