import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from utils import soft_update, hard_update, get_device, linear_schedule, zeros, zeros_like, rand, get_device, quantile_regression_loss
from model import GaussianPolicy, DeterministicPolicy, VNetwork, DoubleQNetwork, QuantileMlp, FlattenMlp, softmax
from risk import distortion_de, muti_distortion_de


class BehaviorAgent(object):
    def __init__(self, num_inputs, action_space, args):
        self.device = get_device()

        # RL Setting
        self.gamma = args.gamma
        self.eps_safe = args.eps_safe

        # Updating Setting
        self.updates = 0
        self.target_update_interval = args.target_update_interval
        self.tau = args.tau

        # Policy Setting
        if args.policy == "Gaussian":
            self.alpha = args.alpha
            self.automatic_entropy_tuning = args.automatic_entropy_tuning
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - \
                    torch.prod(torch.Tensor(
                        action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(
                    1, requires_grad=True, device=self.device)
                self.alpha_optimizer = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optimizer = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optimizer = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, *args):
        raise NotImplementedError

    def select_actions(self, *args):
        raise NotImplementedError

    def update_parameters(self, *args):
        raise NotImplementedError

    def get_state_dict(self):
        raise NotImplementedError

    # Save model parameters
    def save_checkpoint(self, ckpt_path=None):
        print('Saving SAC models to {}'.format(ckpt_path))
        save_dict = self.get_state_dict()
        torch.save(save_dict, ckpt_path)

    def load_state_dict(self, checkpoint):
        raise NotImplementedError

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading SAC models from {}'.format(ckpt_path))
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            self.load_state_dict(checkpoint)

            if evaluate:
                self.eval()
            else:
                self.train()
        else:
            print('Invalid path!')

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class SAC(BehaviorAgent):
    def __init__(self, num_inputs, action_space, safety_critic, args):
        super(SAC, self).__init__(num_inputs, action_space, args)

        print('Use SAC Agent')
        # Critic Setting
        self.critic = DoubleQNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = DoubleQNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        lr_rate = 100000 / args.num_steps
        # LR  : Lagrangian Relaxation
        self.use_lr = args.use_lr
        if self.use_lr:
            print('use Lagrangian Relaxation')
            self.nu = args.nu
            self.log_nu = torch.tensor(
                np.log(self.nu), requires_grad=True, device=self.device)
            self.nu_optimizer = Adam([self.log_nu], lr=args.lr * lr_rate)

        # SQRL: Safety Q-Functions for RL
        self.use_sqrl = args.use_sqrl
        self.safe_samples = args.safe_samples

        # RSPO: Risk Sensitive Policy Optimization
        self.use_rspo = args.use_rspo
        if self.use_rspo:
            if True or args.nu_schedule:
                self.nu_schedule = linear_schedule(
                    args.nu_start, args.nu_end, args.num_steps)
            else:
                self.nu_schedule = linear_schedule(args.nu, args.nu, 0)

        # RCPO: Critic Penalty Reward Constrained Policy Optimization
        self.use_rcpo = args.use_rcpo
        if self.use_rcpo:
            self.rcpo_lambda = args.rcpo_lambda
            self.log_rcpo_lambda = torch.tensor(
                np.log(self.rcpo_lambda), requires_grad=True, device=self.device)
            # Make lambda update slower for stability
            self.rcpo_lambda_optimizer = Adam(
                [self.log_rcpo_lambda], lr=args.lr*lr_rate)

        # Safety Critic
        if self.use_lr or self.use_rcpo or self.use_rspo or self.use_sqrl:
            self.safety_critic = safety_critic
        else:
            self.safety_critic = None

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # Action Sampling for SQRL
        if self.use_sqrl:
            state_batch = state.repeat(self.safe_samples, 1)
            pi, log_pi, _ = self.policy.sample(state_batch)
            max_qf_constraint_pi = self.safety_critic.get_value(
                state_batch, pi)

            thresh_idxs = (max_qf_constraint_pi <=
                           self.eps_safe).nonzero()[:, 0]
            # Note: these are auto-normalized
            thresh_probs = torch.exp(log_pi[thresh_idxs])
            thresh_probs = thresh_probs.flatten()

            if list(thresh_probs.size())[0] == 0:
                min_q_value_idx = torch.argmin(max_qf_constraint_pi)
                action = pi[min_q_value_idx, :].unsqueeze(0)
            else:
                prob_dist = torch.distributions.Categorical(thresh_probs)
                sampled_idx = prob_dist.sample()
                action = pi[sampled_idx, :].unsqueeze(0)
        # Action Sampling for all other algorithms
        else:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def select_actions(self, states, eval=False):
        if self.use_rspo:
            return np.array(list(map(lambda x: self.select_action(x, eval), states)))
        else:
            states = torch.FloatTensor(states).to(self.device)
            if eval is False:
                actions, _, _ = self.policy.sample(states)
            else:
                _, _, actions = self.policy.sample(states)
            return actions.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size, updates):
        param = 0
        param1 = 0

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(
            reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        ''' Update Critic '''
        with torch.no_grad():

            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_qf_next_target = torch.min(
                qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * \
                self.gamma * (min_qf_next_target)

            if self.use_rcpo:
                qsafe_batch = self.safety_critic(state_batch, action_batch)
                next_q_value -= self.rcpo_lambda * qsafe_batch

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        ''' Update Policy '''
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if self.use_lr or self.use_rspo:
            if self.use_lr:
                nu = self.nu
            else:
                nu = self.nu_schedule(updates)
            max_sqf_pi = self.safety_critic(state_batch, pi)
            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) + μ * (Qc(st,π(f(εt;st)|st)) - ε) - Q(st,f(εt;st))]
            policy_loss = ((self.alpha * log_pi) + nu *
                           (max_sqf_pi - self.eps_safe).clamp(0., 1.) - 1. * min_qf_pi).mean()
        else:
            # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        ''' Update Alpha '''
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi +
                           self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        ''' Update Nu for LR '''
        if self.use_lr:
            nu_loss = (self.log_nu *
                       (self.eps_safe - max_sqf_pi).detach()).mean()
            self.nu_optimizer.zero_grad()
            nu_loss.backward()
            self.nu_optimizer.step()
            self.nu = self.log_nu.exp()

            param = self.nu.item()
            param1 = max_sqf_pi.mean().item()  # self.nu.item()

        ''' Update Lambda for RCPO '''
        if self.use_rcpo:
            rcpo_lambda_loss = (self.log_rcpo_lambda *
                                (self.eps_safe - qsafe_batch).detach()).mean()
            self.rcpo_lambda_optimizer.zero_grad()
            rcpo_lambda_loss.backward()
            self.rcpo_lambda_optimizer.step()
            self.rcpo_lambda = self.log_rcpo_lambda.exp()

            param = self.rcpo_lambda.item()
            param1 = qsafe_batch.mean().item()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item() + qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), param, param1

    def get_state_dict(self):
        save_dict = {'policy_state_dict': self.policy.state_dict(),
                     'critic_state_dict': self.critic.state_dict(),
                     'critic_target_state_dict': self.critic_target.state_dict(),
                     'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                     'policy_optimizer_state_dict': self.policy_optimizer.state_dict()}
        if self.safety_critic is not None:
            save_dict.update(self.safety_critic.get_state_dict())
        return save_dict

    def load_state_dict(self, checkpoint):
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(
            checkpoint['critic_target_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(
            checkpoint['policy_optimizer_state_dict'])

        if self.safety_critic is not None:
            self.safety_critic.load_state_dict(checkpoint)

    def train(self):
        self.critic.train()
        self.critic_target.train()
        self.policy.train()

    def eval(self):
        self.policy.eval()
        self.critic.eval()
        self.critic_target.eval()


class DSAC(BehaviorAgent):
    def __init__(self, num_inputs, action_space, recovery_agent, args):
        super(DSAC, self).__init__(num_inputs, action_space, args)

        print('Use DSAC Agent')

        self.tau_type = args.tau_type
        self.risk_type = args.risk_type
        self.risk_param = args.risk_param
        self.adaptive_risk = args.adaptive_risk
        self.zf_criterion = quantile_regression_loss

        self.num_quantiles = args.num_quantiles
        input_space = num_inputs + action_space.shape[0]
        M = args.hidden_size

        self.zf1 = QuantileMlp(
            input_space, [M, M]).to(device=self.device)
        self.zf1_optimizer = Adam(self.zf1.parameters(), lr=args.lr)
        self.zf2 = QuantileMlp(
            input_space, [M, M]).to(device=self.device)
        self.zf2_optimizer = Adam(self.zf2.parameters(), lr=args.lr)

        self.zf1_target = QuantileMlp(
            input_space, [M, M]).to(device=self.device)
        self.zf2_target = QuantileMlp(
            input_space, [M, M]).to(device=self.device)

        hard_update(self.zf1_target, self.zf1)
        hard_update(self.zf2_target, self.zf2)

        if args.tau_type == 'fqf':
            self.fp = FlattenMlp(
                input_size=input_space,
                output_size=self.num_quantiles,
                hidden_sizes=[M // 2, M // 2],
                output_activation=softmax,
            )
            self.fp_target = FlattenMlp(
                input_size=input_space,
                output_size=self.num_quantiles,
                hidden_sizes=[M // 2, M // 2],
                output_activation=softmax,
            )
            self.fp_optimizer = torch.optim.Adam(
                self.fp.parameters(), lr=self.lr / 6.)  # default lr=1e-5
            hard_update(self.fp_target, self.fp)
        else:
            self.fp = None
            self.fp_target = None

        if self.adaptive_risk == 'qrisk':
            self.recovery_agent = recovery_agent
        else:
            self.recovery_agent = None

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def select_actions(self, states, eval=False):
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            if eval is False:
                actions, _, _ = self.policy.sample(states)
            else:
                _, _, actions = self.policy.sample(states)
            return actions.detach().cpu().numpy()

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(
            reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        param_1 = 0
        param_2 = 0
        ''' Update Quantile Network '''
        with torch.no_grad():
            new_next_actions, new_log_pi, _ = self.policy.sample(
                next_state_batch)
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(
                next_state_batch, new_next_actions, fp=self.fp_target)
            target_z1_values = self.zf1_target(
                next_state_batch, new_next_actions, next_tau_hat)
            target_z2_values = self.zf2_target(
                next_state_batch, new_next_actions, next_tau_hat)
            target_z_values = torch.min(
                target_z1_values, target_z2_values) - self.alpha * new_log_pi
            z_target = reward_batch + \
                mask_batch * self.gamma * target_z_values

        tau, tau_hat, presum_tau = self.get_tau(
            state_batch, action_batch, fp=self.fp)
        z1_pred = self.zf1(state_batch, action_batch, tau_hat)
        z2_pred = self.zf2(state_batch, action_batch, tau_hat)
        zf1_loss = self.zf_criterion(
            z1_pred, z_target, tau_hat, next_presum_tau)
        zf2_loss = self.zf_criterion(
            z2_pred, z_target, tau_hat, next_presum_tau)

        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        self.zf1_optimizer.step()

        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        self.zf2_optimizer.step()

        """ Update FP """
        if self.tau_type == 'fqf':
            with torch.no_grad():
                dWdtau = 0.5 * (2 * self.zf1(state_batch, action_batch, tau[:, :-1]) - z1_pred[:, :-1] - z1_pred[:, 1:] +
                                2 * self.zf2(state_batch, action_batch, tau[:, :-1]) - z2_pred[:, :-1] - z2_pred[:, 1:])
                dWdtau /= dWdtau.shape[0]  # (N, T-1)

            self.fp_optimizer.zero_grad()
            tau[:, :-1].backward(gradient=dWdtau)
            self.fp_optimizer.step()

        ''' Update Policy '''
        new_actions, log_pi, _ = self.policy.sample(state_batch)

        if self.adaptive_risk == 'qrisk':

            with torch.no_grad():
                eps_safe = self.eps_safe
                sample_actions = new_actions
                
                qrisk_value = self.recovery_agent.critic(
                    state_batch, sample_actions).detach()
            mask = (qrisk_value > eps_safe).squeeze(1)
            risk_param = torch.zeros_like(qrisk_value)
            risk_param[mask] = (
                (1.0 - qrisk_value[mask]) / (1.0 - eps_safe)).clamp(0.3, 1.0)
            risk_param[~mask] = (
                qrisk_value[~mask] / eps_safe).clamp(0.3, 1.0)
            q_new_actions = torch.cat([
                self.get_raw_torch_value(
                    state_batch[mask], new_actions[mask], 'cvar', risk_param[mask]),
                self.get_raw_torch_value(state_batch[~mask], new_actions[~mask], 'ncvar', risk_param[~mask])], dim=0)
            param_1 = risk_param[mask].mean().item() - \
                risk_param[~mask].mean().item()
            param_2 = qrisk_value.mean().item()
        else:
            q_new_actions = self.get_raw_torch_value(
                state_batch, new_actions, self.risk_type, self.risk_param)
            param_1 = self.risk_param
            param_2 = 0

        policy_loss = (self.alpha * log_pi - q_new_actions).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        ''' Update Alpha '''
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi +
                           self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        ''' Soft Update '''
        if updates % self.target_update_interval == 0:
            soft_update(self.zf1_target, self.zf1, self.tau)
            soft_update(self.zf2_target, self.zf2, self.tau)
            if self.risk_type == 'fqf':
                soft_update(self.fp_target, self.fp, self.tau)

        return zf1_loss.item() + zf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), param_1, param_2

    def get_raw_torch_value(self, states, actions, risk_type, risk_param=0.3):
        with torch.no_grad():
            new_tau, new_tau_hat, new_presum_tau = self.get_tau(
                states, actions, fp=self.fp)

        z1_new_actions = self.zf1(states, actions, new_tau_hat)
        z2_new_actions = self.zf2(states, actions, new_tau_hat)

        if risk_type in ['neutral', 'std']:
            q1_new_actions = torch.sum(
                new_presum_tau * z1_new_actions, dim=1, keepdims=True)
            q2_new_actions = torch.sum(
                new_presum_tau * z2_new_actions, dim=1, keepdims=True)
            if risk_type == 'std':
                q1_std = new_presum_tau * \
                    (z1_new_actions - q1_new_actions).pow(2)
                q2_std = new_presum_tau * \
                    (z2_new_actions - q2_new_actions).pow(2)
                q1_new_actions -= risk_param * \
                    q1_std.sum(dim=1, keepdims=True).sqrt()
                q2_new_actions -= risk_param * \
                    q2_std.sum(dim=1, keepdims=True).sqrt()
        else:
            with torch.no_grad():
                if type(risk_param) is float:
                    risk_weights = distortion_de(
                        new_tau_hat, risk_type, risk_param)
                else:
                    risk_weights = muti_distortion_de(
                        new_tau_hat, risk_type, risk_param)
            q1_new_actions = torch.sum(
                risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
            q2_new_actions = torch.sum(
                risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)
        return q_new_actions

    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = zeros(
                len(actions), self.num_quantiles) + 1. / self.num_quantiles
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)
        # (N, T), note that they are tau1...tauN in the paper
        tau = torch.cumsum(presum_tau, dim=1)
        with torch.no_grad():
            tau_hat = zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau

    def get_state_dict(self):
        save_dict = {'zf1_state_dict': self.zf1.state_dict(),
                     'zf2_state_dict': self.zf2.state_dict(),
                     'zf1_target_state_dict': self.zf1_target.state_dict(),
                     'zf2_target_state_dict': self.zf2_target.state_dict(),
                     'zf1_optimizer_state_dict': self.zf1_optimizer.state_dict(),
                     'zf2_optimizer_state_dict': self.zf2_optimizer.state_dict(),
                     'policy_state_dict': self.policy.state_dict(),
                     'policty_optimizer_state_dict': self.policy_optimizer.state_dict(),
                     }
        if self.tau_type == 'fqf':
            save_dict['fp_state_dict'] = self.fp.state_dict()
            save_dict['fp_target_state_dict'] = self.fp_target.state_dict()
            save_dict['fp_optimizer_state_dict'] = self.fp_optimizer.state_dict()

        if self.recovery_agent is not None:
            save_dict.update(self.recovery_agent.get_state_dict())

        return save_dict

    def load_state_dict(self, checkpoint):
        self.zf1.load_state_dict(checkpoint['zf1_state_dict'])
        self.zf2.load_state_dict(checkpoint['zf2_state_dict'])
        self.zf1_target.load_state_dict(
            checkpoint['zf1_target_state_dict'])
        self.zf2_target.load_state_dict(
            checkpoint['zf2_target_state_dict'])
        self.zf1_optimizer.load_state_dict(
            checkpoint['zf1_optimizer_state_dict'])
        self.zf2_optimizer.load_state_dict(
            checkpoint['zf2_optimizer_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_optimizer.load_state_dict(
            checkpoint['policty_optimizer_state_dict'])

        if self.tau_type == 'fpf':
            self.fp.load_state_dict(checkpoint['fp_state_dict'])
            self.fp_target.load_state_dict(
                checkpoint['fp_target_state_dict'])
            self.fp_optimizer.load_state_dict(
                checkpoint['fp_optimizer_state_dict'])

        if self.recovery_agent is not None:
            self.recovery_agent.load_state_dict(checkpoint)

    def train(self):
        self.zf1.train()
        self.zf2.train()
        self.zf1_target.train()
        self.zf2_target.train()
        self.policy.train()
        if self.tau_type == 'fpf':
            self.fp.train()
            self.fp_target.train()

    def eval(self):
        self.zf1.eval()
        self.zf2.eval()
        self.zf1_target.eval()
        self.zf2_target.eval()
        self.policy.eval()
        if self.tau_type == 'fpf':
            self.fp.eval()
            self.fp_target.eval()

class IQL(BehaviorAgent):
    def __init__(self, num_inputs, action_space, args):
        super(IQL, self).__init__(num_inputs, action_space, args)
        
        print('Use IQL Agent')
        
        self.quantile = args.iql_quantile
        self.iql_beta = args.iql_beta
        self.clip_score = args.clip_score
        
        self.value = VNetwork(num_inputs, args.hidden_size).to(device=self.device)
        self.value_optimizer = Adam(self.value.parameters(), lr=args.lr)
        # Critic Setting
        
        self.critic = DoubleQNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = DoubleQNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
    
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def select_actions(self, states, eval=False):
        states = torch.FloatTensor(states).to(self.device)
        if eval is False:
            actions, _, _ = self.policy.sample(states)
        else:
            _, _, actions = self.policy.sample(states)
        return actions.detach().cpu().numpy()
        
    def update_parameters(self, memory, batch_size, updates):
        param = 0
        param1 = 0

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(
            reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        ''' Update QNetwork '''
        qf1, qf2 = self.critic(state_batch, action_batch)
        with torch.no_grad():
            next_q_value = reward_batch + mask_batch * self.gamma * self.value(next_state_batch)
        
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        ''' Update VNetwork '''
        vf = self.value(state_batch)
        target_qf = torch.min(*self.critic_target(state_batch, action_batch)).detach()
        vf_err = vf - target_qf
            
        vf_sign = (vf_err > 0).to(torch.float)
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * vf_err.pow(2)).mean()

        self.value_optimizer.zero_grad()
        vf_loss.backward()
        self.value_optimizer.step()
        
        ''' Update Policy '''
        log_pi = self.policy.logprob(state_batch, action_batch)
        with torch.no_grad():
            exp_adv = torch.exp(torch.clamp(self.iql_beta * -vf_err, max=self.clip_score))
            # if self.clip_score is not None:
            #     exp_adv = torch.clamp(exp_adv, max=self.clip_score)
        policy_loss = (-log_pi * exp_adv).mean()
                
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item() + qf2_loss.item(), policy_loss.item(), 0, 0, param, param1

    def get_state_dict(self):
        save_dict = {'value_state_dict': self.value.state_dict(),
                     'policy_state_dict': self.policy.state_dict(),
                     'critic_state_dict': self.critic.state_dict(),
                     'critic_target_state_dict': self.critic_target.state_dict(),
                     'value_optimizer_state_dict' : self.value_optimizer.state_dict(),
                     'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                     'policy_optimizer_state_dict': self.policy_optimizer.state_dict()}
        return save_dict

    def load_state_dict(self, checkpoint):
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(
            checkpoint['critic_target_state_dict'])
        self.value_optimizer.load_state_dict(
            checkpoint['value_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(
            checkpoint['policy_optimizer_state_dict'])

    def train(self):
        self.critic.train()
        self.critic_target.train()
        self.policy.train()
        self.value.train()

    def eval(self):
        self.policy.eval()
        self.critic.eval()
        self.critic_target.eval()
        self.value.train()