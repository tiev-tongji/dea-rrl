from model import VRiskNetwork, QRiskNetwork, DoubleQRiskNetwork, QuantileQriskMlp, FlattenMlp, softmax
from utils import hard_update, soft_update, get_device, zeros, zeros_like, rand, quantile_regression_loss
from risk import distortion_de

import os
import torch
import torch.nn.functional as F


class SafetyCritic:
    def __init__(self, args):
        self.device = get_device()

        # RL Settings
        self.gamma_safe = args.gamma_safe
        self.pos_fraction = args.pos_fraction

        # Target Network Updates
        self.updates = 0
        self.tau = args.tau
        self.target_update_interval = args.target_update_interval

    def update_parameters(self, *args):
        raise NotImplementedError

    def get_value(self, states, actions):
        raise NotImplementedError

    def get_state_dict(self, *args):
        raise NotImplementedError

    # Save model parameters
    def save_checkpoint(self, ckpt_path=None):
        print('Saving recovery models to {}'.format(ckpt_path))
        torch.save(self.get_state_dict, ckpt_path)

    def load_state_dict(self, *args):
        raise NotADirectoryError

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

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class IQLSafetyCritic(SafetyCritic):
    def __init__(self, num_inputs, action_space, args):
        super(IQLSafetyCritic, self).__init__(args)
        print('Use IQL Safety Critic')

        self.quantile = args.iql_quantile_recovery

        self.value = VRiskNetwork(num_inputs, args.hidden_size).to(self.device)
        self.critic = DoubleQRiskNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = DoubleQRiskNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

    def update_parameters(self, memory, batch_size, policy=None):
        state_batch, action_batch, constraint_batch, next_state_batch, _ = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)

        ''' Update QRisk '''
        qf1, qf2 = self.critic(state_batch, action_batch)
        with torch.no_grad():
            q_target = constraint_batch + self.gamma_safe * (1. - constraint_batch) * self.value(next_state_batch)

        q1_loss = F.mse_loss(qf1, q_target)
        q2_loss = F.mse_loss(qf2, q_target)
        qf_loss = (q1_loss + q2_loss).mean()

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        ''' Update VRisk '''
        vf = self.value(state_batch)
        vf_err = vf - \
            torch.max(*self.critic_target(state_batch, action_batch)).detach()
        vf_sign = (vf_err > 0).to(torch.float)
        vf_weight = (1 - vf_sign) * self.quantile + \
            vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * vf_err.pow(2)).mean()

        self.value_optimizer.zero_grad()
        vf_loss.backward()
        self.value_optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        return qf_loss.item()

    def __call__(self, states, actions):
        return torch.max(*self.critic(states, actions))

    def get_value(self, states, actions):
        return torch.max(*self.critic(states, actions)).detach().cpu().numpy()

    def get_state_dict(self):
        save_dict = {'safety_critic_state_dict': self.critic.state_dict(),
                     'safety_critic_target_state_dict': self.critic_target.state_dict(),
                     'safety_critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                     'safety_value_state_dict': self.value.state_dict(),
                     'safety_value_optimizer_state_dict': self.value_optimizer.state_dict(),
                     }
        return save_dict

    def load_state_dict(self, checkpoint):
        self.critic.load_state_dict(checkpoint['safety_critic_state_dict'])
        self.critic_target.load_state_dict(
            checkpoint['safety_critic_target_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['safety_critic_optimizer_state_dict'])
        self.value.load_state_dict(checkpoint['safety_value_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['safety_value_optimizer_state_dict'])

    def train(self):
        self.critic.train()
        self.critic_target.train()
        self.value.train()

    def eval(self):
        self.critic.eval()
        self.critic_target.eval()
        self.value.eval()


class DQNSafetyCritic(SafetyCritic):
    def __init__(self, num_inputs, action_space, args):
        super(DQNSafetyCritic, self).__init__(args)
        print('Use DQN Safety Critic')

        self.critic = QRiskNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = QRiskNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

    def update_parameters(self, memory, batch_size, policy_sample=None):
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, _, _ = policy_sample(
                next_state_batch)

            qf_next_target = self.critic_target(
                next_state_batch, next_state_action)
            next_q_value = self.gamma_safe * (constraint_batch + mask_batch *
                                              qf_next_target)

        qf = self.critic(state_batch, action_batch)
        qf_loss = -(torch.log(qf) * next_q_value +
                    torch.log(1-qf) * (1-next_q_value))
        qf_loss = qf_loss.mean()

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        return qf_loss.item()

    def __call__(self, states, actions):
        return self.critic(states, actions)

    def get_value(self, states, actions):
        self.eval()
        with torch.no_grad():
            return self.critic_target(states, actions).detach().cpu().numpy()

    def get_state_dict(self):
        save_dict = {'safety_critic_state_dict': self.critic.state_dict(),
                     'safety_critic_target_state_dict': self.critic_target.state_dict(),
                     'safety_critic_optimizer_state_dict': self.critic_optimizer.state_dict()
                     }

        return save_dict

    def load_state_dict(self, checkpoint):
        self.critic.load_state_dict(checkpoint['safety_critic_state_dict'])
        self.critic_target.load_state_dict(
            checkpoint['safety_critic_target_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])

    def train(self):
        self.critic.train()
        self.critic_target.train()

    def eval(self):
        self.critic.eval()
        self.critic_target.eval()


class DoubleDQNSafetyCritic(SafetyCritic):
    def __init__(self, num_inputs, action_space, args):
        super(DoubleDQNSafetyCritic, self).__init__(args)

        print('Use Double DQN Safety Critic')

        self.critic = DoubleQRiskNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target = DoubleQRiskNetwork(
            num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

    def update_parameters(self, memory, batch_size, policy_sample=None):
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action = policy_sample(
                next_state_batch)

            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            target_values = torch.max(qf1_next_target, qf2_next_target)
            next_q_value = self.gamma_safe * (constraint_batch + mask_batch *
                                              target_values)

        qf1, qf2 = self.critic(
            state_batch, action_batch
        )
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        qf_loss = (qf1_loss + qf2_loss).mean()

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        return qf_loss.item()

    def __call__(self, states, actions):
        qf1_pi, qf2_pi = self.critic(states, actions)
        return torch.max(qf1_pi, qf2_pi)

    def get_value(self, states, actions):
        self.eval()
        with torch.no_grad():
            q1, q2 = self.critic_target(states, actions)
            return torch.max(q1, q2).detach().cpu().numpy()

    def get_state_dict(self):
        save_dict = {'safety_critic_state_dict': self.critic.state_dict(),
                     'safety_critic_target_state_dict': self.critic_target.state_dict(),
                     'safety_critic_optimizer_state_dict': self.critic_optimizer.state_dict()
                     }

        return save_dict

    def load_state_dict(self, checkpoint):
        self.critic.load_state_dict(checkpoint['safety_critic_state_dict'])
        self.critic_target.load_state_dict(
            checkpoint['safety_critic_target_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['safety_critic_optimizer_state_dict'])

    def train(self):
        self.critic.train()
        self.critic_target.train()

    def eval(self):
        self.critic.eval()
        self.critic_target.eval()


class DistributedSafetyCritic(SafetyCritic):
    def __init__(self, num_inputs, action_space, args):
        super(DistributedSafetyCritic, self).__init__(args)

        self.tau_type = args.tau_type_recovery
        self.risk_type = args.risk_type_recovery
        self.num_quantiles = args.num_quantiles_recovery
        self.risk_param = args.risk_param_recovery

        self.zf_criterion = quantile_regression_loss

        if self.tau_type == 'fqf':
            M = args.hidden_size
            self.fp = FlattenMlp(
                input_size=num_inputs + action_space.shape[0],
                output_size=self.num_quantiles,
                hidden_sizes=[M // 2, M // 2],
                output_activation=softmax,
            )
            self.fp_target = FlattenMlp(
                input_size=num_inputs + action_space.shape[0],
                output_size=self.num_quantiles,
                hidden_sizes=[M // 2, M // 2],
                output_activation=softmax,
            )
            self.fp_optimizer = torch.optim.Adam(
                self.fp.parameters(), lr=self.lr / 6)
            hard_update(self.fp_target, self.fp)
        else:
            self.fp = None
            self.fp_target = None
            self.fp_optimizer = None

    def __call__(self, states, actions, risk_type=None, risk_param=None):
        return self.get_raw_torch_value(states, actions, risk_type, risk_param)

    def get_value(self, states, actions, risk_type=None, risk_param=None):
        self.eval()
        with torch.no_grad():
            return self.get_raw_torch_value(states, actions, risk_type, risk_param).cpu().numpy().clip(0., 1.)

    def get_raw_torch_value(self, *args):
        raise NotImplementedError

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


class IQNSafetyCritic(DistributedSafetyCritic):
    def __init__(self, num_inputs, action_space, args):
        super(IQNSafetyCritic, self).__init__(
            num_inputs, action_space, args)

        print('Use IQN Safety Critic')

        input_space = num_inputs + action_space.shape[0]
        M = args.hidden_size

        self.critic = QuantileQriskMlp(
            input_space, [M, M]).to(device=self.device)
        self.critic_target = QuantileQriskMlp(
            input_space, [M, M]).to(device=self.device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)

    def update_parameters(self, memory, batch_size, policy_sample = None):
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)

        with torch.no_grad():
            new_next_actions, new_log_pi, _ = policy_sample(
                next_state_batch)
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(
                next_state_batch, new_next_actions, fp=self.fp_target)

            target_values = self.critic_target(
                next_state_batch, new_next_actions, next_tau_hat)

            z_target = constraint_batch + mask_batch * self.gamma_safe * target_values

        tau, tau_hat, presum_tau = self.get_tau(
            state_batch, action_batch, fp=self.fp)
        z_pred = self.critic(state_batch, action_batch, tau_hat)
        zf_loss = self.zf_criterion(
            z_pred, z_target, tau_hat, next_presum_tau)

        self.critic_optimizer.zero_grad()
        zf_loss.backward()
        self.critic_optimizer.step()

        """
        Update FP
        """
        if self.tau_type == 'fqf':
            with torch.no_grad():
                dWdtau = 2 * self.critic(state_batch, action_batch,
                                         tau[:, :-1]) - z_pred[:, :-1] - z_pred[:, 1:]
                dWdtau /= dWdtau.shape[0]  # (N, T-1)

            self.fp_optimizer.zero_grad()
            tau[:, :-1].backward(gradient=dWdtau)
            self.fp_optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            if self.tau_type == 'fpf':
                soft_update(self.fp_target, self.fp)

        return zf_loss.item()

    def get_raw_torch_value(self, states, actions, risk_type=None, risk_param=None):
        if risk_type == None:
            risk_type = self.risk_type
        if risk_param == None:
            risk_param = self.risk_param

        with torch.no_grad():
            new_tau, new_tau_hat, new_presum_tau = self.get_tau(
                states, actions, fp=self.fp)

        z_new_actions = self.critic(states, actions, new_tau_hat)
        if risk_type in ['neutral', 'std']:
            q_new_actions = torch.sum(
                new_presum_tau * z_new_actions, dim=-1, keepdims=True)
            if risk_type == 'std':
                q_std = new_presum_tau * \
                    (z_new_actions - q_new_actions).pow(2)
                q_new_actions -= risk_param * \
                    q_std.sum(dim=-1, keepdims=True).sqrt()
        else:
            with torch.no_grad():
                risk_weights = distortion_de(
                    new_tau_hat, risk_type, risk_param)
            q_new_actions = torch.sum(
                risk_weights * new_presum_tau * z_new_actions, dim=-1, keepdims=True)
        return q_new_actions

    def get_state_dict(self):
        save_dict = {'safety_critic_state_dict': self.critic.state_dict(),
                     'safety_critic_target_state_dict': self.critic_target.state_dict(),
                     'safety_critic_optimizer_state_dict': self.critic_optimizer.state_dict()}

        if self.tau_type == 'fqf':
            save_dict['safety_fp_state_dict'] = self.fp.state_dict()
            save_dict['safety_fp_target_state_dict'] = self.fp_target.state_dict()
            save_dict['safety_fp_optimizer_state_dict'] = self.fp_optimizer.state_dict()

        return save_dict

    def load_state_dict(self, checkpoint):
        self.critic.load_state_dict(
            checkpoint['safety_critic_state_dict'])
        self.critic_target.load_state_dict(
            checkpoint['safety_critic_target_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['safety_critic_optimizer_state_dict'])

        if self.tau_type == 'fpf':
            self.fp.load_state_dict(checkpoint['safety_fp_state_dict'])
            self.fp_target.load_state_dict(
                checkpoint['safety_fp_target_state_dict'])
            self.fp_optimizer.load_state_dict(
                checkpoint['safety_fp_optimizer_state_dict'])

    def train(self):
        self.critic.train()
        self.critic_target.train()

        if self.tau_type == 'fpf':
            self.fp.train()
            self.fp_target.train()

    def eval(self):
        self.critic.eval()
        self.critic_target.eval()

        if self.tau_type == 'fpf':
            self.fp.eval()
            self.fp_target.eval()


class DoubleIQNSafetyCritic(DistributedSafetyCritic):
    def __init__(self, num_inputs, action_space, args):
        super(DoubleIQNSafetyCritic, self).__init__(
            num_inputs, action_space, args)

        print('Use Double IQN Safety Critic')

        input_space = num_inputs + action_space.shape[0]
        M = args.hidden_size

        self.zf1 = QuantileQriskMlp(
            input_space, [M, M]).to(device=self.device)
        self.zf2 = QuantileQriskMlp(
            input_space, [M, M]).to(device=self.device)
        self.zf1_target = QuantileQriskMlp(
            input_space, [M, M]).to(device=self.device)
        self.zf2_target = QuantileQriskMlp(
            input_space, [M, M]).to(device=self.device)

        self.zf1_optimizer = torch.optim.Adam(
            self.zf1.parameters(), lr=args.lr)
        self.zf2_optimizer = torch.optim.Adam(
            self.zf2.parameters(), lr=args.lr)
        hard_update(self.zf1_target, self.zf1)
        hard_update(self.zf2_target, self.zf2)

    def update_parameters(self, memory, batch_size, policy_sample=None):
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch).to(
            self.device).unsqueeze(1)

        with torch.no_grad():
            new_next_actions, new_log_pi, _ = policy_sample(
                next_state_batch)
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(
                next_state_batch, new_next_actions, fp=self.fp_target)
            target_z1_values = self.zf1_target(
                next_state_batch, new_next_actions, next_tau_hat)
            target_z2_values = self.zf2_target(
                next_state_batch, new_next_actions, next_tau_hat)
            target_z_values = torch.max(
                target_z1_values, target_z2_values)

            z_target = constraint_batch + mask_batch * self.gamma_safe * target_z_values

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

        """
        Update FP
        """
        if self.tau_type == 'fqf':
            with torch.no_grad():
                dWdtau = 0.5 * (2 * self.zf1(state_batch, action_batch, tau[:, :-1]) - z1_pred[:, :-1] - z1_pred[:, 1:] +
                                2 * self.zf2(state_batch, action_batch, tau[:, :-1]) - z2_pred[:, :-1] - z2_pred[:, 1:])
                dWdtau /= dWdtau.shape[0]  # (N, T-1)

            self.fp_optimizer.zero_grad()
            tau[:, :-1].backward(gradient=dWdtau)
            self.fp_optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            soft_update(self.zf1_target, self.zf1, self.tau)
            soft_update(self.zf2_target, self.zf2, self.tau)

        return zf1_loss.item() + zf2_loss.item()

    def get_raw_torch_value(self, states, actions, risk_type, risk_param):
        if risk_type == None:
            risk_type = self.risk_type
        if risk_param == None:
            risk_param = self.risk_param

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
                risk_weights = distortion_de(
                    new_tau_hat, risk_type, risk_param)
            q1_new_actions = torch.sum(
                risk_weights * new_presum_tau * z1_new_actions, dim=1, keepdims=True)
            q2_new_actions = torch.sum(
                risk_weights * new_presum_tau * z2_new_actions, dim=1, keepdims=True)
        q_new_actions = torch.max(q1_new_actions, q2_new_actions)
        return q_new_actions

    # Save model parameters
    def get_state_dict(self):
        save_dict = {'safety_zf1_state_dict': self.zf1.state_dict(),
                     'safety_zf1_target_state_dict': self.zf1_target.state_dict(),
                     'safety_zf2_state_dict': self.zf2.state_dict(),
                     'safety_zf2_target_state_dict': self.zf2_target.state_dict(),
                     'safety_zf1_optimizer_state_dict': self.zf1_optimizer.state_dict(),
                     'safety_zf2_optimizer_state_dict': self.zf2_optimizer.state_dict(), }

        if self.tau_type == 'fqf':
            save_dict['safety_fp_state_dict'] = self.fp.state_dict()
            save_dict['safety_fp_target_state_dict'] = self.fp_target.state_dict()
            save_dict['safety_fp_optimizer_state_dict'] = self.fp_optimizer.state_dict()

        return save_dict

    # Load model parameters
    def load_state_dict(self, checkpoint):
        self.zf1.load_state_dict(checkpoint['zf1_state_dict'])
        self.zf1_target.load_state_dict(
            checkpoint['zf1_target_state_dict'])
        self.zf2.load_state_dict(checkpoint['zf2_state_dict'])
        self.zf2_target.load_state_dict(
            checkpoint['zf2_target_state_dict'])
        self.zf1_optimizer.load_state_dict(
            checkpoint['zf1_optimizer_state_dict'])
        self.zf2_optimizer.load_state_dict(
            checkpoint['zf2_optimizer_state_dict'])

        if self.tau_type == 'fpf':
            self.fp.load_state_dict(checkpoint['safety_fp_state_dict'])
            self.fp_target.load_state_dict(
                checkpoint['safety_fp_target_state_dict'])
            self.fp_optimizer.load_state_dict(
                checkpoint['safety_fp_optimizer_state_dict'])

    def train(self):
        self.zf1.train()
        self.zf1_target.train()
        self.zf2.train()
        self.zf2_target.train()

        if self.tau_type == 'fpf':
            self.fp.train()
            self.fp_target.train()

    def eval(self):
        self.zf1.eval()
        self.zf2.eval()
        self.zf1_target.eval()
        self.zf2_target.eval()

        if self.tau_type == 'fpf':
            self.fp.eval()
            self.fp_target.eval()
