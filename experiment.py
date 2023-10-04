from behavior import SAC, DSAC, IQL
from recovery import Recovery
from replay_memory import ReplayMemory, UnitedReplayMemory
from safety_critic import DQNSafetyCritic, DoubleDQNSafetyCritic, IQNSafetyCritic, DoubleIQNSafetyCritic, IQLSafetyCritic
from torch.utils.tensorboard import SummaryWriter
from parallel_runner import ParallelRunner
from utils import set_gpu_mode, get_device, extend_and_repeat

import gym
import safety_gym
import os
import torch
import numpy as np
import time
import pickle
import random
import shutil


class Experimemt(object):
    def __init__(self, args):
        self.train_mode = False

        self.args = args

        set_gpu_mode(self.args.cuda)
        self.device = get_device()

        # Environment
        if self.args.single_core:  # single core
            self.env = gym.make(self.args.env_name)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        else:  # muti core
            self.envs = ParallelRunner(args)
            self.observation_space = self.envs.observation_space
            self.action_space = self.envs.action_space

        # Seed
        if self.args.seed != None:
            if self.args.single_core:
                self.env.seed(self.args.seed)
                self.env.action_space.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        # Safety Critic
        if self.args.use_safety_critic:
            if self.args.use_iql_recovery:
                self.safety_critic = IQLSafetyCritic(
                    self.observation_space.shape[0], self.action_space, self.args)
            else:
                safety_critic_dict = {
                    'DQN': {
                        'Single': DQNSafetyCritic,
                        'Double': DoubleDQNSafetyCritic,
                    },
                    'IQN': {
                        'Single': IQNSafetyCritic,
                        'Double': DoubleIQNSafetyCritic,
                    },
                }
                self.safety_critic = safety_critic_dict[self.args.safety_critic_network_type][self.args.safety_critic_network_num](
                    self.observation_space.shape[0], self.action_space, self.args)
        else:
            self.safety_critic = None

        # Recovery Agent
        if self.args.use_recovery:
            self.recovery = Recovery(self.observation_space.shape[0], self.action_space, self.safety_critic, self.args)
            # Load recovery model
            if self.args.load_pretrain_model_path != None and os.path.exists(self.args.load_pretrain_model_path):
                self.recovery.load_checkpoint(
                    ckpt_path=self.args.load_pretrain_model_path)
            elif self.args.not_pretrain_recovery:
                print(
                    'Invalid load pretrain model path, using initial recovery model...')
        else:
            self.recovery = None
            print('loading ...')
            if self.args.load_pretrain_model_path != None and os.path.exists(self.args.load_pretrain_model_path):
                self.safety_critic.load_checkpoint(ckpt_path=self.args.load_pretrain_model_path)
             

        # Replay Buffer
        if not self.args.not_train:
            self.memory = ReplayMemory(self.args.replay_size)
        if self.args.sample_data or self.args.save_expert_data or (not self.args.not_train and self.args.use_safety_critic):
            self.constraint_memory = UnitedReplayMemory(
                self.args.pretrain_data_size)
        else:
            self.constraint_memory = None

        # Behavior Agent
        if self.args.use_iql:
            self.agent = self.agent = IQL(
                self.observation_space.shape[0], self.action_space, self.args)
        elif self.args.use_dsac:
            self.agent = DSAC(
                self.observation_space.shape[0], self.action_space, self.recovery, self.args)
        else:
            self.agent = SAC(
                self.observation_space.shape[0], self.action_space, self.safety_critic, self.args)
        # load behavior model if exists
        if self.args.load_model_path != None and os.path.exists(self.args.load_model_path):
            self.agent.load_checkpoint(self.args.load_model_path)

        # SummaryWriter
        if self.args.experiment_name:
            self.experiment_name = self.args.experiment_name
        else:
            print('not experiment name')
            exit(0)
        if self.args.not_use_summary_writer:
            self.writer = None
        else:
            sw_path = os.path.join(
                self.args.logdir, self.args.env_name) + '/' + self.experiment_name
            self.writer = SummaryWriter(sw_path)

        # Sample Data or Load Data
        if self.constraint_memory is not None:
            if self.args.load_data_path and os.path.exists(self.args.load_data_path):
                self.constraint_memory.load_buffer(self.args.load_data_path)
                print(
                    f'total steps {len(self.constraint_memory)}, violations {self.constraint_memory.violations()}')
            elif args.sample_data:
                self.sampleOfflineData()
                self.constraint_memory.save_buffer(os.path.join(
                    self.args.save_data_path, self.args.data_name))

        # Make Dirs
        if not self.args.not_train:
            self.save_model_path = os.path.join(
                self.args.save_model_path, self.args.experiment_name)
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)

        if self.args.save_expert_data:
            self.save_expert_data_path = os.path.join(
                self.args.save_expert_data_path, self.args.experiment_name)
            if not os.path.exists(self.save_expert_data_path):
                os.makedirs(self.save_expert_data_path)

        self.total_numsteps = 0

    def sampleOfflineData(self):
        print(
            f'sampling offline data ..., total steps {self.args.pretrain_data_size}, sample last num steps {self.args.sample_last_num_steps}')
        sample_steps = 0

        # only use some steps before failed.
        if self.args.sample_last_num_steps >= 10:
            tmp_memorys = np.array(
                [ReplayMemory(self.args.sample_last_num_steps) for _ in range(self.args.batch_size_run)])

        states = self.envs.reset()
        i_sample = 1
        t_sample_start = time.time()
        while sample_steps < self.args.pretrain_data_size:
            if self.args.load_model_path:
                if self.args.load_pretrain_model_path and random.random() <= 0.5:
                    actions = self.recovery.select_actions(states, eval=False)
                else:
                    actions = self.agent.select_actions(states, eval=False)
                self.envs.send_actions(actions)
                new_states, _, dones, faileds, steps, infos = self.envs.recv_results()
            elif self.args.load_pretrain_model_path:
                actions = self.recovery.select_actions(
                    states, eval=False)
                self.envs.send_actions(actions)
                new_states, _, dones, faileds, steps, infos = self.envs.recv_results()
            else:
                new_states, actions, _, dones, faileds, steps, infos = self.envs.sample()
            masks = (~dones).astype(np.float)

            if self.args.sample_last_num_steps >= 10:
                tuple(map(lambda x, s, a, f, n, m: x.push(s, a, f, n, m),
                          tmp_memorys, states, actions, faileds, new_states, masks))

                def insert_memory(tmp_memory):
                    tuple(map(lambda x: self.constraint_memory.push(
                        *x), tmp_memory.buffer))

                if np.sum(dones) > 0:
                    if np.sum(faileds) > 0:
                        sample_steps += np.sum(
                            np.clip(steps[faileds], 0, self.args.sample_last_num_steps))
                        tuple(
                            map(insert_memory, tmp_memorys[faileds]))
                    tuple(map(lambda x: x.clear(), tmp_memorys[dones]))
            else:
                tuple(map(lambda s, a, f, n, m: self.constraint_memory.push(s, a, f, n, m),
                          states, actions, faileds, new_states, masks))
                sample_steps += self.args.batch_size_run

            states = new_states
            if sample_steps >= self.args.pretrain_data_size / 10 * i_sample:
                print(
                    f'{i_sample}0% finished, time consuming: {time.time() - t_sample_start}')
                i_sample += 1

        t_sample_end = time.time()
        print(f'time consuming: {t_sample_end - t_sample_start}')
        print(
            f'total steps: {len(self.constraint_memory)}, violations: {self.constraint_memory.violations()}')

    def runAll(self):
        # pretrain mode
        if not self.args.not_pretrain:
            self.pretrainRecoveryOffline()

        # train model
        if not self.args.not_train:
            if self.constraint_memory is not None:
                if self.args.not_finetune:
                    del self.constraint_memory
                    self.constraint_memory = None
                elif len(self.constraint_memory.mem) > self.args.pretrain_data_size:
                    self.constraint_memory.zip(self.args.pretrain_data_size)
            
            if self.args.use_iql:
                self.offlineTraining()
            else:
                self.trainModel()

        # test model
        if not self.args.not_test:
            self.testModel()

    def pretrainRecoveryOffline(self):
        print('pretrain offline ...')
        if self.args.add_noise_on_state_during_pretraining == True:
            self.constraint_memory.add_noise(0, self.args.noise_std_on_state_during_pretraining)

        # pretrai safety critic and recovery policy
        if self.args.use_recovery and not self.args.not_pretrain_recovery:
            self.recovery.train()
            for i_pretrain in range(1, self.args.pretrain_steps_recovery + 1):
                actor_loss = self.recovery.update_parameters(
                    self.constraint_memory,
                    self.args.batch_size
                )
                safety_critic_loss = self.safety_critic.update_parameters(
                    self.constraint_memory,
                    self.args.batch_size,
                    # self.recovery.policy if self.args.dea_recovery else self.agent.policy
                    self.sample
                )
                if self.writer and i_pretrain % 1000 == 0:
                    self.writer.add_scalar(
                        'pretrain_loss/safety_critic', safety_critic_loss, i_pretrain)
                    self.writer.add_scalar(
                        'pretrain_loss/recovery_policy', actor_loss, i_pretrain)
            if self.args.pretrain_model_name is not None:
                self.recovery.save_checkpoint(ckpt_path=os.path.join(
                    self.args.save_pretrain_model_path, self.args.pretrain_model_name))
        print('pretrain finished')

    def offlineTraining(self):
        print('training offline ...')
        agent_updates = 0
        self.agent.train()
        for i_epoch in range(1, self.args.num_epochs + 1):
            epoch_update_steps = 0
            while epoch_update_steps < int(self.args.num_steps / self.args.num_epochs):
                
                critic_loss, actor_loss, _, _, param, param1 = self.agent.update_parameters(
                    self.memory, self.args.batch_size, agent_updates)
                agent_updates += 1
                epoch_update_steps += 1

                if self.writer and agent_updates % 1000 == 0:
                    self.writer.add_scalar(
                        'loss/critic', critic_loss, agent_updates)
                    self.writer.add_scalar(
                        'loss/policy', actor_loss, agent_updates)
                    self.writer.add_scalar(
                        'value/param', param, agent_updates)
                    self.writer.add_scalar(
                        'value/param1', param1, agent_updates)
                if self.args.eval is True and agent_updates % 2000 == 0:
                    result = self.testEpisodes(episodes=20)
                    if self.writer:
                        self.writer.add_scalar(
                            'reward/test_while_training', result['avg_reward'], agent_updates)
                        self.writer.add_scalar(
                            'steps/test_while_training', result['avg_steps'], agent_updates)
                        self.writer.add_scalar(
                            'violation/test_while_training', result['avg_failed'], agent_updates)
                    self.agent.train()
                        
            self.agent.save_checkpoint(ckpt_path=os.path.join(self.save_model_path, 'agent_') + str(i_epoch))  
                
    def trainModel(self):
        print('training ...')
        # update steps
        self.total_numsteps = 0
        i_test = 1

        # result record
        total_reward = 0
        total_failed = 0
        i_episode = 0
        total_steps = 0
        agent_updates = 0
        last_agent_updates = 0
        episode_reward = np.zeros(self.args.batch_size_run)
        self.agent.train()

        if self.args.use_safety_critic:
            safety_critic_updates = 0

        if self.args.use_recovery:
            recovery_updates = 0
            recovery_ratio = 0
            recovery_rate = 0
            episode_recovery = np.zeros(self.args.batch_size_run)
            self.recovery.train()

        states = self.envs.reset()
        if self.args.add_noise_on_state_during_training == True:
            states = states + np.random.normal(0, self.args.noise_std_on_state_during_training, states.shape)
        
        for i_epoch in range(1, self.args.num_epochs + 1):
            epoch_update_steps = 0

            while epoch_update_steps < int(self.args.num_steps / (self.args.num_epochs * self.args.updates_per_step)):
                origin_actions, actions, recovery_used = self.getActions(states)
                
                if self.args.add_noise_on_action_during_training == True:
                    actions = np.clip(actions + np.random.normal(0, self.args.noise_std_on_action_during_training, actions.shape), -1.0, 1.0)

                self.envs.send_actions(actions)

                if not self.args.not_finetune:
                    # Update Safety Critic
                    if self.args.use_safety_critic and len(self.constraint_memory) > self.args.batch_size:
                        self.safety_critic.train()
                        for _ in range(self.args.updates_per_step_safety_critic):
                            safety_critic_loss = self.safety_critic.update_parameters(
                                self.constraint_memory,
                                self.args.batch_size,
                                # self.recovery.policy if self.args.dea_recovery else self.agent.policy
                                self.sample
                            )
                            safety_critic_updates += 1
                        if self.writer and (safety_critic_updates * self.args.updates_per_step_safety_critic) % 1000 == 0:
                            self.writer.add_scalar(
                                'loss/safety_critic', safety_critic_loss, safety_critic_updates)

                    # Update recovery agent
                    if self.args.use_recovery and len(self.constraint_memory) > self.args.batch_size:
                        self.recovery.train()
                        for _ in range(self.args.updates_per_step_recovery):
                            actor_loss = self.recovery.update_parameters(
                                self.constraint_memory,
                                self.args.batch_size)
                            recovery_updates += 1
                        if self.writer and (recovery_updates * self.args.updates_per_step_recovery) % 1000 == 0:
                            self.writer.add_scalar(
                                'loss/recovery_policy', actor_loss, recovery_updates)

                # Update behavior agent
                if len(self.memory) >= self.args.batch_size:
                    self.agent.train()
                    for _ in range(self.args.updates_per_step):
                        critic_loss, actor_loss, ent_loss, alpha, param, param1 = self.agent.update_parameters(
                            self.memory, self.args.batch_size, agent_updates)
                        agent_updates += 1
                        epoch_update_steps += 1

                    if self.writer and (agent_updates * self.args.updates_per_step) % 1000 == 0:
                        self.writer.add_scalar(
                            'loss/critic', critic_loss, agent_updates)
                        self.writer.add_scalar(
                            'loss/policy', actor_loss, agent_updates)
                        self.writer.add_scalar(
                            'value/param', param, agent_updates)
                        self.writer.add_scalar(
                            'value/param1', param1, agent_updates)
                        if self.args.automatic_entropy_tuning:
                            self.writer.add_scalar(
                                'loss/entropy_train', ent_loss, agent_updates)
                            self.writer.add_scalar(
                                'entropy_temprature/alpha', alpha, agent_updates)

                    if self.args.not_finetune:
                        if self.args.use_recovery:
                            recovery_updates = agent_updates
                        if self.args.use_safety_critic:
                            safety_critic_updates = agent_updates

                next_states, rewards, dones, faileds, steps, infos = self.envs.recv_results()
                masks = (~dones).astype(np.float)
                
                if self.args.add_noise_on_state_during_training == True:
                    next_states = next_states + np.random.normal(0, self.args.noise_std_on_state_during_training, next_states.shape)

                # reward penalty
                if self.args.use_rp:
                    rewards = rewards + faileds * self.args.rp_lambda

                tuple(map(self.memory.push,
                        states,
                        origin_actions,
                        rewards,
                        next_states,
                        masks))

                if self.constraint_memory is not None:
                    tuple(map(self.constraint_memory.push,
                              states,
                              actions,
                              faileds,
                              next_states,
                              masks))
                    
                states = next_states

                # total env num steps
                self.total_numsteps += self.args.batch_size_run

                # calculate episode rewards pre env
                episode_reward += rewards

                if self.args.use_recovery:
                    # calculate recovery use pre env
                    episode_recovery += recovery_used

                if np.sum(dones) > 0:
                    # total num episodes
                    i_episode += np.sum(dones)
                    # calculate total violations
                    total_failed += np.sum(faileds)
                    # calculate total episode steps
                    total_steps += np.sum(steps[dones])
                    # calculate total episode reward
                    total_reward += np.sum(episode_reward[dones])
                    episode_reward[dones] = 0

                    if self.args.use_recovery:
                        # calculate recovery ratio
                        recovery_ratio += np.sum(episode_recovery[dones])
                        # calculate recovery rate when failed
                        recovery_rate += np.sum(recovery_used[faileds])
                        episode_recovery[dones] = 0

                    # draw tensorboard
                    if self.writer and agent_updates >= last_agent_updates + 1000:
                        last_agent_updates = agent_updates
                        self.writer.add_scalar(
                            'reward/train', total_reward / i_episode, agent_updates)
                        self.writer.add_scalar(
                            'episodes/train', i_episode, agent_updates)
                        self.writer.add_scalar(
                            'episodes/failed', total_failed, agent_updates)
                        self.writer.add_scalar(
                            'violation/train', total_failed / i_episode, agent_updates)
                        self.writer.add_scalar(
                            'steps/train', total_steps / i_episode, agent_updates)

                        if self.args.use_recovery:
                            self.writer.add_scalar(
                                'recovery_ratio/train', recovery_ratio / total_steps, recovery_updates)
                            self.writer.add_scalar(
                                'recovery_rate/train', recovery_rate / total_failed if total_failed else 1, recovery_updates)

                    # test while training
                    if self.args.eval is True and agent_updates // 2000 >= i_test and self.args.start_steps <= self.total_numsteps:
                        i_test += 1
                        result = self.testEpisodes(episodes=20)
                        if self.writer:
                            self.writer.add_scalar(
                                'reward/test_while_training', result['avg_reward'], agent_updates)
                            self.writer.add_scalar(
                                'steps/test_while_training', result['avg_steps'], agent_updates)
                            self.writer.add_scalar(
                                'violation/test_while_training', result['avg_failed'], agent_updates)

                            if self.args.use_recovery:
                                self.writer.add_scalar(
                                    'recovery_ratio/test_while_training', result['recovery_ratio'], recovery_updates)
                                self.writer.add_scalar(
                                    'recovery_rate/test_while_training', result['recovery_rate'], recovery_updates)

                # while finished
            # Save model every 100k training steps
            self.agent.save_checkpoint(
                ckpt_path=os.path.join(self.save_model_path, 'agent_') + str(i_epoch))
            if self.args.use_recovery:
                self.recovery.save_checkpoint(
                    ckpt_path=os.path.join(self.save_model_path, 'recovery_') + str(i_epoch))

            # Save replay buffer
            if self.args.save_expert_data == True:
                if i_epoch > 1:
                    os.remove(os.path.join(self.save_expert_data_path,
                              'replay_memory_') + str(i_epoch-1))
                    os.remove(os.path.join(self.save_expert_data_path,
                              'constraint_memory_') + str(i_epoch-1))
                self.memory.save_buffer(os.path.join(
                    self.save_expert_data_path, 'replay_memory_') + str(i_epoch))
                self.constraint_memory.save_buffer(os.path.join(
                    self.save_expert_data_path, 'constraint_memory_') + str(i_epoch))

            # for i_epoch finished
        # tranModel() return

    def testModel(self):
        print('testing ...')
        if self.args.single_core:
            result = self.testEpisodesSingleCore(
                episodes=1)
        else:
            result = self.testEpisodes(episodes=self.args.final_test_times)
        print('--------------------')
        print(f'{self.args.final_test_times} episodes test:')
        print(
            f'avg reward: {result["avg_reward"]}, avg steps {result["avg_steps"]}, violation rate: {result["avg_failed"]}')
        if self.args.use_recovery:
            print(
                f'recovery ratio: {result["recovery_ratio"]}, recovery rate: {result["recovery_rate"]}')
        print('--------------------')

    def testEpisodes(self, behavior_agent=None, recovery_agent=None, episodes=1):
        if behavior_agent == None:
            behavior_agent = self.agent
        if recovery_agent == None:
            recovery_agent = self.recovery

        if behavior_agent:
            behavior_agent.eval()
        if recovery_agent:
            recovery_agent.eval()

        total_reward = 0
        total_failed = 0
        total_steps = 0
        total_des_lv = 0
        total_des_av = 0
        i_episode = 0
        episode_reward = np.zeros(self.args.batch_size_test)

        if self.args.use_recovery:
            recovery_ratio = 0
            recovery_rate = 0
            episode_recovery = np.zeros(self.args.batch_size_test)

        states = self.envs.reset(test=True)
        
        while i_episode < episodes:
            if self.args.add_noise_on_state_during_testing == True:
                states = states + np.random.normal(0, self.args.noise_std_on_state_during_testing, states.shape)
                
            if self.args.test_recovery:
                actions = recovery_agent.select_actions(states, eval=True)
                recovery_used = np.ones(self.args.batch_size_test)
            else:
                _, actions, recovery_used = self.getActions(
                    states, behavior_agent=behavior_agent, recovery_agent=recovery_agent, eval=True)

            if self.args.add_noise_on_action_during_testing == True:
                actions = np.clip(actions + np.random.normal(0, self.args.noise_std_on_action_during_testing, actions.shape), -1.0, 1.0)
            
            self.envs.send_actions(actions, test=True)
            next_states, rewards, dones, faileds, steps, infos = self.envs.recv_results(
                test=True)

            states = next_states

            episode_reward += rewards
            total_des_lv += np.sum(np.abs(actions[:, 0]))

            total_des_av += np.sum(np.abs(actions[:, 1]))

            if self.args.use_recovery:
                episode_recovery += recovery_used

            if np.sum(dones) > 0:
                if self.args.use_recovery:
                    recovery_rate += np.sum(recovery_used[faileds])
                    recovery_ratio += np.sum(episode_recovery[dones])
                    episode_recovery[dones] = 0

                i_episode += np.sum(dones)
                total_reward += np.sum(episode_reward[dones])
                total_steps += np.sum(steps[dones])
                total_failed += np.sum(faileds)
                episode_reward[dones] = 0

        result = {
            'avg_reward': total_reward / i_episode,
            'avg_steps': total_steps / i_episode,
            'avg_failed': total_failed / i_episode,
            'avg_des_lv': total_des_lv / total_steps,
            'avg_des_av': total_des_av / total_steps,
        }
        if self.args.use_recovery:
            result['recovery_ratio'] = recovery_ratio / total_steps if total_steps else 0
            result['recovery_rate'] = recovery_rate / total_failed if total_failed else 1
        return result

    def testEpisodesSingleCore(self, episodes=1):
        if episodes == 0:
            return 0
        total_reward = 0
        violations = 0
        recovery_ratio = 0
        recovery_rate = 0
        total_steps = 0

        for _ in range(episodes):
            state = self.env.reset()

            done = False
            episode_steps = 0
            episode_recovery_used = 0
            total_sample = 0
            while not done:
                self.env.render()
                episode_steps += 1
                total_steps += 1
                if self.args.test_recovery:
                    action = self.recovery.select_action(state)
                    recovery_used = True
                else:
                    _, action, recovery_used = self.getAction(
                        state, eval=True)
                total_sample += episode_recovery_used > 10
                if recovery_used:
                    episode_recovery_used += 2
                else:
                    episode_recovery_used -= 1

                if self.args.safety_gym:
                    next_state, reward, done, info = self.env.step(
                        action)
                    truncated = info['cost'] > 0
                else:
                    next_state, reward, done, info = self.env.step(
                        action)
                    truncated = not 'TimeLimit.truncated' in info.keys()

                total_reward += reward
                state = next_state
                recovery_ratio += recovery_used

                done = done or truncated

                violations += truncated
                recovery_rate += truncated and recovery_used

        result = {
            'avg_reward': total_reward / episodes,
            'avg_steps': total_steps / episodes,
            'avg_failed': violations / episodes,
        }
        if self.args.use_recovery:
            result['recovery_ratio'] = recovery_ratio / \
                total_steps if total_steps else 0
            result['recovery_rate'] = recovery_rate / \
                violations if violations > 0 else 1
        return result

    def getAction(self, state, behavior_agent=None, recovery_agent=None, eval=False, sample=False):
        if behavior_agent == None:
            behavior_agent = self.agent
        if recovery_agent == None:
            recovery_agent = self.recovery

        if sample or (not eval and self.args.start_steps > self.total_numsteps):
            action = self.env.action_space.sample()  # Sample random action
        else:
            action = behavior_agent.select_action(
                state, eval)  # Sample action from policy

        if self.args.use_recovery:
            critic_val = recovery_agent.get_value(
                torch.FloatTensor(state).to(self.device).unsqueeze(0),
                torch.FloatTensor(action).to(self.device).unsqueeze(0))
            print(critic_val)
            if critic_val >= self.args.eps_safe:
                recovery = True
                recovery_action = recovery_agent.select_action(state, eval)
            else:
                recovery = False
        else:
            recovery = False

        if recovery:
            real_action = recovery_action
        else:
            real_action = np.copy(action)

        return action, real_action, recovery

    def getActions(self, states, behavior_agent=None, recovery_agent=None, eval=False):
        if behavior_agent == None:
            behavior_agent = self.agent
        if recovery_agent == None:
            recovery_agent = self.recovery

        if not eval and self.args.start_steps > self.total_numsteps:
            actions = self.envs.sample_actions()  # Sample random action
        else:
            actions = behavior_agent.select_actions(states, eval)

        if self.args.use_recovery:
            critic_val = recovery_agent.get_value(
                torch.FloatTensor(states).to(self.device),
                torch.FloatTensor(actions).to(self.device)).ravel()
            recovery = critic_val > self.args.eps_safe
        else:
            if eval:
                recovery = np.full(self.args.batch_size_test, False)
            else:
                recovery = np.full(self.args.batch_size_run, False)

        real_actions = actions.copy()
        if np.sum(recovery) > 0:
            # when activate recovery policy, it shouldn't sample random action
            real_actions[recovery] = recovery_agent.select_actions(
                states[recovery], eval=eval or self.args.recovery_not_explore)

        return actions, real_actions, recovery
    
    def sample(self, state_batch):
        with torch.no_grad():
            if self.args.dea_recovery:
                actions, _, _ = self.recovery.policy.sample(state_batch)
            elif self.args.com_recovery:
                actions, _, _ = self.agent.policy.sample(state_batch)
                critic_val = self.safety_critic.get_value(state_batch, actions)
                actions[(critic_val > self.args.eps_safe).squeeze()], _, _ = self.recovery.policy.sample(state_batch[(critic_val > self.args.eps_safe).squeeze()])
                
            else:
                actions, _, _ = self.agent.policy.sample(state_batch)
            
            return actions
