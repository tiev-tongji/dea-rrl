import multiprocessing as mp
from multiprocessing import Pipe, Process
import numpy as np
import os
import gym
import safety_gym

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py

from gym.envs.registration import register

config1 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'goal_locations': [(1.1, 1.1)],
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 1,
    'hazards_size': 0.7,
    'hazards_keepout': 0.705,
    'hazards_locations': [(0, 0)]
}
register(id='StaticEnv-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config1})

config2 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'observe_goal_lidar': True,
    'observe_hazards': True,
    'constrain_hazards': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'hazards_num': 3,
    'hazards_size': 0.3,
    'hazards_keepout': 0.305
}
register(id='DynamicEnv-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config2})


class ParallelRunner:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size_run
        self.batch_size_test = args.batch_size_test

        # Make subprocesses for the envs
        mp.set_start_method("spawn", force=True)
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)])
        self.parent_conns_test, self.worker_conns_test = zip(
            *[Pipe() for _ in range(self.batch_size_test)]
        )
        self.parent_conns = np.array(self.parent_conns)
        self.parent_conns_test = np.array(self.parent_conns_test)

        self.ps = []
        self.cpuid = args.start_cpuid
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker,
                         args=(worker_conn, args.env_name, args.safety_gym, args.seed + i, False, self.cpuid + i))

            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.ps_test = []
        for i, worker_conn in enumerate(self.worker_conns_test):
            ps = Process(target=env_worker,
                         args=(worker_conn, args.env_name, args.safety_gym, args.seed + i + self.batch_size, True, self.cpuid + self.batch_size + i))

            self.ps_test.append(ps)

        for p in self.ps_test:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("info", None))
        self.observation_space, self.action_space = np.stack(
            self.parent_conns[0].recv())

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

        for parent_conn in self.parent_conns_test:
            parent_conn.send(("close", None))

    def reset(self, mask=None, test=False):
        if test:
            if mask is None:
                mask = np.full(self.batch_size_test, True)
            tuple(map(lambda x: x.send(('reset', None)),
                      self.parent_conns_test[mask]))

            states = list(
                map(lambda x: x.recv(), self.parent_conns_test[mask]))
        else:
            if mask is None:
                mask = np.full(self.batch_size, True)
            tuple(map(lambda x: x.send(('reset', None)),
                      self.parent_conns[mask]))
            states = list(map(lambda x: x.recv(), self.parent_conns[mask]))
        return np.array(states)

    def sample(self):
        tuple(map(lambda x: x.send(
            ('sample', None)), self.parent_conns))
        states, actions, rewards, dones, faileds, steps, infos = zip(
            *map(lambda x: x.recv(), self.parent_conns))
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(faileds), np.array(steps), np.array(infos)

    def sample_actions(self, test=False):
        if test:
            tuple(map(lambda x: x.send(('sample_actions', None)),
                      self.parent_conns_test))
            actions = list(map(lambda x: x.recv(), self.parent_conns_test))
        else:
            tuple(map(lambda x: x.send(('sample_actions', None)),
                      self.parent_conns))
            actions = list(map(lambda x: x.recv(), self.parent_conns))
        return np.array(actions)

    def send_actions(self, actions, test=False):
        if test:
            tuple(map(lambda x, action: x.send(
                ('step', action)), self.parent_conns_test, actions))
        else:
            tuple(map(lambda x, action: x.send(
                ('step', action)), self.parent_conns, actions))

    def recv_results(self, test=False):
        if test:
            states, rewards, dones, faileds, steps, infos = zip(
                *map(lambda x: x.recv(), self.parent_conns_test))
        else:
            states, rewards, dones, faileds, steps, infos = zip(
                *map(lambda x: x.recv(), self.parent_conns))
        return np.array(states), np.array(rewards), np.array(dones), np.array(faileds), np.array(steps), np.array(infos)


def env_worker(remote, env_name, safety_gym, seed, test_env, cpuid):
    # Make environment
    pid = os.getpid()
    ppid = os.getppid()
    affinity_mask = {cpuid}
    os.sched_setaffinity(pid, affinity_mask)
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    # p = psutil.Process()
    episode_steps = 0

    def env_reset():
        state = env.reset()
        return state

    def env_step(action):
        if safety_gym:
            next_state, reward, done, info = env.step(action)
            truncated = info['cost'] > 0
        else:
            next_state, reward, done, info = env.step(action)
            truncated = done and not 'TimeLimit.truncated' in info.keys()

        done = done or truncated

        if test_env == False and safety_gym:
            reward *= 100
        return next_state, reward, done, truncated, info

    while True:
        if not remote.poll(0.0001):
            continue
        cmd, data = remote.recv()
        if cmd == "step":
            episode_steps += 1
            action = data
            # Take a step in the environment
            next_state, reward, done, failed, info = env_step(action)
            # Return the observations, avail_actions and state to make the next action
            if done:
                next_state = env_reset()
            remote.send([next_state, reward, done,
                        failed, episode_steps, info])
            if done:
                episode_steps = 0

        elif cmd == "sample":  # sample one step
            episode_steps += 1
            action = env.action_space.sample()
            if np.random.rand() < 0.9:
                action = np.clip(action, -0.9, 0.9)
            next_state, reward, done, failed, info = env_step(action)
            if done:
                next_state = env_reset()
            remote.send([next_state, action, reward,
                        done, failed, episode_steps, info])
            if done:
                episode_steps = 0

        elif cmd == "sample_actions":  # sample action only
            action = env.action_space.sample()
            remote.send(action)

        elif cmd == "reset":
            state = env_reset()
            episode_steps = 0
            remote.send(state)

        elif cmd == "close":
            env.close()
            remote.close()
            break

        elif cmd == "info":
            remote.send([env.observation_space, env.action_space])
        else:
            raise NotImplementedError
