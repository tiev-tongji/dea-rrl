import random
import numpy as np
import pickle
from operator import itemgetter
from itertools import chain


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, pos_fraction=None):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask

    def add_noise(self, mean, std):
        for i in range(len(self.buffer)):
            self.buffer[i] = (np.random.normal(mean, std, self.buffer[i][0].shape) + self.buffer[i][0], self.buffer[i][1], self.buffer[i][2], self.buffer[i][3], self.buffer[i][4])
        
    def zip(self, size):
        if len(self.buffer) > size:
            new_buffer = []
            new_buffer.extend(random.sample(self.buffer, size))
            del self.buffer
            self.buffer = new_buffer
            self.position = 0
            
    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.position = 0

    def save_buffer(self, save_path):
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class ConstraintReplayMemory:
    '''
        Replay buffer for training recovery policy and associated safety critic
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.pos_idx = np.zeros(self.capacity)
        self.vios = 0

    def push(self, state, action, reward, next_state, mask):
        if reward or mask:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (
                state, action, reward, next_state, mask)
            self.pos_idx[self.position] = reward > 0
            self.position = (self.position + 1) % self.capacity

            self.vios = np.sum(self.pos_idx)

    def sample(self, batch_size, pos_fraction=None):
        pos_fraction = None
        if pos_fraction is not None and self.vios > 0:
            if self.vios >= int(batch_size * pos_fraction):
                pos_size = int(batch_size * pos_fraction)
            else:
                pos_size = self.vios
            neg_size = batch_size - pos_size
            pos_idx = np.array(
                np.random.choice(tuple(np.argwhere(self.pos_idx).ravel()),
                                 pos_size))
            neg_idx = np.array(
                np.random.choice(tuple(np.argwhere((1 - self.pos_idx)[:len(self.buffer)]).ravel()),
                                 neg_size))
            idx = np.hstack((pos_idx, neg_idx))
            batch = itemgetter(*idx)(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask
    
    def violations(self):
        return self.vios

    def clear(self):
        self.buffer = []
        self.pos_idx = np.zeros(self.capacity)
        self.position = 0
        self.vios = 0

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, save_path):
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump((self.buffer, self.pos_idx), f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer, self.pos_idx = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
            self.vios = np.sum(self.pos_idx)


class UnitedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.mem = ReplayMemory(capacity)
        self.cmem = ReplayMemory(capacity)

    def push(self, state, action, reward, next_state, mask, spec=False):
        if reward > 0:
            self.cmem.push(state, action, reward, next_state, mask)
        # not push safe final state because it dosen't mean that this state is really safe
        elif mask:
            self.mem.push(state, action, reward, next_state, mask)

    def sample(self, batch_size, pos_fraction=None):
        if pos_fraction == 0:
            batch = random.sample(
                self.mem.buffer, batch_size)
            state, action, reward, next_state, done = map(
                np.stack, zip(*batch))
        else:
            if pos_fraction == None:
                pos_size = batch_size * int(len(self.cmem) / self.__len__())
            elif len(self.mem) <= int(batch_size * (1 - pos_fraction)):
                pos_size = batch_size - len(self.mem)
            elif len(self.cmem) <= int(pos_fraction * batch_size):
                pos_size = len(self.cmem)
            else:
                pos_size = int(batch_size * pos_fraction)
            pos_batch = random.sample(
                self.cmem.buffer, pos_size)
            neg_batch = random.sample(
                self.mem.buffer, batch_size - pos_size)
            state, action, reward, next_state, done = map(
                np.stack, zip(*chain(pos_batch, neg_batch)))
        return state, action, reward, next_state, done

    def zip(self, size):
        self.mem.zip(size)
        self.cmem.zip(size)
    
    def add_noise(self, mean, std):
        self.mem.add_noise(mean, std)
        self.cmem.add_noise(mean, std)
        
    def clear(self):
        self.mem.clear()
        self.cmem.clear()
        
    def __len__(self):
        return len(self.mem) + len(self.cmem)

    def violations(self):
        return len(self.cmem)

    def save_buffer(self, save_path):
        print('Saving buffer to {}'.format(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump((self.mem.buffer, self.cmem.buffer), f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, 'rb') as f:
            self.mem.buffer, self.cmem.buffer = pickle.load(f)
            self.mem.position = len(self.mem.buffer) % self.capacity
            self.cmem.position = len(self.cmem.buffer) % self.capacity
