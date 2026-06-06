import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch

Transition = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayMemory:
    """
    Positive/Negative buffer-aware replay memory.

    Transitions with positive feedback (e.g. reward >= 0) and negative feedback
    are stored in separate buffers so that sampling can draw from both fairly.
    """

    def __init__(self, capacity: int, seed: int):
        random.seed(seed)
        self.capacity = capacity

        self.pos_buffer: List[Transition] = []
        self.neg_buffer: List[Transition] = []

        self.pos_position: int = 0
        self.neg_position: int = 0

    def _push_to_buffer(self, buffer: List[Transition], position: int, transition: Transition):
        if len(buffer) < self.capacity:
            buffer.append(None)  # placeholder
        buffer[position] = transition
        position = (position + 1) % self.capacity
        return position

    def push(self, state, action, reward, next_state, done, positive: bool):
        transition = (state, action, reward, next_state, done)
        if positive:
            self.pos_position = self._push_to_buffer(self.pos_buffer, self.pos_position, transition)
        else:
            self.neg_position = self._push_to_buffer(self.neg_buffer, self.neg_position, transition)

    def _prepare_batch(self, buffer: List[Transition], count: int) -> List[Transition]:
        if count <= 0:
            return []
        if len(buffer) < count:
            raise ValueError(f"Not enough samples in buffer to satisfy request: requested {count}, available {len(buffer)}")
        return random.sample(buffer, count)

    def sample(self, batch_size: int):
        if len(self) < batch_size:
            raise ValueError(f"Not enough samples to draw a batch of size {batch_size}. Current size: {len(self)}")

        half = batch_size // 2
        pos_available = len(self.pos_buffer)
        neg_available = len(self.neg_buffer)

        pos_take = min(pos_available, half)
        neg_take = min(neg_available, half)

        remainder = batch_size - pos_take - neg_take

        def take_extra(available, taken):
            return max(0, available - taken)

        if remainder > 0:
            # allocate remaining slots to the buffer with more remaining samples
            pos_remaining = take_extra(pos_available, pos_take)
            neg_remaining = take_extra(neg_available, neg_take)

            if pos_remaining >= neg_remaining and pos_remaining > 0:
                extra = min(remainder, pos_remaining)
                pos_take += extra
                remainder -= extra

            if remainder > 0 and neg_remaining > 0:
                extra = min(remainder, neg_remaining)
                neg_take += extra
                remainder -= extra

        if remainder > 0:
            raise ValueError("Unable to fulfil batch request from replay buffers.")

        batch = []
        if pos_take > 0:
            batch.extend(self._prepare_batch(self.pos_buffer, pos_take))
        if neg_take > 0:
            batch.extend(self._prepare_batch(self.neg_buffer, neg_take))

        random.shuffle(batch)

        states, actions, rewards, next_states, dones = zip(*batch)

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.array(x)

        state = np.stack([to_numpy(s) for s in states])
        action = np.stack([to_numpy(a) for a in actions])
        reward = np.stack([to_numpy(r) for r in rewards])
        next_state = np.stack([to_numpy(ns) for ns in next_states])
        done = np.stack([to_numpy(d) for d in dones])

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.pos_buffer) + len(self.neg_buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        payload = {
            "capacity": self.capacity,
            "pos_buffer": self.pos_buffer,
            "neg_buffer": self.neg_buffer,
            "pos_position": self.pos_position,
            "neg_position": self.neg_position,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(payload, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            payload = pickle.load(f)

        self.capacity = payload.get("capacity", self.capacity)
        self.pos_buffer = payload.get("pos_buffer", [])
        self.neg_buffer = payload.get("neg_buffer", [])
        self.pos_position = payload.get("pos_position", len(self.pos_buffer) % self.capacity)
        self.neg_position = payload.get("neg_position", len(self.neg_buffer) % self.capacity)

