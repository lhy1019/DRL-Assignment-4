# sac_dmc_humanoid/replay_buffer.py

import numpy as np
import torch

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device # device is a torch.device object

        self.obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            # Or raise an error: raise ValueError("Cannot sample from an empty buffer.")
            # Or return None and handle it in the agent's update logic.
            # For now, let's assume the agent checks len(buffer) >= batch_size before calling sample.
            # If we must return something, empty tensors might be an option, but can lead to downstream errors.
            # The most robust is to ensure this is not called when empty or when size < batch_size
            # if not strictly allowing sampling with replacement from a smaller set.
            # Let's stick to the original assumption that the caller ensures enough samples.
             raise ValueError(f"Cannot sample {batch_size} from a buffer of size {self.size}. Ensure buffer size >= batch_size for sampling.")


        # Sample 'batch_size' indices with replacement. This is standard.
        # If self.size < batch_size, it will sample from the available self.size elements,
        # leading to duplicates in the batch, which is acceptable.
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32).to(self.device),
            actions=torch.as_tensor(self.action_buf[idxs], dtype=torch.float32).to(self.device),
            rewards=torch.as_tensor(self.reward_buf[idxs], dtype=torch.float32).to(self.device),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], dtype=torch.float32).to(self.device),
            dones=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(self.device)
        )
        return batch

    def __len__(self):
        return self.size

# --- Example Usage (for testing the replay buffer) ---
if __name__ == '__main__':
    obs_dim_env = 67
    action_dim_env = 21
    buffer_capacity = 1000
    batch_s = 32
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device for tensors: {test_device}")

    replay_buffer = ReplayBuffer(buffer_capacity, obs_dim_env, action_dim_env, test_device)

    print(f"Initial buffer size: {len(replay_buffer)}")
    assert len(replay_buffer) == 0

    # Store some dummy transitions
    num_dummy_transitions = 500 # Ensure this is >= batch_s for the first sampling test
    for i in range(num_dummy_transitions):
        dummy_obs = np.random.randn(obs_dim_env).astype(np.float32)
        dummy_action = np.random.randn(action_dim_env).astype(np.float32)
        dummy_reward = np.random.rand()
        dummy_next_obs = np.random.randn(obs_dim_env).astype(np.float32)
        dummy_done = np.random.choice([True, False])
        
        replay_buffer.store(dummy_obs, dummy_action, dummy_reward, dummy_next_obs, dummy_done)
    
    print(f"Buffer size after {num_dummy_transitions} stores: {len(replay_buffer)}")
    assert len(replay_buffer) == num_dummy_transitions
    assert replay_buffer.ptr == num_dummy_transitions % buffer_capacity

    # Test sampling
    if len(replay_buffer) >= batch_s: # Standard check before sampling
        print(f"\nSampling a batch of size {batch_s}")
        batch_data = replay_buffer.sample(batch_s)
        
        assert isinstance(batch_data, dict)
        
        def check_device(tensor_device, target_device):
            # Simpler check for torch.device objects if index might be None for target
            if target_device.index is None: # e.g. torch.device('cpu') or torch.device('cuda')
                return tensor_device.type == target_device.type
            # For specific devices like torch.device('cuda:0')
            return tensor_device.type == target_device.type and tensor_device.index == target_device.index

        assert 'obs' in batch_data and batch_data['obs'].shape == (batch_s, obs_dim_env)
        assert check_device(batch_data['obs'].device, test_device)
        assert batch_data['obs'].dtype == torch.float32

        assert 'actions' in batch_data and batch_data['actions'].shape == (batch_s, action_dim_env)
        assert check_device(batch_data['actions'].device, test_device)
        assert batch_data['actions'].dtype == torch.float32

        assert 'rewards' in batch_data and batch_data['rewards'].shape == (batch_s, 1)
        assert check_device(batch_data['rewards'].device, test_device)
        assert batch_data['rewards'].dtype == torch.float32

        assert 'next_obs' in batch_data and batch_data['next_obs'].shape == (batch_s, obs_dim_env)
        assert check_device(batch_data['next_obs'].device, test_device)
        assert batch_data['next_obs'].dtype == torch.float32

        assert 'dones' in batch_data and batch_data['dones'].shape == (batch_s, 1)
        assert check_device(batch_data['dones'].device, test_device)
        assert batch_data['dones'].dtype == torch.float32
        assert torch.all((batch_data['dones'] == 0.0) | (batch_data['dones'] == 1.0))

        print("Sampled batch shapes, types, and device are correct.")
    else:
        print(f"Skipping sampling test as buffer size ({len(replay_buffer)}) < batch size ({batch_s})")

    # Test buffer overflow (circular property)
    print("\nTesting buffer overflow (circular property)")
    replay_buffer_overflow_test = ReplayBuffer(buffer_capacity, obs_dim_env, action_dim_env, test_device)
    total_stores_for_overflow = buffer_capacity + 100
    
    for i in range(total_stores_for_overflow):
        # (same dummy data generation as before)
        dummy_obs = np.random.randn(obs_dim_env).astype(np.float32); dummy_action = np.random.randn(action_dim_env).astype(np.float32)
        dummy_reward = np.random.rand(); dummy_next_obs = np.random.randn(obs_dim_env).astype(np.float32)
        dummy_done = np.random.choice([True, False])
        replay_buffer_overflow_test.store(dummy_obs, dummy_action, dummy_reward, dummy_next_obs, dummy_done)

    print(f"Buffer size after {total_stores_for_overflow} total stores (capacity {buffer_capacity}): {len(replay_buffer_overflow_test)}")
    assert len(replay_buffer_overflow_test) == buffer_capacity
    
    expected_ptr = total_stores_for_overflow % buffer_capacity
    print(f"Current ptr: {replay_buffer_overflow_test.ptr}, Expected ptr: {expected_ptr}")
    assert replay_buffer_overflow_test.ptr == expected_ptr 
    print("Buffer overflow and pointer wrapping seems correct.")

    print("\nReplayBuffer implementation seems correct.")