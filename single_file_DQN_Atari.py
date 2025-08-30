import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import gymnasium as gym
import random
import ale_py
from dataclasses import dataclass
import os
import datetime

### * 1. Replay Buffer
class ReplayBuffer:
    # NOTE: Since this is designed for atari, we only need the arguments to __init__ below, but with other envs we may need the action_shape, etc.
    def __init__(self, capacity, state_shape=(4,84,84), device='cuda'):
        """ Initialize ReplayBuffer - designed for Atari envs only """
        self.capacity = capacity
        self.pointer_index = 0
        self.buffer_size = 0
        self.device = device

        # NOTE: Dtypes are significant here.. using higher dtypes than necessary is memory inefficient. 
        # It can make a huge difference depending on a number of factors
        # At the very least, its a good habit to get into
        self.states = np.zeros((capacity,*state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # NOTE: Since DQN clips rewards to {-1, 0, 1}, you could theoretically use:
        # self.rewards = np.zeros(capacity, dtype=np.int8)
        # But float32 is more general and not much memory overhead for 1D array
        self.next_states = np.zeros( (capacity,*state_shape),dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype = np.bool_)

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.states[self.pointer_index] = state
        self.actions[self.pointer_index] = action
        self.rewards[self.pointer_index] = reward
        self.next_states[self.pointer_index] = next_state
        self.dones[self.pointer_index] = done

        # Increase the pointer index by 1 for next batch, but use mod operator to compensate for reset of buffer when its too large
        self.pointer_index = (self.pointer_index + 1) % self.capacity
        # NOTE: Here is why we we use the mod operator
        # When pointer_index=4 and capacity=5: (4+1) % 5 = 0
        # When pointer_index=0 and capacity=5: (0+1) % 5 = 1

        self.buffer_size = min(self.buffer_size + 1, self.capacity)
        # We use min for the same reason as above
    
    # def sample(self, batch_size) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
    #     """ Return random batch as tensors"""
    #     sample_indicies = np.random.choice(self.buffer_size,batch_size,replace=False)


    #     return (
    #         torch.FloatTensor(self.states[sample_indicies]).to(self.device),
    #         torch.LongTensor(self.actions[sample_indicies]).to(self.device),
    #         torch.FloatTensor(self.rewards[sample_indicies]).to(self.device),
    #         torch.FloatTensor(self.next_states[sample_indicies]).to(self.device),
    #         torch.BoolTensor(self.dones[sample_indicies]).to(self.device)
    #     )
    def sample(self, batch_size) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        """
        Improved sample() method
        
        CPU sampling → GPU transfer is more efficient than creating on GPU directly
        (i.e torch.FloatTensor(self.states[sample_indicies]).to(self.device) is less efficient than creating on CPU and transfering to GPU)
        """
        indices = np.random.choice(self.buffer_size, batch_size, replace=False)
        
        # Create tensors on CPU first, then transfer to GPU
        states = torch.from_numpy(self.states[indices]).float()
        actions = torch.from_numpy(self.actions[indices]).long()
        rewards = torch.from_numpy(self.rewards[indices]).float()
        next_states = torch.from_numpy(self.next_states[indices]).float()
        dones = torch.from_numpy(self.dones[indices]).bool()
        
        # Transfer entire batch to GPU at once (efficient)
        return (
            states.to(self.device, non_blocking=True),
            actions.to(self.device, non_blocking=True),
            rewards.to(self.device, non_blocking=True),
            next_states.to(self.device, non_blocking=True),
            dones.to(self.device, non_blocking=True)
        )
    
    def __len__(self):
        # Return current size
        return self.buffer_size
    



### * 2. PreProcessor
class AtariPreprocessor:
    """ Implementation with manual circular buffer """
    
    def __init__(self, frame_skip=4, stack_size=4):
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        
        # Manual buffer management
        self.frame_buffer = np.zeros((stack_size, 84, 84), dtype=np.uint8)
        self.buffer_index = 0
        
    def reset(self, initial_frame):
        processed = self.preprocess_frame(initial_frame)
        
        # Fill entire buffer with initial frame
        for i in range(self.stack_size):
            self.frame_buffer[i] = processed
            
        return self.frame_buffer.copy()
    
    def step(self, env, action):
        total_reward = 0
        done = False
        
        for _ in range(self.frame_skip):
            frame, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                done = True
                break
                
        processed_frame = self.preprocess_frame(frame)
        
        # Manually manage circular buffer
        self.buffer_index = (self.buffer_index + 1) % self.stack_size
        self.frame_buffer[self.buffer_index] = processed_frame
        
        # Return frames in chronological order
        # Roll array so newest frame is last
        stacked = np.roll(self.frame_buffer, -self.buffer_index - 1, axis=0)
        
        return stacked.copy(), total_reward, done, info
    
    def preprocess_frame(self, frame):
        # Same as before
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
        # crops to 84x84
        cropped = resized[18:102, :]
        return cropped.astype(np.uint8)

def example_preprocess():


    # Create environment and preprocessor
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    preprocessor = AtariPreprocessor(frame_skip=4, stack_size=4)

    # Episode loop
    obs, info = env.reset()
    stacked_state = preprocessor.reset(obs)

    print(f"Initial stacked state shape: {stacked_state.shape}")  # (4, 84, 84)

    done = False
    total_reward = 0

    while not done:

        action = env.action_space.sample()  # Random action for example
        
        # Step with preprocessing
        next_state, reward, done, info = preprocessor.step(env, action)
        
        print(f"State shape: {next_state.shape}")  # (4, 84, 84)
        print(f"Reward: {reward}")
        
        total_reward += reward
        stacked_state = next_state

    print(f"Episode finished with total reward: {total_reward}")




def store_transition(replay_buffer:ReplayBuffer, state, action, reward, next_state, done):
    """
    Store transition in replay buffer
    
    Args:
        state: Current stacked state (4, 84, 84)
        action: Action taken (scalar)
        reward: Reward received (scalar)
        next_state: Next stacked state (4, 84, 84)
        done: Episode termination flag
    """
    replay_buffer.push(
        state=state,           # (4, 84, 84) uint8
        action=action,         # int
        reward=reward,         # float
        next_state=next_state, # (4, 84, 84) uint8
        done=done             # bool
    )

### * Deep Q-Net Architechture

class DQN(nn.Module):
    def __init__(self, state_channels:int, num_actions:int, device:str='cuda') -> None:
        '''
        state_channels:int = len(batch_size, channels, height, width)
        num_actions:int = number of possible actions
        '''
        super(DQN,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=state_channels,
                out_channels=16,
                kernel_size=(8,8),
                stride=4,
                device=device
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(4,4),
                stride=2,
                device=device
            ),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate MLP input size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, state_channels, 84, 84, device=device)
            n_flatten = self.conv_layers(dummy).shape[1]
            
        self.action_classifier = nn.Sequential(
            nn.Linear(n_flatten, 256, device=device),
            nn.ReLU(),
            nn.Linear(256, num_actions,device=device)
        )
    def forward(self,x):
        x = self.conv_layers(x)
        q_values = self.action_classifier(x)
        return q_values


### * Trainer




class DQNTrainer:
    def __init__(self, q_network, target_network, optimizer, replay_buffer,
                 batch_size, gamma, target_update_freq,run_id=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), use_amp=False, device='cuda', scaler=None, log_path="logs") -> None:
        """
        Initialize the DQN trainer with networks, optimizer, and hyperparameters.

        Args:
            q_network (nn.Module): Main Q-network for training
            target_network (nn.Module): Target Q-network for stable targets
            optimizer (torch.optim): Optimizer for the main Q-network
            replay_buffer (ReplayBuffer): Experience replay buffer
            batch_size (int): Default batch size for training
            gamma (float): Discount factor for future rewards
            target_update_freq (int): Steps between target network updates
            use_amp (bool): Whether to use mixed precision training
            device (str): Device to run computations on ('cuda' or 'cpu')
        """
        self.q_network = q_network
        self.use_amp = use_amp
        self.target_network = target_network
        self.scaler = scaler
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.device = device
        self.log_path = log_path
        self.run_id = run_id
        self.checkpoint_dir = self.create_checkpoint_dir()


        # Training step counter for target network updates
        self.step_count = 0

        # Move networks to specified device
        self.q_network.to(device)
        self.target_network.to(device)

        # Set target network to evaluation mode (no gradient computation)
        self.target_network.eval()

        # Initialize target network with same weights as main network
        self.update_target_network()

    def create_checkpoint_dir(self):
        """Create log directory if it doesn't exist"""
        if not os.path.exists(self.log_path + "/" + self.run_id ):
            os.makedirs(self.log_path + "/" + self.run_id)
            self.log_path = self.log_path + "/" + self.run_id
            print(f"Log directory created at {self.log_path}")
        else:
            self.log_path = self.log_path + "/" + self.run_id
            print(f"Log directory already exists at {self.log_path}")

        
        return self.log_path

    def compute_td_loss(self, batch_size):
        """
        Compute temporal difference loss using sampled batch from replay buffer.

        This implements the core DQN loss function:
        L = E[(r + γ * max_a Q_target(s', a) - Q(s, a))²]

        Args:
            batch_size (int): Number of transitions to sample

        Returns:
            torch.Tensor: MSE loss between current and target Q-values
            None: If replay buffer doesn't have enough samples
        """
        # Check if replay buffer has enough samples
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample random batch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Move all tensors to the specified device (GPU/CPU)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current q-values: Q(s, a) for actions that were taken
        current_q_values = self.q_network(states)  # Shape: (batch_size, num_actions)

        # Select Q-values for the specific actions that were taken
        #NOTE: FYI - gather() selects values at indices specified by actions
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Final shape: (batch_size,)

        # Compute target q-values using target network (no gradients)
        with torch.no_grad():
            # Get Q-values for next states from target network
            next_q_values = self.target_network(next_states)  # Shape: (batch_size, num_actions)

            # Find maximum Q-value for each next state: max_a Q_target(s', a)
            max_next_q_values = next_q_values.max(1)[0]  # Shape: (batch_size,)

            # Apply the Bellman equation: r + γ * max_a Q_target(s', a)
            # For terminal states (done=True), target = reward only
            # For non-terminal states, target = reward + discounted future value
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones.float()))

        # Compute Mean Squared Error loss between current and target Q-values
        loss = F.mse_loss(current_q_values, target_q_values)

        return loss

    def update_target_network(self):
        """
        Copy weights from main Q-network to target Q-network.

        This provides stable targets for training by using a separate network
        that is updated less frequently than the main network.
        """
        # Copy all parameters from main network to target network
        self.target_network.load_state_dict(self.q_network.state_dict())



    def train_step(self, batch_size=None):
        """Optimized training step with mixed precision"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch (CPU → GPU transfer)
        # states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Mixed precision training
        if self.use_amp:
            with torch.amp.autocast():
                loss = self.compute_td_loss(batch_size)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.compute_td_loss(batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

    def should_update_target(self, step_count):
        """
        Check if target network should be updated based on step count.

        Args:
            step_count (int): Current training step number

        Returns:
            bool: True if target network should be updated
        """
        return step_count % self.target_update_freq == 0

    def get_q_values(self, states):
        """
        Get Q-values from main network for given states.

        Used for action selection during environment interaction.

        Args:
            states (torch.Tensor): Batch of states

        Returns:
            torch.Tensor: Q-values for all actions for each state
        """
        # Ensure network is in evaluation mode for inference
        self.q_network.eval()

        with torch.no_grad():
            # Move states to correct device
            states = states.to(self.device)

            # Get Q-values for all actions
            q_values = self.q_network(states)

        # Return to training mode
        self.q_network.train()

        return q_values

    def get_target_q_values(self, next_states):
        """
        Get Q-values from target network for given next states.

        Used primarily for debugging or analysis. Normal training uses
        compute_td_loss() which handles target Q-value computation internally.

        Args:
            next_states (torch.Tensor): Batch of next states

        Returns:
            torch.Tensor: Q-values from target network
        """
        with torch.no_grad():
            # Move states to correct device
            next_states = next_states.to(self.device)

            # Get Q-values from target network
            target_q_values = self.target_network(next_states)

        return target_q_values


    def set_training_mode(self, training=True):
        """
        Set training mode for the main Q-network.

        Args:
            training (bool): True for training mode, False for evaluation
        """
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()

        # Target network should always be in eval mode
        self.target_network.eval()




class DQNAgent:
    def __init__(self, q_network, target_network, replay_buffer,run_id, optimizer:str="Adam", log_path="logs", use_amp=False,
                 learning_rate=0.00025, batch_size=32, gamma=0.99, 
                 target_update_freq=10000, device='cuda', num_actions=4):
        """
        Main DQN Agent that handles both environment interaction and training.
        """
        self.q_network = q_network
        self.target_network = target_network
        self.log_path = log_path
        self.num_actions = num_actions
        self.device = device
        self.replay_buffer = replay_buffer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.run_id = run_id
        
        # Enable Automatic Mixed Precision for RTX 5090
        self.use_amp = use_amp
        if use_amp:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None
        
        # Create optimizer
        if optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(q_network.parameters(), lr=learning_rate)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.. please choose from RMSprop or Adam or implement your own optimizer")
        
        # Create trainer for training logic (no checkpoint methods)
        self.trainer = DQNTrainer(
            q_network=self.q_network,
            target_network=self.target_network, 
            optimizer=self.optimizer,
            replay_buffer=self.replay_buffer,
            batch_size=self.batch_size,
            gamma=self.gamma,
            target_update_freq=self.target_update_freq,
            device=self.device,
            scaler=self.scaler,
            run_id=self.run_id,
            log_path=self.log_path
        )

        self.checkpoint_dir = self.trainer.create_checkpoint_dir()
        
        # Move networks to device
        self.q_network.to(device)
        self.target_network.to(device)


    
    def select_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        # Prepare state for network
        if state.dim() == 3:  # Add batch dimension
            state = state.unsqueeze(0)

        state = state.float() / 255.0 if state.dtype == torch.uint8 else state
        state = state.to(self.device)
        
        # Get Q-values and select best action
        with torch.no_grad():
            self.q_network.eval()
            q_values = self.q_network(state)
            self.q_network.train()
        
        return q_values.argmax(dim=1).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Delegate training to trainer"""
        return self.trainer.train_step()
    
    def save_checkpoint(self, **additional_data):
        """
        Save complete agent state to checkpoint file.
        
        Args:
            filepath (str): Path to save checkpoint
            **additional_data: Additional data to save (e.g., episode count, total steps)
        """
        checkpoint = {
            # Network states
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            
            # Agent hyperparameters
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'target_update_freq': self.target_update_freq,
                'num_actions': self.num_actions,
                'device': self.device
            },
            
            # Training state
            'trainer_step_count': self.trainer.step_count,
            
            # Network architecture info (for reconstruction)
            'network_config': {
                'input_channels': self.q_network.conv_layers[0].in_channels,
                'num_actions': self.num_actions,
                # Add other architecture details if needed
            },
            
            # Additional training data
            **additional_data
        }
        
        torch.save(checkpoint, self.checkpoint_dir + "/checkpoint.pt")
        print(f"Checkpoint saved to {self.checkpoint_dir + "/checkpoint.pt"}")
    
    @classmethod
    def load_from_checkpoint(cls, filepath, replay_buffer, q_network_class=None, **network_kwargs):
        """
        Load agent from checkpoint file.
        
        Args:
            filepath (str): Path to checkpoint file
            replay_buffer (ReplayBuffer): Replay buffer to use (not saved in checkpoint)
            q_network_class (nn.Module): Network class to instantiate (if None, assumes networks exist)
            **network_kwargs: Additional arguments for network construction
            
        Returns:
            DQNAgent: Fully configured agent loaded from checkpoint
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')  # Load to CPU first
        
        # Extract hyperparameters
        hyperparams = checkpoint['hyperparameters']
        
        # Create networks if not provided
        if q_network_class is not None:
            # Create new networks from class
            q_network = q_network_class(**network_kwargs)
            target_network = q_network_class(**network_kwargs)
        else:
            raise ValueError("Must provide q_network_class to reconstruct networks from checkpoint")
        
        # Create agent with loaded hyperparameters
        agent = cls(
            q_network=q_network,
            target_network=target_network,
            replay_buffer=replay_buffer,
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            gamma=hyperparams['gamma'],
            target_update_freq=hyperparams['target_update_freq'],
            device=hyperparams['device'],
            num_actions=hyperparams['num_actions']
        )
        
        # Load network weights
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        # Load optimizer state
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        agent.trainer.step_count = checkpoint['trainer_step_count']
        
        # Move to correct device
        agent.q_network.to(agent.device)
        agent.target_network.to(agent.device)
        
        print(f"Agent loaded from {filepath} at training step {agent.trainer.step_count}")
        
        return agent
    
    def get_training_stats(self):
        """Get comprehensive training statistics"""
        return {
            'step_count': self.trainer.step_count,
            'buffer_size': self.replay_buffer.__len__(),
            'target_updates': self.trainer.step_count // self.target_update_freq,
            'learning_rate': self.learning_rate,
            'device': self.device,
            'hyperparameters': {
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'target_update_freq': self.target_update_freq
            }
        }

def monitor_gpu_usage():
    """Monitor GPU memory during training"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# # Call periodically during training to see GPU usage
# if step % 1000 == 0:
#     monitor_gpu_usage()

# NOTE: These batch sizes were calculated with the assumption that it will be run on an Nvidia RTX 5090, which has 32GB of VRAM.
# You may need to adjust these batch sizes for your own hardware.
BATCH_SIZES = {
    "Standard DQN": 256,       # 8x larger than paper (32)
    "Large Batch": 512,        # Even larger for stability
    "Memory Limit": 1024,      # Maximum before running out - ASSUMING 32GB of VRAM i.e Nvidia 5090
}

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@dataclass
class TrainingConfig:
    env_id: str # environment id
    rb_capacity: int # replay buffer capacity
    total_time_steps: int # total number of time steps
    use_amp: bool = False # use automatic mixed precision
    device: str = "cuda" # device NOTE: Currently only supports CUDA, but will be updated to support CPU & MPS in the future
    batch_size: int = 256 # batch size
    # epsilon: float = 0.0 # epsilon
    epsilon_start: float = 1.0 # epsilon start
    epsilon_end: float = 0.01 # epsilon end
    epsilon_decay_fraction: float = .5 # epsilon decay fraction (fraction of total time steps it takes to decay to epsilon_end)
    log_path: str = "logs" # log path
    # num_steps_per_update: int = 4 # number of steps per update
    target_update_freq: int = 10000 # target update frequency
    run_id: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # run id
    optimizer: str = "Adam" # optimizer NOTE: The paper uses RMSprop, but Adam is more stable and faster to train. You may choose to use RMSprop or implement your own optimizer.

def train_new_agent(config: TrainingConfig):

    gym.register_envs(ale_py)
    
    env = gym.make(config.env_id)

    state_shape = env.observation_space._shape

    preprocessor = AtariPreprocessor(frame_skip=4,stack_size=4)

    rb = ReplayBuffer(capacity=config.rb_capacity,state_shape=(4,84,84),device=config.device)

    q_network = DQN(state_channels=(1+ len(state_shape)),num_actions=env.action_space.n)
    target_network = DQN(state_channels=(1+ len(state_shape)),num_actions=env.action_space.n)

    agent = DQNAgent(q_network,target_network,rb,use_amp=config.use_amp,run_id=config.run_id,device=config.device,optimizer=config.optimizer)

    agent.trainer.set_training_mode(True)

    state, info = env.reset()
    next_state = preprocessor.reset(state)

    for step in range(config.total_time_steps):

        state = next_state
        epsilon = linear_schedule(config.epsilon_start, config.epsilon_end, config.epsilon_decay_fraction * config.total_time_steps , step)
        action = agent.select_action(state,epsilon)
        next_state, reward, done, info = preprocessor.step(env,action)

        agent.store_experience(state,action,reward,next_state,done)

        if step % config.batch_size == 0:
            loss = agent.train_step()

        if step % 1000 == 0:
            print(f"Step {step} - Loss: {loss}")
            agent.save_checkpoint(step=step)
            monitor_gpu_usage()

        if done:
            state, info = env.reset()
            next_state = preprocessor.reset(state)
            

        if step % config.target_update_freq == 0:
            agent.trainer.update_target_network()


            

    env.close()
        

if __name__ == "__main__":
    config = TrainingConfig(
        env_id="PongNoFrameskip-v4",
        rb_capacity=100_000,
        total_time_steps=10_000_000,
        use_amp=True,
        device="cuda",
        batch_size=256,
        # epsilon=0.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_fraction=0.5,
        log_path="logs",
        target_update_freq=10000,
        run_id="test_run2_10m_steps",
        optimizer="Adam"
    )
    train_new_agent(config)