from typing import Literal
from tap import Tap


class Hyperparams(Tap):
    env: str = 'miniatari'

    arch: Literal['gru', 'nn'] = 'gru'  # What function approximation architecture do we use? (gru | nn)
    algo: Literal['sarsa', 'policy_grad', 'actor_critic', 'sac'] = 'sarsa'  # What RL algorithm do we use? (sarsa | policy_grad | actor_critic)
    optimizer: Literal['adam', 'sgd', 'rmsprop'] = 'adam'

    gamma: float = 0.99
    epsilon: float  = 0.1

    head_layers: int = 1  # After the GRU, for our NN head, how many layers do we have?
    rnn_hidden_size: int = 100  # Hidden size of our RNN hidden state.
    head_hidden_size: int = 100  # Hidden size of our value neural network.
    replay_size: int = None  # Size of our replay buffer.
    batch_size: int = 16
    trunc_length: int = 10  # [RNNs] truncation length for T-BPTT
    action_cond: bool = False  # [RNNs] Do we condition on actions?
    freeze_rnn: bool = False  # [RNNs] do we freeze the recurrent part of our network?
    augment_fixed_mem: str = None  # [FIXED MEMORY] What kind of fixed memory augmentation do we do? (random_discrete | None)
    n_mem_states: int = 2  # [RANDOM DISCRETE] How many memory states?
    step_size: float = 5e-4

    max_episode_steps: int = int(1e3)
    total_steps: int = int(1e6)
    checkpoint_freq: int = -1
    offline_eval_freq: int = None
    offline_eval_episodes: int = 1
    seed: int = 2020
    platform: Literal['cpu', 'gpu'] = 'cpu'
    study_name: str = 'test'

    def process_args(self) -> None:
        if self.augment_fixed_mem == 'None':
            self.augment_fixed_mem = None
