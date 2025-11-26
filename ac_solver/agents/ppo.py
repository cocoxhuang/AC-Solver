"""
This file trains a PPO (Proximal Policy Optimization) agent on AC Environment.
It sets up the training environment, initializes the agent, and runs the PPO training loop.
Run this script directly to start training the PPO agent, as simply as 

```
python ppo.py
```

To see the entire list of command line arguments you may pass, check args.py
"""

import numpy as np
import torch
import random
from torch.optim import Adam
from ac_solver.agents.ppo_agent import Agent, initialize_layer
from ac_solver.agents.args import parse_args
from ac_solver.agents.environment import get_env
from ac_solver.agents.training import ppo_training_loop
from pretrain.models.models import FFN

def train_ppo():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    (
        envs,
        initial_states,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
    ) = get_env(args)

    agent = Agent(envs, args.nodes_counts).to(device)
    
    # load the pretrained model if specified
    if args.pretrained_actor_model_path is not None:        
        # Load the pretrained state dict
        print("Loading pretrained actor model from:", args.pretrained_actor_model_path)
        pretrained_state = torch.load(args.pretrained_actor_model_path, map_location=device)
        pretrained_dims = [param.shape[1] for name, param in pretrained_state.items() if 'weight' in name]
        pretrained_dims.append(pretrained_state[list(pretrained_state.keys())[-1]].shape[0])  # output layer size

        # load the pretrained actor model
        agent.actor = FFN(pretrained_dims[0], pretrained_dims[1:-1], pretrained_dims[-1])
        agent.actor.load_state_dict(pretrained_state)
        print("Pretrained actor model loaded successfully.")

        # pretrain model is only for smaller action space, so we need to modify the last layer
        print("Modifying the last layer of the pretrained actor model to match the new action space size.")
        last_layer = agent.actor.layers[-1]
        padded_last_layer = torch.nn.Linear(last_layer.in_features, envs.single_action_space.n)
        # initialize new weights
        initialize_layer(padded_last_layer, std=0.01)
        # copy old weights from the pretrained model
        # copy into the top-left submatrix: [0:last_out, 0:last_in]
        last_out = last_layer.out_features
        last_in = last_layer.in_features
        padded_last_layer.weight.data[:last_out, :last_in] = last_layer.weight.data
        # copy bias for the first last_out entries
        padded_last_layer.bias.data[:last_out] = last_layer.bias.data
        agent.actor.layers[-1] = padded_last_layer
        print("Last layer modified successfully.")

        agent.actor.to(device)

    optimizer = Adam(agent.parameters(), lr=args.learning_rate, eps=args.epsilon)

    ppo_training_loop(
        envs,
        args,
        device,
        optimizer,
        agent,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
        initial_states,
    )

    envs.close()


if __name__ == "__main__":
    train_ppo()
