"""
Andrews-Curtis (AC) Environment for balanced presentations with two generators.
"""

from dataclasses import dataclass, field
from typing import Union
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from ac_solver.envs.ac_moves import ACMove, substituition
from ac_solver.envs.utils import is_array_valid_presentation, simplify_relator


@dataclass
class ACEnvConfig:
    initial_state: Union[np.ndarray, list] = field(
        default_factory=lambda: np.array([1, 0, 2, 0])
    )  # trivial state <x, y>
    horizon_length: any = 1000
    use_supermoves: any = False

    def __post_init__(self):
        # Convert initial_state to a numpy array if it's a list
        if isinstance(self.initial_state, list):
            self.initial_state = np.array(self.initial_state)

        # Assert that initial_state is a 1-dimensional numpy array
        if not isinstance(self.initial_state, np.ndarray):
            raise TypeError("initial_state must be a numpy array")
        if self.initial_state.ndim != 1:
            raise ValueError("initial_state must be a 1-dimensional array")
        if len(self.initial_state) % 2 != 0:
            raise ValueError("initial state must have even length")
        if not is_array_valid_presentation(self.initial_state):
            raise ValueError("initial state must be a valid presentation")

    @property
    def max_relator_length(self):
        return len(self.initial_state) // 2

    @property
    def max_relator_length(self):
        return len(self.initial_state) // 2

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            initial_state=np.array(
                config_dict.get("initial_state", cls().initial_state)
            ),
            horizon_length=config_dict.get("horizon_length", cls().horizon_length),
            use_supermoves=config_dict.get("use_supermoves", cls().use_supermoves),
        )

class ACEnv(Env):
    
    def __init__(self, config: ACEnvConfig = ACEnvConfig()):
        self.n_gen = 2  # number of generators of the presentation
        self.max_relator_length = config.max_relator_length
        self.initial_state = config.initial_state
        self.horizon_length = config.horizon_length
        if config.use_supermoves:
            raise NotImplementedError(
                "ACEnv with supermoves is not yet implemented in this library."
            )
        else:
            self.supermoves = None

        # state space
        low = np.ones(self.max_relator_length * self.n_gen, dtype=np.int8) * (
            -self.n_gen
        )
        high = np.ones(self.max_relator_length * self.n_gen, dtype=np.int8) * (
            self.n_gen
        )
        self.observation_space = Box(low, high, dtype=np.int8)

        # action space
        self._substitution_dict = self.substitution_dict()
        self.action_space = Discrete(len(self._substitution_dict))

        # max-reward needed for reward function
        self.max_reward = self.horizon_length * self.max_relator_length * self.n_gen

        # variables to keep track of current state of the environment
        self.state = np.copy(self.initial_state)  # current state
        self.count_steps = 0  # number of environment steps
        self.lengths = [
            np.count_nonzero(
                self.state[
                    i * self.max_relator_length : (i + 1) * self.max_relator_length
                ]
            )
            for i in range(self.n_gen)
        ]  # lengths of relators in the current state
        self.actions = []  # list of actions from the initial state

    def get_action_mask(self, state=None):
        """
        Compute a boolean mask over actions indicating which substitutions are valid
        (i.e., will not increase the length of either relator beyond `max_relator_length`).

        Parameters:
        state (np.ndarray, optional): Presentation state to evaluate. If None, uses current `self.state`.

        Returns:
        np.ndarray: Boolean array of shape (num_actions,) where True denotes a valid action.
        """
        s = np.copy(self.state) if state is None else np.array(state, copy=True)
        maxL = self.max_relator_length
        # Extract non-padded relators
        r1, r2 = s[:self.max_relator_length], s[self.max_relator_length:]
        r1, r2 = r1[r1 != 0], r2[r2 != 0]

        mask = np.ones(self.action_space.n, dtype=bool)
        # Evaluate each substitution's impact after simplification
        for ind, action in self._substitution_dict.items():
            try:
                r1p, r2p = substituition(r1, r2, action)
                # Simplify with padded=False to get resultant lengths; assertion will fire if overflow
                _, len1 = simplify_relator(r1p, maxL, cylically_reduce=True, padded=False)
                _, len2 = simplify_relator(r2p, maxL, cylically_reduce=True, padded=False)
                mask[ind] = (len1 <= maxL) and (len2 <= maxL)
            except AssertionError as ve:
                if "Increase max length!" in str(ve):
                    mask[ind] = False
                else:
                    raise ve
            except Exception as e:
                raise e

        return mask

    def substitution_dict(self) -> dict:
        '''Create a substitution dictionary mapping action indices to action tuples.'''
        substitution_dict = {}
        ind = 0
        for relator in [0,1]:
            for i in range(self.max_relator_length):
                for j in range(self.max_relator_length):
                    for inverse in [True, False]:
                        substitution_dict[ind] = (relator, i, j, inverse)
                        ind += 1
        return substitution_dict

    def step(self, action):
        self.actions += [self._substitution_dict[action]]
        # now the action is a tuple 
        self.state, self.lengths = ACMove(
            self._substitution_dict[action], self.state, self.max_relator_length, self.lengths
        )

        done = sum(self.lengths) == 2
        reward = self.max_reward * done - sum(self.lengths) * (1 - done)

        self.count_steps += 1
        truncated = self.count_steps >= self.horizon_length

        return (
            self.state,
            reward,
            done,
            truncated,
            {"actions": self.actions.copy()} if done else {},
        )

    def reset(self, *, seed=None, options=None):
        self.state = (
            np.copy(options["starting_state"])
            if options and "starting_state" in options
            else np.copy(self.initial_state)
        )
        self.lengths = [
            np.count_nonzero(
                self.state[
                    i * self.max_relator_length : (i + 1) * self.max_relator_length
                ]
            )
            for i in range(self.n_gen)
        ]
        self.count_steps = 0
        self.actions = []
        return self.state, {}

    def render(self):
        pass
