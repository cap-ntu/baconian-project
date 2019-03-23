import numpy as np
from baconian.algo.rl.model_based.misc.reward_func.reward_func import CostFunc


# ========================================================
#
# Environment-specific cost functions:
#
def reacher_cost_fn(state, action, next_state):
    if len(state.shape) > 1:
        scores = np.zeros((state.shape[0],))

        scores += np.linalg.norm(next_state[:, -1])
        return scores

    score = np.linalg.norm(next_state[-1])
    return score


def pendulum_cost_fn(state, action, next_state):
    if len(state.shape) > 1:
        scores = np.zeros((state.shape[0],))

        scores += np.abs(next_state[:, 1])  # + 0.001 * action**2
        return scores

    score = np.abs(next_state[1])  # + 0.001 * action**2

    return score


def cheetah_cost_fn(state, action, next_state):
    if len(state.shape) > 1:
        heading_penalty_factor = 10
        scores = np.zeros((state.shape[0],))

        # dont move front shin back so far that you tilt forward
        front_leg = state[:, 5]
        my_range = 0.2
        scores[front_leg >= my_range] += heading_penalty_factor

        front_shin = state[:, 6]
        my_range = 0
        scores[front_shin >= my_range] += heading_penalty_factor

        front_foot = state[:, 7]
        my_range = 0
        scores[front_foot >= my_range] += heading_penalty_factor

        scores -= (next_state[:, 17] - state[:, 17]) / 0.01  # + 0.1 * (np.sum(action**2, axis=1))
        return scores

    heading_penalty_factor = 10
    score = 0

    # dont move front shin back so far that you tilt forward
    front_leg = state[5]
    my_range = 0.2
    if front_leg >= my_range:
        score += heading_penalty_factor

    front_shin = state[6]
    my_range = 0
    if front_shin >= my_range:
        score += heading_penalty_factor

    front_foot = state[7]
    my_range = 0
    if front_foot >= my_range:
        score += heading_penalty_factor

    score -= (next_state[17] - state[17]) / 0.01  # + 0.1 * (np.sum(action**2))
    return score


# ========================================================
#
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states):
    trajectory_cost = 0
    for i in range(len(actions)):
        trajectory_cost += cost_fn(states[i], actions[i], next_states[i])
    return trajectory_cost


class GymEnvCostFunc(CostFunc):
    ALLOWED_ENV_ID = ['Reacher-v1', 'HalfCheetah-v1', 'Pendulum-v0']
    ENV_COST_FN_MAP_DICT = \
        {'Reacher-v1': reacher_cost_fn,
         'HalfCheetah-v1': cheetah_cost_fn,
         'Pendulum-v0': pendulum_cost_fn
         }

    def __init__(self, env_id, name='reward_func'):
        super().__init__(name)
        if env_id not in self.ALLOWED_ENV_ID:
            raise ValueError('Not support this env id: {}, only use from {}'.format(env_id, self.ALLOWED_ENV_ID))
        self.cost_fn = self.ENV_COST_FN_MAP_DICT[env_id]

    def __call__(self, state, action, new_state, **kwargs) -> float:
        return self.cost_fn(state=state, action=action, next_state=new_state)
