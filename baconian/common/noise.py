"""
From openai baselines
"""
import numpy as np
from typeguard import typechecked
from baconian.common.schedules import Scheduler


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0


class NormalActionNoise(ActionNoise):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class UniformNoise(ActionNoise):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self):
        uniformNoiseValue = self.scale * (np.random.rand() - 0.5)
        return uniformNoiseValue


class OUNoise(ActionNoise):
    def __init__(self, theta=0.05, sigma=0.25, init_state=0.0):
        self.theta = theta
        self.sigma = sigma
        self.state = init_state

    def __call__(self):
        state = self.state - self.theta * self.state + self.sigma * np.random.randn()
        self.state = state
        return self.state

    def reset(self):
        self.state = 0.0


class AgentActionNoiseWrapper(object):
    INJECT_TYPE = ['WEIGHTED_DECAY', '']

    @typechecked
    def __init__(self, noise: ActionNoise, action_weight_scheduler: Scheduler, noise_weight_scheduler: Scheduler):
        self.action_weight_scheduler = action_weight_scheduler
        self.noise_weight_scheduler = noise_weight_scheduler
        self.noise = noise

    def __call__(self, action, **kwargs):
        noise = self.noise()
        return self.action_weight_scheduler.value() * action + self.noise_weight_scheduler.value() * noise

    def reset(self):
        self.noise.reset()
