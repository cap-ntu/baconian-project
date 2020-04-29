from baconian.core.core import Basic, Env
from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from typeguard import typechecked
import numpy as np


class Sampler(object):
    """
    Sampler module that handle the sampling procedure for training/testing of the agent.
    """

    @staticmethod
    @typechecked
    def sample(env: Env,
               agent,
               sample_count: int,
               sample_type='transition',
               reset_at_start=None) -> (TransitionData, TrajectoryData):
        """
        a static method of sample function

        :param env: environment object to sample from.
        :param agent: agent object to offer the sampling policy
        :param in_which_status: a string, "TEST" or "TRAIN" indicate this sample is used for training or testing
                                            (evaluation)

        :param sample_count: number of samples. If the sample_type == "transition", then this value means the number of
                                    transitions, usually for off-policy method like DQN, DDPG. If the sample_type ==
                                    "trajectory", then this value means the numbers of trajectories.

        :param sample_type: a string, "transition" or "trajectory".
        :param reset_at_start: A bool, if True, will reset the environment at the beginning, if False, continue sampling
                                    based on previous state (this is useful for certain tasks that you need to preserve
                                    previous state to reach the terminal goal state). If None, for sample_type ==
                                    "transition", it will set to False, for sample_type == "trajectory", it will set to True.

        :return: SampleData object.
        """
        state = None
        if reset_at_start is True or (reset_at_start is None and sample_type == 'trajectory'):
            state = env.reset()
        elif reset_at_start is False or (reset_at_start is None and sample_type == 'transition'):
            state = env.get_state()
        if sample_type == 'transition':
            return Sampler._sample_transitions(env, agent, sample_count, state)
        elif sample_type == 'trajectory':
            return Sampler._sample_trajectories(env, agent, sample_count, state)
        else:
            raise ValueError()

    @staticmethod
    def _sample_transitions(env: Env, agent, sample_count, init_state):
        state = init_state
        sample_record = TransitionData(env_spec=env.env_spec)

        for i in range(sample_count):
            action = agent.predict(obs=state)
            new_state, re, done, info = env.step(action)
            if not isinstance(done, bool):
                raise TypeError()
            sample_record.append(state=state,
                                 action=action,
                                 reward=re,
                                 new_state=new_state,
                                 done=done)
            if done:
                state = env.reset()
                agent.reset_on_terminal_state()
            else:
                state = new_state
        return sample_record

    @staticmethod
    def _sample_trajectories(env, agent, sample_count, init_state):
        state = init_state
        sample_record = TrajectoryData(env.env_spec)
        done = False
        for i in range(sample_count):
            traj_record = TransitionData(env.env_spec)
            while done is not True:
                action = agent.predict(obs=state)
                new_state, re, done, info = env.step(action)
                if not isinstance(done, bool):
                    raise TypeError()
                traj_record.append(state=state,
                                   action=action,
                                   reward=re,
                                   new_state=new_state,
                                   done=done)
                state = new_state
            agent.reset_on_terminal_state()
            done = False
            state = env.reset()
            sample_record.append(traj_record)
        return sample_record
