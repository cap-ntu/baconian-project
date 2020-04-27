from baconian.core.core import Basic, Env
from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from typeguard import typechecked


class Sampler(Basic):
    """
    Sampler module that handle the sampling procedure for training/testing of the agent.
    """

    def __init__(self, env_spec, name='sampler'):
        """
        Constructor
        :param env_spec: env_spec type, that indicate what is the environment spec that the sampler will work on.
        :param name: a string.
        """
        super().__init__(name)
        self._data = TransitionData(env_spec)
        self.env_spec = env_spec

    def init(self):
        self._data.reset()

    @typechecked
    def sample(self, env: Env,
               agent,
               in_which_status: str,
               sample_count: int,
               sample_type='transition',
               reset_at_start=None) -> (TransitionData, TrajectoryData):
        """
        sample function

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

        :return: SampleData object or TrajectoryData object based on passed in sample_type.
        """
        self.set_status(in_which_status)
        state = None
        if reset_at_start is True or (reset_at_start is None and sample_type == 'trajectory'):
            state = env.reset()
        elif reset_at_start is False or (reset_at_start is None and sample_type == 'transition'):
            state = env.get_state()
        if sample_type == 'transition':
            return self._sample_transitions(env, agent, sample_count, state)
        elif sample_type == 'trajectory':
            return self._sample_trajectories(env, agent, sample_count, state)
        else:
            raise ValueError()

    def _sample_transitions(self, env: Env, agent, sample_count, init_state):
        state = init_state
        sample_record = TransitionData(env_spec=self.env_spec)

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

    def _sample_trajectories(self, env, agent, sample_count, init_state):
        state = init_state
        sample_record = TrajectoryData(self.env_spec)
        done = False
        for i in range(sample_count):
            traj_record = TransitionData(self.env_spec)
            while done is not True:
                action = agent.predict(obs=state)
                new_state, re, done, info = env.step(action.squeeze())
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
