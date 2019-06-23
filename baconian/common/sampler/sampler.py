from baconian.core.core import Basic, Env
from baconian.common.sampler.sample_data import TransitionData, TrajectoryData
from typeguard import typechecked


class Sampler(Basic):
    def __init__(self, env_spec, name='sampler'):
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
               reset_at_start=False) -> (TransitionData, TrajectoryData):
        self.set_status(in_which_status)
        if reset_at_start is True:
            state = env.reset()
        else:
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
                new_state, re, done, info = env.step(action)
                if not isinstance(done, bool):
                    raise TypeError()
                traj_record.append(state=state,
                                   action=action,
                                   reward=re,
                                   new_state=new_state,
                                   done=done)
                state = new_state
            state = env.reset()
            sample_record.append(traj_record)
        return sample_record
