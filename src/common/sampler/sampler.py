from src.core.basic import Basic
from src.common.sampler.sample_data import TransitionData, TrajectoryData
from src.core.global_config import GlobalConfig
from typeguard import typechecked
from src.envs.env import Env
from src.envs.env_spec import EnvSpec


class Sampler(Basic):
    def __init__(self, env_spec):
        super().__init__()
        self._data = TransitionData(env_spec)
        self.env_spec = env_spec

    def init(self):
        self._data.reset()

    @typechecked
    def sample(self, env: GlobalConfig.DEFAULT_ALLOWED_GYM_ENV_TYPE + (Env,),
               agent,
               in_test_flag: bool,
               sample_count: int,
               sample_type='transition',

               reset_at_start=False) -> (TransitionData, TrajectoryData):
        if reset_at_start is True:
            state = env.reset()
        else:
            state = env.get_state()
        if sample_type == 'transition':
            return self._sample_transitions(env, agent, sample_count, state, in_test_flag)
        elif sample_type == 'trajectory':
            return self._sample_trajectories(env, agent, sample_count, state, in_test_flag)
        else:
            raise ValueError()

    def _sample_transitions(self, env: Env, agent, sample_count, init_state, in_test_flag):
        state = init_state
        sample_record = TransitionData(env_spec=self.env_spec)

        for i in range(sample_count):
            action = agent.predict(obs=state, in_test_flag=in_test_flag)
            new_state, re, done, info = env.step(action)
            if not isinstance(done, bool):
                if done[0] == 1:
                    done = True
                else:
                    done = False

            sample_record.append(state=state,
                                 action=action,
                                 reward=re,
                                 new_state=new_state,
                                 done=done)
            state = new_state
        return sample_record

    def _sample_trajectories(self, env, agent, sample_count, init_state, in_test_flag):
        state = init_state
        sample_record = TrajectoryData(self.env_spec)
        done = False
        for i in range(sample_count):
            traj_record = TransitionData(self.env_spec)
            while done is not True:
                action = agent.predict(obs=state, in_test_flag=in_test_flag)
                new_state, re, done, info = env.step(action)
                # todo done signal should be bool, which should be strict at the env codes
                if not isinstance(done, bool):
                    if done[0] == 1:
                        done = True
                    else:
                        done = False

                traj_record.append(state=state,
                                   action=action,
                                   reward=re,
                                   new_state=new_state,
                                   done=done)
                state = new_state
            sample_record.append(traj_record)
        return sample_record
