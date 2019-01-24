from src.core.basic import Basic
from src.common.sampler.sample_data import TransitionData
from src.envs.env import Env
from src.core.global_config import GlobalConfig
from typeguard import typechecked


class Sampler(Basic):
    def __init__(self):
        super().__init__()
        self._data = TransitionData()
        self.step_count_per_episode = 0

    def init(self):
        self._data.reset()

    @typechecked
    def sample(self, env: GlobalConfig.DEFAULT_ALLOWED_GYM_ENV_TYPE + (Env,),
               agent,
               in_test_flag: bool,
               sample_count: int,
               reset_at_start=False) -> TransitionData:
        if reset_at_start is True:
            state = env.reset()
        else:
            state = env.get_state()
        sample_record = TransitionData()
        for i in range(sample_count):
            action = agent.predict(obs=state, in_test_flag=in_test_flag)
            new_state, re, done, info = env.step(action)
            if not isinstance(done, bool):
                if done[0] == 1:
                    done = True
                else:
                    done = False
            self.step_count_per_episode += 1

            sample_record.append(state=state,
                                 action=action,
                                 reward=re,
                                 new_state=new_state,
                                 done=done)
            state = new_state
            if done is True:
                self.step_count_per_episode = 0
        return sample_record
