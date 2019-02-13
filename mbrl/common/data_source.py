from mbrl.core.basic import Basic
import abc
from mbrl.common.sampler.sampler import Sampler
from mbrl.config.global_config import GlobalConfig


class DataSource(Basic):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_data(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError


class DataSourceBySample(DataSource):
    def __init__(self):
        super().__init__()
        self.sampler = Sampler()

    def get_batch_data(self, batch_size, shuffle_flag=False):
        pass

    def get_path_data(self, path_num, shuffle_flag=False):
        pass

    def sample(self, agent, env, sample_type, sample_count, reset_at_start, in_test_flag):
        if sample_type == GlobalConfig.SAMPLE_TYPE_SAMPLE_TRANSITION_DATA:
            sample_record = self.sampler.sample(
                env=env,
                agent=agent,
                sample_type='transition',
                in_test_flag=in_test_flag,
                sample_count=sample_count,
                reset_at_start=reset_at_start
            )
            return sample_record
        elif sample_type == GlobalConfig.SAMPLE_TYPE_SAMPLE_TRAJECTORY_DATA:
            sample_record = self.sampler.sample(
                env=env,
                agent=agent,
                sample_type='trajectory',
                in_test_flag=in_test_flag,
                sample_count=sample_count,
                reset_at_start=reset_at_start
            )
            return sample_record
