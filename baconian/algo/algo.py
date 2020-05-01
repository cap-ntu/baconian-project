from baconian.core.core import Basic, EnvSpec, Env
from baconian.core.status import StatusWithSubInfo
import abc
from typeguard import typechecked
from baconian.common.logging import Recorder
from baconian.core.parameters import Parameters
from baconian.common.sampler.sample_data import TrajectoryData


class Algo(Basic):
    """
    Abstract class for algorithms
    """
    STATUS_LIST = ['CREATED', 'INITED', 'TRAIN', 'TEST']
    INIT_STATUS = 'CREATED'

    @typechecked
    def __init__(self, env_spec: EnvSpec, name: str = 'algo', warm_up_trajectories_number=0):
        """
        Constructor

        :param env_spec: environment specifications
        :type env_spec: EnvSpec
        :param name: name of the algorithm
        :type name: str
        :param warm_up_trajectories_number: how many trajectories used to warm up the training
        :type warm_up_trajectories_number: int
        """

        super().__init__(status=StatusWithSubInfo(obj=self), name=name)
        self.env_spec = env_spec
        self.parameters = Parameters(dict())
        self.recorder = Recorder(default_obj=self)
        self.warm_up_trajectories_number = warm_up_trajectories_number

    def init(self):
        """
        Initialization method, such as network random initialization in Tensorflow

        :return:
        """
        self._status.set_status('INITED')

    def warm_up(self, trajectory_data: TrajectoryData):
        """
        Use some data to warm up the algorithm, e.g., compute the mean/std-dev of the state to perform normalization.
        Data used in warm up process will not be added into the memory
        :param trajectory_data: TrajectoryData object
        :type trajectory_data: TrajectoryData

        :return: None
        """
        pass

    def train(self, *arg, **kwargs) -> dict:
        """
        Training API, specific arguments should be defined by each algorithms itself.

        :return: training results, e.g., loss
        :rtype: dict
        """

        self._status.set_status('TRAIN')
        return dict()

    def test(self, *arg, **kwargs) -> dict:
        """
        Testing API, most of the evaluation can be done by agent instead of algorithms, so this API can be skipped

        :return: test results, e.g., rewards
        :rtype: dict
        """

        self._status.set_status('TEST')
        return dict()

    @abc.abstractmethod
    def predict(self, *arg, **kwargs):
        """
        Predict function, given the obs as input, return the action, obs will be read as the first argument passed into
        this API, like algo.predict(obs=x, ...)

        :return: predicted action
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @abc.abstractmethod
    def append_to_memory(self, *args, **kwargs):
        """
        For off-policy algorithm, use this API to append the data into replay buffer. samples will be read as the first
        argument passed into this API, like algo.append_to_memory(samples=x, ...)

        """
        raise NotImplementedError

    @property
    def is_training(self):
        """
        A boolean indicate the if the algorithm is in training status

        :return: True if in training
        :rtype: bool
        """
        return self.get_status()['status'] == 'TRAIN'

    @property
    def is_testing(self):
        """
        A boolean indicate the if the algorithm is in training status

        :return: True if in testing
        :rtype: bool
        """
        return self.get_status()['status'] == 'TEST'
