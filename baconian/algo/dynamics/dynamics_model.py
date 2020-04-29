import tensorflow as tf
from baconian.core.core import Basic, EnvSpec
import numpy as np
import abc
from baconian.core.parameters import Parameters
from typeguard import typechecked
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian as tf_batch_jacobian
from baconian.common.logging import Recorder
from baconian.core.status import register_counter_info_to_status_decorator, StatusWithSingleInfo
from baconian.common.logging import ConsoleLogger
from baconian.common.error import *
from baconian.core.core import EnvSpec, Env
from baconian.algo.dynamics.reward_func.reward_func import RewardFunc
from baconian.algo.dynamics.terminal_func.terminal_func import TerminalFunc
from baconian.common.data_pre_processing import DataScaler, IdenticalDataScaler


class DynamicsModel(Basic):
    STATUS_LIST = ('CREATED', 'INITED')
    INIT_STATUS = 'CREATED'

    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, init_state=None, name='dynamics_model',
                 state_input_scaler: DataScaler = None,
                 action_input_scaler: DataScaler = None,
                 state_output_scaler: DataScaler = None):

        """

        :param env_spec: environment specifications, such as observation space and action space
        :type env_spec: EnvSpec
        :param parameters: parameters
        :type parameters: Parameters
        :param init_state: initial state of dymamics model
        :type init_state: str
        :param name: name of instance, 'dynamics_model' by default
        :type name: str
        :param state_input_scaler: data preprocessing scaler of state input
        :type state_input_scaler: DataScaler
        :param action_input_scaler: data preprocessing scaler of action input
        :type action_input_scaler: DataScaler
        :param state_output_scaler: data preprocessing scaler of state output
        :type state_output_scaler: DataScaler
        """
        super().__init__(name=name)
        self.env_spec = env_spec
        self.state = init_state
        self.parameters = parameters
        self.state_input = None
        self.action_input = None
        self.new_state_output = None
        self.recorder = Recorder(flush_by_split_status=False, default_obj=self)
        self._status = StatusWithSingleInfo(obj=self)
        self.state_input_scaler = state_input_scaler if state_input_scaler else IdenticalDataScaler(
            dims=env_spec.flat_obs_dim)
        self.action_input_scaler = action_input_scaler if action_input_scaler else IdenticalDataScaler(
            dims=env_spec.flat_action_dim)
        self.state_output_scaler = state_output_scaler if state_output_scaler else IdenticalDataScaler(
            dims=env_spec.flat_obs_dim)

    def init(self, *args, **kwargs):
        self.set_status('INITED')
        self.state = self.env_spec.obs_space.sample()

    @register_counter_info_to_status_decorator(increment=1, info_key='step_counter')
    def step(self, action: np.ndarray, state=None, allow_clip=False, **kwargs_for_transit):

        """
        State transition function (only support one sample transition instead of batch data)

        :param action: action to be taken
        :type action: np.ndarray
        :param state: current state, if None, will use stored state (saved from last transition)
        :type state: np.ndarray
        :param allow_clip: allow clip of observation space, default False
        :type allow_clip: bool
        :param kwargs_for_transit: extra kwargs for calling the _state_transit, this is typically related to the
                                    specific mode you used
        :type kwargs_for_transit:
        :return: new state after step
        :rtype: np.ndarray
        """
        state = np.array(state).reshape(self.env_spec.obs_shape) if state is not None else self.state
        action = action.reshape(self.env_spec.action_shape)
        if allow_clip is True:
            if state is not None:
                    state = self.env_spec.obs_space.clip(state)
            action = self.env_spec.action_space.clip(action)
        if self.env_spec.action_space.contains(action) is False:
            raise StateOrActionOutOfBoundError(
                'action {} out of bound of {}'.format(action, self.env_spec.action_space.bound()))
        if self.env_spec.obs_space.contains(state) is False:
            raise StateOrActionOutOfBoundError(
                'state {} out of bound of {}'.format(state, self.env_spec.obs_space.bound()))
        new_state = self._state_transit(state=state, action=self.env_spec.flat_action(action),
                                        **kwargs_for_transit)
        if allow_clip is True:
            new_state = self.env_spec.obs_space.clip(new_state)
        if self.env_spec.obs_space.contains(new_state) is False:
            raise StateOrActionOutOfBoundError(
                'new state {} out of bound of {}'.format(new_state, self.env_spec.obs_space.bound()))
        self.state = new_state
        return new_state

    @abc.abstractmethod
    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        """

        :param state: original state
        :type state: np.ndarray
        :param action:  action taken by agent
        :type action: np.ndarray
        :param kwargs:
        :type kwargs:
        :return: new state after transition
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def copy_from(self, obj) -> bool:
        """

        :param obj: object to copy from
        :type obj:
        :return: True if successful else raise an error
        :rtype: bool
        """
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True

    def make_copy(self):
        """ Make a copy of parameters and environment specifications."""
        raise NotImplementedError

    def reset_state(self, state=None):
        """

        :param state: original state
        :type state: np.ndarray
        :return: a random sample space in observation space
        :rtype: np.ndarray
        """
        if state is not None:
            assert self.env_spec.obs_space.contains(state)
            self.state = state
        else:
            self.state = self.env_spec.obs_space.sample()

    def return_as_env(self) -> Env:
        """

        :return: an environment with this dynamics model
        :rtype: DynamicsEnvWrapper
        """
        return DynamicsEnvWrapper(dynamics=self,
                                  name=self._name + '_env')


class LocalDyanmicsModel(DynamicsModel):
    pass


class GlobalDynamicsModel(DynamicsModel):
    pass


class TrainableDyanmicsModel(object):
    def train(self, *args, **kwargs):
        raise NotImplementedError


class DifferentiableDynamics(object):
    @typechecked
    def __init__(self, input_node_dict: dict, output_node_dict: dict):
        for node in input_node_dict.values():
            if not isinstance(node, tf.Tensor):
                raise TypeError('Derivable only support tf.Tensor as node')
        for node in output_node_dict.values():
            if not isinstance(node, tf.Tensor):
                raise TypeError('Derivable only support tf.Tensor as node')
        self.input_node_dict = input_node_dict
        self.output_node_dict = output_node_dict
        self.output_node_list = []
        for key in output_node_dict.keys():
            self.output_node_list.append(output_node_dict[key])
        self._grad_dict = [{}, {}, {}]
        for val in input_node_dict:
            self._grad_dict[0][val] = self.output_node_list

    def grad_on_input(self, key_or_node: (str, tf.Tensor), order=1, batch_flag=False):
        if batch_flag:
            raise NotImplementedError
        node = key_or_node if isinstance(key_or_node, tf.Tensor) else self.input_node_dict[key_or_node]
        if node not in self._grad_dict:
            if order == 1:
                grad_op = [tf_batch_jacobian(output=o_node, inp=node) for o_node in self.output_node_list]
            else:
                grad_op = [self.split_and_hessian(out_node=o_node, innode=node) for o_node in self.output_node_list]
            self._grad_dict[order][node] = grad_op
            return grad_op
        else:
            return self._grad_dict[order][node]

    def split_and_hessian(self, out_node, innode):
        out_nodes = tf.split(out_node, 1, axis=1)
        hessian_node = []
        for o_node in out_nodes:
            hessian_node.append(tf.stack(tf.hessians(o_node, innode)))
        new_dim = len(hessian_node[0].shape.as_list()) + 1
        new_dim = list(range(new_dim))
        new_dim[0] = 1
        new_dim[1] = 0
        return tf.transpose(tf.stack(hessian_node), perm=new_dim)


class DynamicsEnvWrapper(Env):
    """
    A wrapper that wrap the dynamics into a standard baconian env
    """

    @typechecked
    def __init__(self, dynamics: DynamicsModel, name: str = 'dynamics_env'):
        super().__init__(name)
        self._dynamics = dynamics
        self._reward_func = None
        self._terminal_func = None
        self.env_spec = dynamics.env_spec

    def step(self, action: np.ndarray, **kwargs):
        super().step(action)
        state = self.get_state() if 'state' not in kwargs else kwargs['state']
        new_state = self._dynamics.step(action=action, **kwargs)
        re = self._reward_func(state=state, new_state=new_state, action=action)
        terminal = self._terminal_func(state=state, action=action, new_state=new_state)
        return new_state, re, terminal, ()

    def reset(self):
        super(DynamicsEnvWrapper, self).reset()
        self._dynamics.reset_state()
        return self.get_state()

    def init(self):
        super().init()
        self._dynamics.init()

    def get_state(self):
        return self._dynamics.state

    def seed(self, seed=None):
        ConsoleLogger().print('warning', 'seed on dynamics model has no effect ')
        pass

    def save(self, *args, **kwargs):
        return self._dynamics.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self._dynamics.load(*args, **kwargs)

    def set_terminal_reward_func(self, terminal_func: TerminalFunc, reward_func: RewardFunc):
        self._terminal_func = terminal_func
        self._reward_func = reward_func


class DynamicsPriorModel(Basic):
    def __init__(self, env_spec: EnvSpec, parameters: Parameters, name: str):
        super().__init__(name=name)
        self.env_spec = env_spec
        self.parameters = parameters
