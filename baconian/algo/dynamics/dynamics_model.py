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
from copy import deepcopy


class DynamicsModel(Basic):
    STATUS_LIST = ('NOT_INIT', 'JUST_INITED')
    INIT_STATUS = 'NOT_INIT'

    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, init_state=None, name='dynamics_model'):
        """
        :param env_spec:
        :param parameters:
        :param init_state:
        :param name:
        """
        super().__init__(name=name)
        self.env_spec = env_spec
        self.state = init_state
        self.parameters = parameters
        self.state_input = None
        self.action_input = None
        self.new_state_output = None
        self.recorder = Recorder(flush_by_split_status=False)
        self._status = StatusWithSingleInfo(obj=self)

    def init(self, *args, **kwargs):
        self.set_status('JUST_INITED')
        self.state = self.env_spec.obs_space.sample()

    @register_counter_info_to_status_decorator(increment=1, info_key='step_counter')
    def step(self, action: np.ndarray, state=None, allow_clip=False, **kwargs_for_transit):
        state = np.array(state).reshape(self.env_spec.obs_shape) if state is not None else self.state
        action = action.reshape(self.env_spec.action_shape)
        if allow_clip is True:
            if state is not None:
                if self.env_spec.obs_space.contains(state) is False:
                    # todo log level seems not working
                    # ConsoleLogger().print('warning', 'state out of bound, allowed clipping')
                    state = self.env_spec.obs_space.clip(state)
                    assert self.env_spec.obs_space.contains(state)
            if self.env_spec.action_space.contains(action) is False:
                # ConsoleLogger().print('warning', 'action out of bound, allowed clipping')
                action = self.env_spec.action_space.clip(action)

        assert self.env_spec.action_space.contains(action)
        assert self.env_spec.obs_space.contains(state)
        new_state = self._state_transit(state=state, action=self.env_spec.flat_action(action),
                                        **kwargs_for_transit)
        if allow_clip is True:
            # ConsoleLogger().print('warning', 'new state out of bound, allowed clipping')
            new_state = self.env_spec.obs_space.clip(new_state)
        if self.env_spec.obs_space.contains(new_state) is False:
            raise DynamicsNextStepOutputBoundError(
                'new state {} out of bound of {}'.format(new_state, self.env_spec.obs_space.bound()))
        self.state = new_state
        return new_state

    @abc.abstractmethod
    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True

    def make_copy(self):
        raise NotImplementedError

    def reset_state(self, state=None):
        if state is not None:
            assert self.env_spec.obs_space.contains(state)
            self.state = state
        else:
            self.state = self.env_spec.obs_space.sample()

    def return_as_env(self) -> Env:
        return DynamicsEnvWrapper(dynamics=self,
                                  name=self._name + '_env')


class LocalDyanmicsModel(DynamicsModel):
    pass


class GlobalDynamicsModel(DynamicsModel):
    pass


class TrainableDyanmicsModel(DynamicsModel):
    def train(self, *args, **kwargs):
        raise NotImplementedError


class DerivableDynamics(object):
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
        # self.grad_all = [self.output_node_list, 0, 0]
        # self.grad_all[1] = [tf.test.compute_gradient(y=y_node,
        #                                              y_shape=y_node.shape.as_list(),
        #                                              x_shape=[node.shape.as_list() for node in
        #                                                       list(self.input_node_dict.values())],
        #                                              x=list(self.input_node_dict.values())) for y_node in
        #                     self.output_node_list]
        # self.grad_all[2] = tf.hessians(self.output_node_list, list(self.input_node_dict.values()))
        self._grad_dict = [{}, {}, {}]
        for val in input_node_dict:
            self._grad_dict[0][val] = self.output_node_list

    @typechecked
    def grad_on_input_(self, key_or_node: (str, tf.Tensor), order=1, batch_flag=False):
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

    # def get_jacobian(self, y_node_list, x_node_list):
    #     pass
    #
    # def get_hessians(self):
    #     pass
    #
    # def _map_to_expr_nodes(self, input_node_list, output_node_list, func):
    #     grad_op = []
    #     for i_node in input_node_list:
    #         for o_node in output_node_list:
    #             grad_op.append(self._map_to_expr_node(input_node=i_node, output_node=o_node, func=func))
    #
    # def _map_to_expr_node(self, input_node, output_node, func):
    #     return func(output=output_node, input=input_node)


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

    def step(self, action: np.ndarray, **kwargs):
        super().step(action)
        state = deepcopy(self.get_state()) if 'state' not in kwargs else kwargs['state']
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
