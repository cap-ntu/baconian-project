import tensorflow as tf
from baconian.core.core import Basic, EnvSpec
import numpy as np
import abc
from baconian.core.parameters import Parameters
from typeguard import typechecked
from tensorflow.python.ops.parallel_for.gradients import jacobian as tf_jacobian, batch_jacobian as tf_batch_jacobian


class DynamicsModel(Basic):
    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, init_state=None, name='dynamics_model'):
        super().__init__(name=name)
        self.env_space = env_spec
        self.state = init_state
        self.parameters = parameters
        self.state_input = None
        self.action_input = None
        self.new_state_output = None

    def init(self):
        raise NotImplementedError

    def step(self, action: np.ndarray, state=None, **kwargs_for_transit):
        state = state if state is not None else self.state
        assert self.env_space.action_space.contains(action)
        assert self.env_space.obs_space.contains(state)
        new_state = self._state_transit(state=state, action=self.env_space.flat_action(action),
                                        **kwargs_for_transit)
        assert self.env_space.obs_space.contains(new_state)
        self.state = new_state
        return new_state

    @abc.abstractmethod
    def _state_transit(self, state, action, **kwargs) -> np.ndarray:
        raise NotImplementedError


class LocalDyanmicsModel(DynamicsModel):
    pass


class GlobalDynamicsModel(DynamicsModel):
    pass


class DerivableDynamics(object):
    @typechecked
    # todo add support for numpy with autograd later
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
