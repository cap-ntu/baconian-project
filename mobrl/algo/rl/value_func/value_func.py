from mobrl.core.basic import Basic
import typeguard as tg
from mobrl.core.parameters import Parameters
from mobrl.envs.env_spec import EnvSpec
import abc
import tensorflow as tf
from mobrl.tf.tf_parameters import TensorflowParameters
from mobrl.common.util.logger import ConsoleLogger


class ValueFunction(Basic):

    @tg.typechecked
    def __init__(self, env_spec: EnvSpec, parameters: Parameters = None, name='value_func'):
        super().__init__(name)
        self.env_spec = env_spec
        self.parameters = parameters

    @property
    def obs_space(self):
        return self.env_spec.obs_space

    @property
    def action_space(self):
        return self.env_spec.action_space

    @tg.typechecked
    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        return True

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @tg.typechecked
    def update(self, param_update_args: dict, *args, **kwargs):
        self.parameters.update(**param_update_args)

    @abc.abstractmethod
    def init(self, source_obj=None):
        raise NotImplementedError

    @abc.abstractmethod
    def make_copy(self, *args, **kwargs):
        raise NotImplementedError

#
# class PlaceholderInputValueFunction(ValueFunction):
#     # todo do we really need this class?
#     @tg.typechecked
#     def __init__(self, name: str, env_spec: EnvSpec, parameters: TensorflowParameters = None, input: tf.Tensor = None):
#         super().__init__(env_spec=env_spec, parameters=parameters, name=name)
#         self.input = input
#
#     def save(self, save_path, global_step,  name=None, *args, **kwargs):
#         if not name:
#             name = self.name
#         sess = kwargs['sess'] if 'sess' in kwargs else None
#         self.parameters.save(save_path=save_path,
#                              global_step=global_step,
#                              sess=sess,
#                              name=name)
#         ConsoleLogger().print('info',
#                               'model: {}, global step: {}, saved at {}-{}'.format(name, global_step, save_path,
#                                                                                   global_step))
#
#     def load(self, path_to_model, global_step=None, model_name=None, *args, **kwargs):
#         if not model_name:
#             model_name = self.name
#         sess = kwargs['sess'] if 'sess' in kwargs else None
#         self.parameters.load(path_to_model=path_to_model,
#                              model_name=model_name,
#                              global_step=global_step,
#                              sess=sess)
#         ConsoleLogger().print('info', 'model: {} loaded from {}'.format(model_name, path_to_model))
