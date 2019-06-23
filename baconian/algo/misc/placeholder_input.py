import tensorflow as tf
import typeguard as tg
import os

from baconian.common.logging import ConsoleLogger
from baconian.tf.tf_parameters import ParametersWithTensorflowVariable
from baconian.core.core import Basic
from baconian.config.global_config import GlobalConfig


class PlaceholderInput(object):
    @tg.typechecked
    def __init__(self, parameters: ParametersWithTensorflowVariable = None,
                 name_scope=None):
        self.parameters = parameters
        if name_scope:
            self.name_scope = name_scope

    def save(self, global_step, save_path=None, name=None, **kwargs):
        save_path = save_path if save_path else GlobalConfig().DEFAULT_MODEL_CHECKPOINT_PATH
        name = name if name else self.name
        sess = kwargs['sess'] if 'sess' in kwargs else None
        self.parameters.save(save_path=save_path,
                             global_step=global_step,
                             sess=sess,
                             name=name)
        ConsoleLogger().print('info',
                              'model: {}, global step: {}, saved at {}-{}'.format(name, global_step, save_path,
                                                                                  global_step))

    def load(self, path_to_model, model_name, global_step=None, **kwargs):
        sess = kwargs['sess'] if 'sess' in kwargs else None
        self.parameters.load(path_to_model=path_to_model,
                             model_name=model_name,
                             global_step=global_step,
                             sess=sess)
        ConsoleLogger().print('info', 'model: {} loaded from {}'.format(model_name, path_to_model))

    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        self.parameters.copy_from(source_parameter=obj.parameters)
        return True


class MultiPlaceholderInput(object):
    @tg.typechecked
    def __init__(self, sub_placeholder_input_list: list,
                 parameters: ParametersWithTensorflowVariable):
        self._placeholder_input_list = sub_placeholder_input_list
        for param in self._placeholder_input_list:
            assert isinstance(param, dict)
            assert 'attr_name' in param
            assert 'obj' in param and isinstance(param['obj'], PlaceholderInput) and isinstance(param['obj'], Basic)

        self._own_placeholder_input_obj = PlaceholderInput(parameters=parameters)

    def save(self, global_step, save_path, name, **kwargs):
        sess = kwargs['sess'] if 'sess' in kwargs else None
        self._own_placeholder_input_obj.parameters.save(save_path=save_path,
                                                        global_step=global_step,
                                                        sess=sess,
                                                        name=name)
        for param in self._placeholder_input_list:
            param['obj'].save(save_path=os.path.join(save_path, param['attr_name']),
                              global_step=global_step,
                              sess=sess,
                              name=param['obj'].name)
        ConsoleLogger().print('info',
                              'model: {}, global step: {}, saved at {}-{}'.format(name, global_step, save_path,
                                                                                  global_step))

    def load(self, path_to_model, model_name, global_step=None, **kwargs):
        sess = kwargs['sess'] if 'sess' in kwargs else None
        self._own_placeholder_input_obj.parameters.load(
            path_to_model=path_to_model,
            model_name=model_name,
            global_step=global_step,
            sess=sess
        )
        for param in self._placeholder_input_list:
            param['obj'].load(path_to_model=os.path.join(path_to_model, param['attr_name']),
                              global_step=global_step,
                              model_name=param['obj'].name,
                              sess=sess)

        ConsoleLogger().print('info', 'model: {} loaded from {}'.format(model_name, path_to_model))

    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        self._own_placeholder_input_obj.copy_from(obj._own_placeholder_input_obj)
        for self_param, src_param in zip(self._placeholder_input_list, obj._placeholder_input_list):
            self_param['obj'].copy_from(src_param['obj'])
        ConsoleLogger().print('info', 'model: {} copyed from {}'.format(self, obj))
        return True
