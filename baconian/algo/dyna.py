from baconian.algo.dynamics.dynamics_model import DynamicsModel
from typeguard import typechecked
from baconian.config.dict_config import DictConfig
from baconian.common.sampler.sample_data import TransitionData
from baconian.core.parameters import Parameters
from baconian.config.global_config import GlobalConfig
from baconian.algo.rl_algo import ModelFreeAlgo, ModelBasedAlgo
from baconian.common.misc import *
from baconian.common.logging import record_return_decorator
from baconian.core.status import register_counter_info_to_status_decorator
from baconian.core.util import init_func_arg_record_decorator
from baconian.algo.misc.placeholder_input import PlaceholderInput
import os


class Dyna(ModelBasedAlgo):
    """
    Dyna algorithms, Sutton, R. S. (1991).
    You can replace the dynamics model with any dynamics models you want.
    """
    required_key_dict = DictConfig.load_json(file_path=GlobalConfig().DEFAULT_ALGO_DYNA_REQUIRED_KEY_LIST)

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self, env_spec, dynamics_model: DynamicsModel,
                 model_free_algo: ModelFreeAlgo,
                 config_or_config_dict: (DictConfig, dict),
                 name='sample_with_dynamics'
                 ):
        super().__init__(env_spec, dynamics_model, name)
        config = construct_dict_config(config_or_config_dict, self)
        parameters = Parameters(parameters=dict(),
                                name='dyna_param',
                                source_config=config)
        sub_placeholder_input_list = []
        if isinstance(dynamics_model, PlaceholderInput):
            sub_placeholder_input_list.append(dict(obj=dynamics_model,
                                                   attr_name='dynamics_model'))
        if isinstance(model_free_algo, PlaceholderInput):
            sub_placeholder_input_list.append(dict(obj=model_free_algo,
                                                   attr_name='model_free_algo'))
        self.model_free_algo = model_free_algo
        self.config = config
        self.parameters = parameters

    @register_counter_info_to_status_decorator(increment=1, info_key='init', under_status='INITED')
    def init(self):
        self.parameters.init()
        self.model_free_algo.init()
        self.dynamics_env.init()
        super().init()

    @record_return_decorator(which_recorder='self')
    @register_counter_info_to_status_decorator(increment=1, info_key='train_counter', under_status='TRAIN')
    def train(self, *args, **kwargs) -> dict:
        super(Dyna, self).train()
        res_dict = {}
        batch_data = kwargs['batch_data'] if 'batch_data' in kwargs else None
        if 'state' in kwargs:
            assert kwargs['state'] in ('state_dynamics_training', 'state_agent_training')
            state = kwargs['state']
            kwargs.pop('state')
        else:
            state = None

        if not state or state == 'state_dynamics_training':

            dynamics_train_res_dict = self._fit_dynamics_model(batch_data=batch_data,
                                                               train_iter=self.parameters('dynamics_model_train_iter'))
            for key, val in dynamics_train_res_dict.items():
                res_dict["{}_{}".format(self._dynamics_model.name, key)] = val
        if not state or state == 'state_agent_training':
            model_free_algo_train_res_dict = self._train_model_free_algo(batch_data=batch_data,
                                                                         train_iter=self.parameters(
                                                                             'model_free_algo_train_iter'))

            for key, val in model_free_algo_train_res_dict.items():
                res_dict['{}_{}'.format(self.model_free_algo.name, key)] = val
        return res_dict

    @register_counter_info_to_status_decorator(increment=1, info_key='test_counter', under_status='TEST')
    def test(self, *arg, **kwargs):
        return super().test(*arg, **kwargs)

    @register_counter_info_to_status_decorator(increment=1, info_key='predict_counter')
    def predict(self, obs, **kwargs):
        return self.model_free_algo.predict(obs)

    def append_to_memory(self, *args, **kwargs):
        self.model_free_algo.append_to_memory(kwargs['samples'])

    @record_return_decorator(which_recorder='self')
    def save(self, global_step, save_path=None, name=None, **kwargs):
        save_path = save_path if save_path else GlobalConfig().DEFAULT_MODEL_CHECKPOINT_PATH
        name = name if name else self.name
        self.model_free_algo.save(global_step=global_step,
                                  name=None,
                                  save_path=os.path.join(save_path, self.model_free_algo.name))
        self.dynamics_env.save(global_step=global_step,
                               name=None,
                               save_path=os.path.join(save_path, self.dynamics_env.name))
        return dict(check_point_save_path=save_path, check_point_save_global_step=global_step,
                    check_point_save_name=name)

    @record_return_decorator(which_recorder='self')
    def load(self, path_to_model, model_name, global_step=None, **kwargs):
        self.model_free_algo.load(path_to_model=os.path.join(path_to_model, self.model_free_algo.name),
                                  model_name=self.model_free_algo.name,
                                  global_step=global_step)
        self.dynamics_env.load(global_step=global_step,
                               path_to_model=os.path.join(path_to_model, self.dynamics_env.name),
                               model_name=self.dynamics_env.name)
        return dict(check_point_load_path=path_to_model, check_point_load_global_step=global_step,
                    check_point_load_name=model_name)

    @register_counter_info_to_status_decorator(increment=1, info_key='dyanmics_train_counter', under_status='TRAIN')
    def _fit_dynamics_model(self, batch_data: TransitionData, train_iter, sess=None) -> dict:
        res_dict = self._dynamics_model.train(batch_data, **dict(sess=sess,
                                                                 train_iter=train_iter))
        return res_dict

    @register_counter_info_to_status_decorator(increment=1, info_key='mode_free_algo_dyanmics_train_counter',
                                               under_status='TRAIN')
    def _train_model_free_algo(self, batch_data=None, train_iter=None, sess=None):
        res_dict = self.model_free_algo.train(**dict(batch_data=batch_data,
                                                     train_iter=train_iter,
                                                     sess=sess))
        return res_dict
