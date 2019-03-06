from baconian.algo.rl.model_based.models.dynamics_model import DynamicsModel
from typeguard import typechecked
from baconian.config.dict_config import DictConfig
from baconian.common.sampler.sample_data import TransitionData
from baconian.tf.tf_parameters import TensorflowParameters
from baconian.config.global_config import GlobalConfig
from baconian.algo.rl.rl_algo import ModelFreeAlgo, ModelBasedAlgo
from baconian.common.misc import *
from baconian.common.logging import record_return_decorator
from baconian.core.status import register_counter_info_to_status_decorator
from baconian.core.util import init_func_arg_record_decorator
from baconian.algo.placeholder_input import MultiPlaceholderInput, PlaceholderInput
from baconian.core.core import Env
from baconian.common.spaces.box import Box
import numpy as np
import types
from baconian.envs.gym_env import GymEnv


class TrainingEnv(Env):
    # todo env wrapper can work here
    # A training env tha encapsulate the sample with dynamics training flow
    def __init__(self, config_or_config_dict: (DictConfig, dict),
                 name='env'):
        super().__init__(name)
        # action space = [sampling ratio of real / dynamics, training time ratio of real / dynamics]
        self.action_space = Box(low=[0.0, 0.0], high=[1.0, 1.0])
        # obs return the
        self.observation_space = Box(low=[-np.inf], high=[np.inf])
        self.observation_space.low = np.nan_to_num(self.observation_space.low)
        self.observation_space.high = np.nan_to_num(self.observation_space.high)
        self.observation_space.sample = types.MethodType(GymEnv._sample_with_nan, self.observation_space)
        self.config = construct_dict_config(config_or_config_dict, self)

    def step(self, action):
        super().step(action)

    def reset(self):
        return super().reset()

    def init(self):
        return super().init()


class SampleWithDynamics(ModelBasedAlgo, MultiPlaceholderInput):
    """
    The naive model based method by approximating a dynamics with nn and sample from it to train the agent.
    """
    required_key_dict = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_ALGO_SAMPLE_WITH_DYNAMICS_REQUIRED_KEY_LIST)

    @init_func_arg_record_decorator()
    @typechecked
    def __init__(self, env_spec, dynamics_model: DynamicsModel,
                 model_free_algo: ModelFreeAlgo,
                 config_or_config_dict: (DictConfig, dict),
                 name='sample_with_dynamics'
                 ):
        super().__init__(env_spec, dynamics_model, name)
        config = construct_dict_config(config_or_config_dict, self)
        parameters = TensorflowParameters(tf_var_list=[],
                                          rest_parameters=dict(),
                                          name='sample_with_model_param',
                                          source_config=config,
                                          require_snapshot=False)
        sub_placeholder_input_list = []
        if isinstance(dynamics_model, PlaceholderInput):
            sub_placeholder_input_list.append(dict(obj=dynamics_model,
                                                   attr_name='dynamics_model'))
        if isinstance(model_free_algo, PlaceholderInput):
            sub_placeholder_input_list.append(dict(obj=model_free_algo,
                                                   attr_name='model_free_algo'))
        MultiPlaceholderInput.__init__(self,
                                       sub_placeholder_input_list=sub_placeholder_input_list,
                                       inputs=(),
                                       parameters=parameters)
        self.model_free_algo = model_free_algo
        self.config = config
        self.parameters = parameters

    @register_counter_info_to_status_decorator(increment=1, info_key='init', under_status='JUST_INITED')
    def init(self):
        super().init()
        self.model_free_algo.init()
        self.dynamics_model.init()

    @record_return_decorator(which_recorder='self')
    @register_counter_info_to_status_decorator(increment=1, info_key='train_counter', under_status='TRAIN')
    def train(self, **kwargs) -> dict:
        super(SampleWithDynamics, self).train()
        res_dict = {}
        batch_data = kwargs['batch_data'] if 'batch_data' in kwargs else None
        if 'state' not in kwargs or ('state' in kwargs and kwargs['state'] == 'state_dynamics_training'):

            dynamics_train_res_dict = self._fit_dynamics_model(batch_data=batch_data,
                                                               train_iter=self.parameters('dynamics_model_train_iter'))
            for key, val in dynamics_train_res_dict.items():
                res_dict["mlp_dynamics_{}".format(key)] = val
        if 'state' not in kwargs or ('state' in kwargs and kwargs['state'] == 'state_agent_training'):
            model_free_algo_train_res_dict = self._train_model_free_algo(batch_data=batch_data,
                                                                         train_iter=self.parameters(
                                                                             'model_free_algo_train_iter'))

            for key, val in model_free_algo_train_res_dict.items():
                res_dict['dqn_{}'.format(key)] = val
        return res_dict

    @register_counter_info_to_status_decorator(increment=1, info_key='test_counter', under_status='TEST')
    def test(self, *arg, **kwargs):
        super().test(*arg, **kwargs)

    @register_counter_info_to_status_decorator(increment=1, info_key='predict_counter')
    def predict(self, obs, **kwargs):
        return self.model_free_algo.predict(obs)

    def append_to_memory(self, *args, **kwargs):
        self.model_free_algo.append_to_memory(kwargs['samples'])

    def save(self, global_step, save_path=None, name=None, **kwargs):

        MultiPlaceholderInput.save(self, save_path=save_path, global_step=global_step, name=name, **kwargs)

    def load(self, path_to_model, model_name, global_step=None, **kwargs):

        MultiPlaceholderInput.load(self, path_to_model, model_name, global_step, **kwargs)

    @register_counter_info_to_status_decorator(increment=1, info_key='dyanmics_train_counter', under_status='TRAIN')
    def _fit_dynamics_model(self, batch_data: TransitionData, train_iter, sess=None) -> dict:
        res_dict = self.dynamics_model.train(batch_data, **dict(sess=sess,
                                                                train_iter=train_iter))
        return res_dict

    @register_counter_info_to_status_decorator(increment=1, info_key='mode_free_algo_dyanmics_train_counter',
                                               under_status='TRAIN')
    def _train_model_free_algo(self, batch_data=None, train_iter=None, sess=None):
        res_dict = self.model_free_algo.train(**dict(batch_data=batch_data,
                                                     train_iter=train_iter,
                                                     sess=sess))
        return res_dict
