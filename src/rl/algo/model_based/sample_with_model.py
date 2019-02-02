from src.rl.algo.algo import ModelBasedAlgo
from src.rl.algo.model_based.models.dynamics_model import DynamicsModel
from typeguard import typechecked
from src.config.dict_config import DictConfig
from src.misc.misc import *
from src.common.sampler.sample_data import TransitionData
from src.tf.tf_parameters import TensorflowParameters
from src.core.global_config import GlobalConfig
from src.rl.algo.algo import ModelFreeAlgo


class SampleWithDynamics(ModelBasedAlgo):
    """
    The naive model based method by approximating a dynamics with nn and sample from it to train the agent.
    """
    required_key_list = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_ALGO_SAMPLE_WITH_DYNAMICS_REQUIRED_KEY_LIST)

    @typechecked
    def __init__(self, env_spec, dynamics_model: DynamicsModel,
                 model_free_algo: ModelFreeAlgo,
                 config_or_config_dict: (DictConfig, dict),
                 ):
        super().__init__(env_spec, dynamics_model)
        self.model_free_algo = model_free_algo
        self.config = construct_dict_config(config_or_config_dict, self)
        self.parameters = TensorflowParameters(tf_var_list=[],
                                               rest_parameters=dict(),
                                               auto_init=False,
                                               name='sample_with_model_para',
                                               source_config=self.config,
                                               require_snapshot=False)

    def init(self):
        super().init()
        self.model_free_algo.init()
        self.dynamics_model.init()

    def train(self, batch_data) -> dict:
        super(SampleWithDynamics, self).train()
        dynamics_train_res_dict = self._fit_dynamics_model(batch_data=batch_data,
                                                           train_iter=self.parameters('dynamics_model_train_iter'))

        model_free_algo_train_res_dict = self._train_model_free_algo(batch_data=batch_data,
                                                                     train_iter=self.parameters(
                                                                         'model_free_algo_train_iter'))

        res_dict = {}
        for key, val in dynamics_train_res_dict.items():
            res_dict["mlp_dynamics_{}".format(key)] = val
        for key, val in model_free_algo_train_res_dict.items():
            res_dict['dqn_{}'.format(key)] = val
        return res_dict

    def test(self, *arg, **kwargs):
        super().test(*arg, **kwargs)

    def _fit_dynamics_model(self, batch_data: TransitionData, train_iter, sess=None) -> dict:
        average_loss = 0.0
        for i in range(train_iter):
            res_dict = self.dynamics_model.train(batch_data, **dict(sess=sess,
                                                                    train_iter=train_iter))
            average_loss += (res_dict['average_loss'])
        return dict(loss=average_loss / train_iter,
                    train_iter=train_iter)

    def _train_model_free_algo(self, batch_data: TransitionData, train_iter, sess=None):
        average_loss = 0.0
        for i in range(train_iter):
            res_dict = self.model_free_algo.train(batch_data, sess=sess)
            average_loss += (res_dict['average_loss'])
        return dict(loss=average_loss / train_iter,
                    train_iter=train_iter)

    def predict(self, obs, **kwargs):
        return self.model_free_algo.predict(obs)

    def append_to_memory(self, *args, **kwargs):
        self.model_free_algo.append_to_memory(kwargs['samples'])
