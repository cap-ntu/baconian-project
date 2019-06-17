from baconian.algo.rl_algo import ModelBasedAlgo
from baconian.algo.dynamics.dynamics_model import DynamicsModel
from baconian.config.dict_config import DictConfig
from baconian.common.sampler.sample_data import TrajectoryData
from baconian.core.parameters import Parameters
from baconian.config.global_config import GlobalConfig
from baconian.common.misc import *
from baconian.algo.policy.policy import Policy
from baconian.common.logging import ConsoleLogger
from baconian.common.sampler.sample_data import TransitionData
from baconian.common.logging import record_return_decorator


class ModelPredictiveControl(ModelBasedAlgo):
    required_key_dict = DictConfig.load_json(file_path=GlobalConfig().DEFAULT_MPC_REQUIRED_KEY_LIST)

    def __init__(self, env_spec, dynamics_model: DynamicsModel,
                 config_or_config_dict: (DictConfig, dict),
                 policy: Policy,
                 name='mpc',
                 ):
        super().__init__(env_spec, dynamics_model, name)
        self.config = construct_dict_config(config_or_config_dict, self)
        self.policy = policy
        self.parameters = Parameters(parameters=dict(),
                                     source_config=self.config,
                                     name=name + '_' + 'mpc_param')
        self.memory = TransitionData(env_spec=env_spec)

    def init(self, source_obj=None):
        super().init()
        self.parameters.init()
        self._dynamics_model.init()
        self.policy.init()
        if source_obj:
            self.copy_from(source_obj)

    def train(self, *arg, **kwargs) -> dict:
        super(ModelPredictiveControl, self).train()
        res_dict = {}
        batch_data = kwargs['batch_data'] if 'batch_data' in kwargs else self.memory

        dynamics_train_res_dict = self._fit_dynamics_model(batch_data=batch_data,
                                                           train_iter=self.parameters('dynamics_model_train_iter'))
        for key, val in dynamics_train_res_dict.items():
            res_dict["mlp_dynamics_{}".format(key)] = val
        return res_dict

    def test(self, *arg, **kwargs) -> dict:
        return super().test(*arg, **kwargs)

    def _fit_dynamics_model(self, batch_data: TransitionData, train_iter, sess=None) -> dict:
        res_dict = self._dynamics_model.train(batch_data, **dict(sess=sess,
                                                                 train_iter=train_iter))
        return res_dict

    def predict(self, obs, **kwargs):
        if self.is_training is True:
            return self.env_spec.action_space.sample()
        rollout = TrajectoryData(env_spec=self.env_spec)
        state = obs
        for i in range(self.parameters('SAMPLED_PATH_NUM')):
            path = TransitionData(env_spec=self.env_spec)
            # todo terminal_func signal problem to be consider?
            for _ in range(self.parameters('SAMPLED_HORIZON')):
                ac = self.policy.forward(obs=state)
                new_state, re, done, _ = self.dynamics_env.step(action=ac, state=state)
                path.append(state=state, action=ac, new_state=new_state, reward=re, done=done)
                state = new_state
            rollout.append(path)
        rollout.trajectories.sort(key=lambda x: x.cumulative_reward, reverse=True)
        ac = rollout.trajectories[0].action_set[0]
        assert self.env_spec.action_space.contains(ac)
        return ac

    def append_to_memory(self, samples: TransitionData):
        self.memory.union(samples)

    def copy_from(self, obj) -> bool:
        if not isinstance(obj, type(self)):
            raise TypeError('Wrong type of obj %s to be copied, which should be %s' % (type(obj), type(self)))
        self.parameters.copy_from(obj.parameters)
        self._dynamics_model.copy_from(obj._dynamics_model)
        ConsoleLogger().print('info', 'model: {} copied from {}'.format(self, obj))
        return True

    @record_return_decorator(which_recorder='self')
    def save(self, global_step, save_path=None, name=None, **kwargs):
        save_path = save_path if save_path else GlobalConfig().DEFAULT_MODEL_CHECKPOINT_PATH
        name = name if name else self.name

        self._dynamics_model.save(save_path=save_path, global_step=global_step, name=name, **kwargs)
        self.policy.save(save_path=save_path, global_step=global_step, name=name, **kwargs)
        return dict(check_point_save_path=save_path, check_point_save_global_step=global_step,
                    check_point_save_name=name)

    @record_return_decorator(which_recorder='self')
    def load(self, path_to_model, model_name, global_step=None, **kwargs):
        self._dynamics_model.load(path_to_model, model_name, global_step, **kwargs)
        self.policy.load(path_to_model, model_name, global_step, **kwargs)
        return dict(check_point_load_path=path_to_model, check_point_load_global_step=global_step,
                    check_point_load_name=model_name)
