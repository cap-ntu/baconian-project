from src.common.misc.special import flatten
from src.rl.algo.algo import ModelBasedAlgo
from src.rl.algo.model_based.models.dynamics_model import DynamicsModel
from src.config.dict_config import DictConfig
from src.common.sampler.sample_data import TrajectoryData, TransitionData
from src.core.parameters import Parameters
from src.config.global_config import GlobalConfig
from src.common.misc.misc import *
from src.rl.algo.model_based.misc.reward_func.reward_func import RewardFunc
from src.rl.algo.model_based.misc.terminal_func.terminal_func import TerminalFunc


class ModelPredictiveControl(ModelBasedAlgo):
    required_key_list = DictConfig.load_json(file_path=GlobalConfig.DEFAULT_MPC_REQUIRED_KEY_LIST)

    def __init__(self, env_spec, dynamics_model: DynamicsModel,
                 config_or_config_dict: (DictConfig, dict),
                 reward_func: RewardFunc,
                 terminal_func: TerminalFunc
                 ):
        super().__init__(env_spec, dynamics_model)
        self.config = construct_dict_config(config_or_config_dict, self)
        self.reward_func = reward_func
        self.terminal_func = terminal_func
        self.parameters = Parameters(parameters=dict(),
                                     source_config=self.config,
                                     auto_init=True,
                                     name='mpc_param')

    def init(self):
        super().init()
        self.dynamics_model.init()

    def train(self, *arg, **kwargs) -> dict:
        # only train dynamics model
        super(ModelPredictiveControl, self).train()
        res_dict = {}
        batch_data = kwargs['batch_data'] if 'batch_data' in kwargs else None

        dynamics_train_res_dict = self._fit_dynamics_model(batch_data=batch_data,
                                                           train_iter=self.parameters('dynamics_model_train_iter'))
        for key, val in dynamics_train_res_dict.items():
            res_dict["mlp_dynamics_{}".format(key)] = val
        return res_dict

    def test(self, *arg, **kwargs) -> dict:
        return super().test(*arg, **kwargs)

    def _fit_dynamics_model(self, batch_data: TransitionData, train_iter, sess=None) -> dict:
        res_dict = self.dynamics_model.train(batch_data, **dict(sess=sess,
                                                                train_iter=train_iter))
        return res_dict

    def predict(self, obs, **kwargs):
        # roll out and choose the init action with max cumulative reward_func
        rollout = TrajectoryData(env_spec=self.env_spec)
        state = obs
        for i in range(self.parameters('SAMPLED_PATH_NUM')):
            path = TransitionData(env_spec=self.env_spec)
            # todo terminal_func signal problem to be consider?
            for _ in range(self.parameters('SAMPLED_HORIZON')):
                ac = self.env_spec.action_space.sample()
                new_state = self.dynamics_model.step(action=ac, state=state)
                re = self.reward_func(state=state, action=ac, new_state=new_state)
                done = self.terminal_func(state=state, action=ac, new_state=new_state)
                path.append(state=state, action=ac, new_state=new_state, reward=re, done=done)
                state = new_state
            rollout.append(path)
        rollout.trajectories.sort(key=lambda x: x.cumulative_reward, reverse=True)
        ac = rollout.trajectories[0].action_set[0]
        assert self.env_spec.action_space.contains(ac)
        return ac

    def append_to_memory(self, *args, **kwargs):
        raise NotImplementedError
