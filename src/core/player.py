import os
from src.config.dict_config import DictConfig
from conf.key import CONFIG_KEY
import numpy as np
import random
from log.intelligentTestLog import INTEL_LOG
from src.core import Logger
from src.core.basic import Basic


class GamePlayer(Basic):
    key_list = DictConfig.load_json(file_path=CONFIG_KEY + '/gamePlayerKey.json')

    def __init__(self, config, agent, env, basic_list, ep_type=0, log_path=None, log_path_end_with=""):
        super(GamePlayer, self).__init__()
        self.config = config
        self.agent = agent
        self.env = env
        self.basic_list = []
        for basic in basic_list:
            if basic is not None:
                self.basic_list.append(basic)
        self.basic_list.append(self)
        if ep_type == 1:
            log = INTEL_LOG
        else:
            log = LOG
        self.logger = Logger(prefix=self.config.config_dict['GAME_NAME'],
                             log=log,
                             log_path=log_path,
                             log_path_end=log_path_end_with)

        self.step_count = 0

    @property
    def real_env_sample_count(self):
        return self.env.target_agent._real_env_sample_count

    def set_seed(self, seed=None):
        if seed is None:
            seed = int(self.config.config_dict['SEED'])
        else:
            self.config.config_dict['SEED'] = seed
        np.random.seed(seed)
        random.seed(seed)

    def save_config(self):
        if self.config.config_dict['SAVE_CONFIG_FILE_FLAG'] == 1:
            for basic in self.basic_list:
                if basic.config is not None:
                    basic.config.save_config(path=self.logger.config_file_log_dir,
                                             name=basic.name + '.json')
            self.config.save_config(path=self.logger.config_file_log_dir,
                                    name='GamePlayer.json')
            # Config.save_to_json(dict=cfg.config_dict,
            #                     file_name='expConfig.json',
            #                     path=self.logger.config_file_log_dir)

    def init(self):
        self.agent.init()
        self.env.init()

    def step(self, step_flag=True):
        trainer_data = self.agent.sample(env=self.env,
                                         sample_count=1,
                                         store_flag=step_flag,
                                         agent_print_log_flag=True)
        self.step_count += 1
        if self.step_count % 1 == 0 and self.step_count > 0:
            self.agent.train()
        return trainer_data

    def play(self, seed_new=None):
        self.set_seed(seed_new)
        self.save_config()
        self.init()

        info_set = []

        # TODO modify here to control the whole training process
        for i in range(self.config.config_dict['EPOCH']):
            for j in range(self.config.config_dict['STEP']):
                print("\nEPOCH %d, STEP %d" % (i, j))
                trainer_data = self.step()
                # info_set[0].append(self.agent.sampler.info[0])
                # info_set[1].append(self.agent.sampler.info[1])
                # info_set[2].append(self.agent.sampler.info[2])
                info_set.append(trainer_data)
                print("self.config.config_dict['MAX_REAL_ENV_SAMPLE']=", self.config.config_dict['MAX_REAL_ENV_SAMPLE'])
                print("self.env.target_agent._real_env_sample_count=", self.real_env_sample_count)
                if self.real_env_sample_count > self.config.config_dict['MAX_REAL_ENV_SAMPLE']:
                    break
            if self.real_env_sample_count > self.config.config_dict['MAX_REAL_ENV_SAMPLE']:
                break
        # END

        return info_set

    def print_log_to_file(self):
        for basic in self.basic_list:
            if 'LOG_FLAG' in basic.config.config_dict and basic.config.config_dict['LOG_FLAG'] == 1:
                basic.status = basic.status_key['TRAIN']
                self.logger.out_to_file(file_path=os.path.join(self.logger.loss_file_log_dir, basic.log_file),
                                        content=basic.log_file_content)
                basic.status = basic.status_key['TEST']
                self.logger.out_to_file(file_path=os.path.join(self.logger.loss_file_log_dir, basic.log_file),
                                        content=basic.log_file_content)

    def save_all_model(self):
        from src.rl.tensorflowBasedModel import TensorflowBasedModel
        for basic in self.basic_list:
            if isinstance(basic, TensorflowBasedModel):
                basic.save_model(path=self.logger.model_file_log_dir, global_step=1)

    def load_all_model(self):
        from src.rl.tensorflowBasedModel import TensorflowBasedModel
        for basic in self.basic_list:
            if isinstance(basic, TensorflowBasedModel):
                basic.load_model(path=self.logger.model_file_log_dir, global_step=1)

    def _security_check(self):
        pass
