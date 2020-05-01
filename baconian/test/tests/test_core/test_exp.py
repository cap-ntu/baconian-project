from baconian.test.tests.set_up.setup import BaseTestCase
from baconian.core.experiment_runner import single_exp_runner, duplicate_exp_runner
from baconian.common.schedules import LinearScheduler
from baconian.config.global_config import GlobalConfig
import os


class TestExperiment(BaseTestCase):
    def test_experiment(self):
        def func():
            GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=200,
                                                                    TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                                                    TOTAL_AGENT_UPDATE_COUNT=None))
            dqn, locals = self.create_dqn()
            env_spec = locals['env_spec']
            env = locals['env']
            agent = self.create_agent(env=locals['env'],
                                      algo=dqn,
                                      name='agent',
                                      eps=self.create_eps(env_spec)[0],
                                      env_spec=env_spec)[0]
            exp = self.create_exp(name='model_free', env=env, agent=agent)
            exp.run()

        single_exp_runner(func, auto_choose_gpu_flag=False, gpu_id=0, del_if_log_path_existed=True)

    def test_exp_with_scheduler(self, algo=None, locals=None):
        def wrap_algo(algo=None, locals=None):
            def func(algo=algo, locals=locals):
                GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=500,
                                                                        TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                                                        TOTAL_AGENT_UPDATE_COUNT=None))
                if not algo:
                    algo, locals = self.create_dqn()
                env_spec = locals['env_spec']
                env = locals['env']
                agent = self.create_agent(env=locals['env'],
                                          algo=algo,
                                          name='agent',
                                          eps=self.create_eps(env_spec)[0],
                                          env_spec=env_spec)[0]

                exp = self.create_exp(name='model_free', env=env, agent=agent)
                algo.parameters.set_scheduler(param_key='LEARNING_RATE',
                                              to_tf_ph_flag=True,
                                              scheduler=LinearScheduler(
                                                  t_fn=exp.TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT,
                                                  schedule_timesteps=GlobalConfig().DEFAULT_EXPERIMENT_END_POINT[
                                                      'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'],
                                                  final_p=0.0001,
                                                  initial_p=0.01))
                exp.run()
                self.assertEqual(exp.TOTAL_AGENT_TEST_SAMPLE_COUNT(), exp.TOTAL_ENV_STEP_TEST_SAMPLE_COUNT())
                self.assertEqual(exp.TOTAL_AGENT_TRAIN_SAMPLE_COUNT(), exp.TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT(), 500)

            return func

        single_exp_runner(wrap_algo(algo, locals), auto_choose_gpu_flag=False, gpu_id=0,
                          del_if_log_path_existed=True)

    def test_duplicate_exp(self):
        def func():
            GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=500,
                                                                    TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                                                    TOTAL_AGENT_UPDATE_COUNT=None))
            dqn, locals = self.create_dqn()
            env_spec = locals['env_spec']
            env = locals['env']
            agent = self.create_agent(env=locals['env'],
                                      algo=dqn,
                                      name='agent',
                                      eps=self.create_eps(env_spec)[0],
                                      env_spec=env_spec)[0]
            exp = self.create_exp(name='model_free', env=env, agent=agent)
            exp.run()

        base_path = GlobalConfig().DEFAULT_LOG_PATH
        duplicate_exp_runner(2, func, auto_choose_gpu_flag=False, gpu_id=0)
        self.assertTrue(os.path.isdir(base_path))
        self.assertTrue(os.path.isdir(os.path.join(base_path, 'exp_0')))
        self.assertTrue(os.path.isdir(os.path.join(base_path, 'exp_1')))
        self.assertTrue(os.path.isdir(os.path.join(base_path, 'exp_0', 'record')))
        self.assertTrue(os.path.isdir(os.path.join(base_path, 'exp_1', 'record')))
        self.assertTrue(os.path.isfile(os.path.join(base_path, 'exp_0', 'console.log')))
        self.assertTrue(os.path.isfile(os.path.join(base_path, 'exp_1', 'console.log')))

    def test_saving_scheduler_on_all_model_free_algo(self):
        to_test_algo_func = (self.create_ppo, self.create_dqn, self.create_ddpg)
        sample_traj_flag = (True, False, False)
        for i, func in enumerate(to_test_algo_func):
            self.setUp()
            single_exp_runner(_saving_scheduler(self, func,
                                                sample_traj_flag=sample_traj_flag[i]),
                              auto_choose_gpu_flag=False,
                              gpu_id=0,
                              del_if_log_path_existed=True)
            self.tearDown()

    def test_saving_scheduler_on_all_model_based_algo(self):
        to_test_algo_func = (self.create_mpc, self.create_dyna)
        for func in to_test_algo_func:
            self.setUp()
            single_exp_runner(_saving_scheduler(self, func), auto_choose_gpu_flag=False,
                              gpu_id=0, del_if_log_path_existed=True)
            self.tearDown()


def _saving_scheduler(self, creat_func=None, sample_traj_flag=False):
    def wrap_algo():
        def func(self, creat_func=None):
            GlobalConfig().set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=500,
                                                                    TOTAL_AGENT_TEST_SAMPLE_COUNT=None,
                                                                    TOTAL_AGENT_UPDATE_COUNT=None))
            if not creat_func:
                algo, locals = self.create_dqn()
            else:
                algo, locals = creat_func()
            env_spec = locals['env_spec']
            env = locals['env']
            agent = self.create_agent(env=locals['env'],
                                      algo=algo,
                                      name='agent',
                                      eps=self.create_eps(env_spec)[0],
                                      env_spec=env_spec)[0]
            flow = None
            from baconian.algo.dyna import Dyna
            if isinstance(algo, Dyna):
                flow = self.create_dyna_flow(agent=agent, env=env)[0]
            exp = self.create_exp(name='model_free', env=env, agent=agent, flow=flow, traj_flag=sample_traj_flag)
            exp.run()
            self.assertEqual(exp.TOTAL_AGENT_TEST_SAMPLE_COUNT(), exp.TOTAL_ENV_STEP_TEST_SAMPLE_COUNT())
            self.assertEqual(exp.TOTAL_AGENT_TRAIN_SAMPLE_COUNT(), exp.TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT(), 500)

        return func(self, creat_func)

    return wrap_algo
