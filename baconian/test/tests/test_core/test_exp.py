from baconian.test.tests.set_up.setup import BaseTestCase
from baconian.core.experiment_runner import single_exp_runner, duplicate_exp_runner
from baconian.common.schedules import LinearSchedule, PiecewiseSchedule
from baconian.config.global_config import GlobalConfig
from baconian.core.status import get_global_status_collect


class TestExperiment(BaseTestCase):
    def test_experiment(self):
        def func():
            GlobalConfig.set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=200,
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

        single_exp_runner(func, auto_choose_gpu_flag=False, gpu_id=0)

    def test_exp_with_scheduler(self, algo=None, locals=None):
        def wrap_algo(algo=None, locals=None):
            def func(algo=algo, locals=locals):
                GlobalConfig.set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=500,
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
                                              scheduler=LinearSchedule(
                                                  t_fn=exp.TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT,
                                                  schedule_timesteps=GlobalConfig.DEFAULT_EXPERIMENT_END_POINT[
                                                      'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'],
                                                  final_p=0.0001,
                                                  initial_p=0.01))
                exp.run()
                self.assertEqual(exp.TOTAL_AGENT_TEST_SAMPLE_COUNT(), exp.TOTAL_ENV_STEP_TEST_SAMPLE_COUNT())
                self.assertEqual(exp.TOTAL_AGENT_TRAIN_SAMPLE_COUNT(), exp.TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT(), 500)

            return func

        single_exp_runner(wrap_algo(algo, locals), auto_choose_gpu_flag=False, gpu_id=0)

    def test_duplicate_exp(self):
        def func():
            GlobalConfig.set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=500,
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
            exp = self.create_exp(name='model_fre', env=env, agent=agent)
            exp.run()

        duplicate_exp_runner(2, func, auto_choose_gpu_flag=False, gpu_id=0)

    def test_saving_scheduler_on_all_model_free_algo(self):
        # to_test_algo_func = (self.create_ppo, self.create_dqn, self.create_ddpg,)
        to_test_algo_func = (self.create_ppo,)
        for func in to_test_algo_func:
            self.setUp()
            single_exp_runner(_saving_scheduler(self, func), auto_choose_gpu_flag=False, gpu_id=0)
            self.tearDown()


def _saving_scheduler(self, creat_func=None):
    def wrap_algo():
        def func(self, creat_func=None):
            GlobalConfig.set('DEFAULT_EXPERIMENT_END_POINT', dict(TOTAL_AGENT_TRAIN_SAMPLE_COUNT=500,
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

            exp = self.create_exp(name='model_free', env=env, agent=agent)
            # agent.explorations_strategy.parameters.set_scheduler(param_key='init_random_prob',
            #                                                      scheduler=PiecewiseSchedule(
            #                                                          t_fn=lambda: get_global_status_collect()(
            #                                                              'TOTAL_AGENT_TRAIN_SAMPLE_COUNT'),
            #                                                          endpoints=((10, 0.3), (100, 0.1), (200, 0.0)),
            #                                                          outside_value=0.0
            #                                                      ))
            exp.run()
            self.assertEqual(exp.TOTAL_AGENT_TEST_SAMPLE_COUNT(), exp.TOTAL_ENV_STEP_TEST_SAMPLE_COUNT())
            self.assertEqual(exp.TOTAL_AGENT_TRAIN_SAMPLE_COUNT(), exp.TOTAL_ENV_STEP_TRAIN_SAMPLE_COUNT(), 500)

        return func(self, creat_func)

    return wrap_algo
