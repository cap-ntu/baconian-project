from baconian.test.tests.set_up.setup import TestWithAll
from baconian.common.logging import Logger, ConsoleLogger, Recorder, record_return_decorator
import numpy as np
from baconian.core.core import Basic, EnvSpec
from baconian.algo.dqn import DQN
from baconian.envs.gym_env import make
from baconian.algo.value_func.mlp_q_value import MLPQValueFunction
from baconian.core.agent import Agent


class Foo(Basic):
    def __init__(self, name='foo'):
        super().__init__(name=name)
        self.loss = 1.0
        self.recorder = Recorder(flush_by_split_status=False, default_obj=self)

    def get_status(self):
        return dict(x=1)

    def get_val(self):
        return np.random.random()

    @record_return_decorator(which_recorder='self')
    def get_by_return(self, res, num=2, *args, **kwargs):
        return dict(val=res * num, val2=res)

    @property
    def name(self):
        return 'foo'


class TestLogger(TestWithAll):
    def test_register(self):
        obj = Foo()
        a = Recorder(flush_by_split_status=False, default_obj=obj)
        a.register_logging_attribute_by_record(obj=obj, attr_name='val', get_method=lambda x: x['obj'].get_val(),
                                               static_flag=False)
        a.register_logging_attribute_by_record(obj=obj, attr_name='loss', static_flag=True)
        a.record()
        print(a._obj_log)
        self.assertTrue('val' in a._obj_log[obj])
        self.assertTrue('loss' in a._obj_log[obj])
        obj.loss = 10.0
        a.record()

        b = Recorder(flush_by_split_status=False, default_obj=obj)
        b.register_logging_attribute_by_record(obj=obj, attr_name='val', get_method=lambda x: x['obj'].get_val(),
                                               static_flag=False)
        b.register_logging_attribute_by_record(obj=obj, attr_name='loss', static_flag=True)

        b.record()
        self.assertTrue('val' in b._obj_log[obj])
        self.assertTrue('loss' in b._obj_log[obj])
        obj.loss = 10.0
        b.record()
        self.assertTrue(b._obj_log is not a._obj_log)
        self.assertTrue(b._registered_log_attr_by_get_dict is not a._registered_log_attr_by_get_dict)

    def test_return_record(self):
        obj = Foo(name='foo')
        obj.get_by_return(res=10, num=2)
        obj.get_by_return(res=1, num=2)
        obj.get_by_return(res=2, num=4)
        print(obj.recorder._obj_log)
        self.assertEqual(len(obj.recorder._obj_log), 1)
        self.assertTrue(obj in obj.recorder._obj_log)
        self.assertTrue('val' in obj.recorder._obj_log[obj])
        self.assertTrue(len(obj.recorder._obj_log[obj]['val']) == 3)
        self.assertTrue(obj.recorder._obj_log[obj]['val'][0]['log_val'] == 20)
        self.assertTrue(obj.recorder._obj_log[obj]['val'][1]['log_val'] == 2)
        self.assertTrue(obj.recorder._obj_log[obj]['val'][2]['log_val'] == 8)

        self.assertTrue('val2' in obj.recorder._obj_log[obj])
        self.assertTrue(len(obj.recorder._obj_log[obj]['val2']) == 3)
        self.assertTrue(obj.recorder._obj_log[obj]['val2'][0]['log_val'] == 10)
        self.assertTrue(obj.recorder._obj_log[obj]['val2'][1]['log_val'] == 1)
        self.assertTrue(obj.recorder._obj_log[obj]['val2'][2]['log_val'] == 2)

        obj = Foo(name='foo2')
        obj.get_by_return(res=10, num=2)
        obj.get_by_return(res=1, num=2)
        obj.get_by_return(res=2, num=4)
        print(obj.recorder._obj_log)
        self.assertTrue(obj in obj.recorder._obj_log)
        self.assertTrue('val' in obj.recorder._obj_log[obj])
        self.assertTrue(len(obj.recorder._obj_log[obj]['val']) == 3)
        self.assertTrue(obj.recorder._obj_log[obj]['val'][0]['log_val'] == 20)
        self.assertTrue(obj.recorder._obj_log[obj]['val'][1]['log_val'] == 2)
        self.assertTrue(obj.recorder._obj_log[obj]['val'][2]['log_val'] == 8)

        self.assertTrue('val2' in obj.recorder._obj_log[obj])
        self.assertTrue(len(obj.recorder._obj_log[obj]['val2']) == 3)
        self.assertTrue(obj.recorder._obj_log[obj]['val2'][0]['log_val'] == 10)
        self.assertTrue(obj.recorder._obj_log[obj]['val2'][1]['log_val'] == 1)
        self.assertTrue(obj.recorder._obj_log[obj]['val2'][2]['log_val'] == 2)


class TesTLoggerWithDQN(TestWithAll):

    def test_integration_with_dqn(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
                                  name='mlp_q',
                                  name_scope='mlp_q',
                                  mlp_config=[
                                      {
                                          "ACT": "RELU",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "1",
                                          "N_UNITS": 16,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      },
                                      {
                                          "ACT": "LINEAR",
                                          "B_INIT_VALUE": 0.0,
                                          "NAME": "OUPTUT",
                                          "N_UNITS": 1,
                                          "TYPE": "DENSE",
                                          "W_NORMAL_STDDEV": 0.03
                                      }
                                  ])
        dqn = DQN(env_spec=env_spec,
                  name='dqn_test',
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             LEARNING_RATE=0.001,
                                             TRAIN_ITERATION=1,
                                             DECAY=0.5),
                  value_func=mlp_q)
        agent = Agent(env=env, env_spec=env_spec,
                      algo=dqn,
                      name='agent')
        agent.init()
        # dqn.init()
        st = env.reset()
        from baconian.common.sampler.sample_data import TransitionData
        a = TransitionData(env_spec)
        res = []
        agent.sample(env=env,
                     sample_count=100,
                     in_which_status='TRAIN',
                     store_flag=True,
                     sample_type='transition')
        agent.sample(env=env,
                     sample_count=100,
                     in_which_status='TRAIN',
                     store_flag=True,
                     sample_type='transition')
        res.append(dqn.train(batch_data=a, train_iter=10, sess=None, update_target=True)['average_loss'])
        res.append(dqn.train(batch_data=None, train_iter=10, sess=None, update_target=True)['average_loss'])
        self.assertTrue(dqn in dqn.recorder._obj_log)
        self.assertTrue('average_loss' in dqn.recorder._obj_log[dqn])
        self.assertTrue(len(dqn.recorder._obj_log[dqn]['average_loss']) == 2)
        self.assertTrue(
            np.equal(np.array(res), [x['log_val'] for x in dqn.recorder._obj_log[dqn]['average_loss']]).all())

        self.assertTrue(len(Logger()._registered_recorders) > 0)
        self.assertTrue(dqn.recorder in Logger()._registered_recorders)
        res = dqn.recorder.get_log(attr_name='average_loss', filter_by_status=dict())
        self.assertEqual(len(res), 2)
        res = agent.recorder.get_log(attr_name='sum_reward', filter_by_status={'status': 'TRAIN'})
        self.assertEqual(len(res), 2)
        res = agent.recorder.get_log(attr_name='sum_reward', filter_by_status={'status': 'TEST'})
        self.assertEqual(len(res), 0)
        Logger().flush_recorder()

    def test_console_logger(self):
        self.assertTrue(ConsoleLogger().inited_flag)
        logger = ConsoleLogger()
        self.assertTrue(logger.inited_flag)
        logger.print('info', 'this is for test %s', 'args')
        logger.print('info', 'this is for test {}'.format('args'))

        logger2 = ConsoleLogger()
        self.assertEqual(id(logger), id(logger2))
        logger.flush()


if __name__ == '__main__':
    import unittest

    unittest.main()
