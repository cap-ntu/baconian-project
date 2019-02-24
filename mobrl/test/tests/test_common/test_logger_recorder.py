from mobrl.test.tests.test_setup import BaseTestCase, TestWithLogSet
from mobrl.common.util.logger import ConsoleLogger, Logger
from mobrl.common.util.recorder import record_return_decorator
import numpy as np
from mobrl.core.basic import Basic
from mobrl.algo.rl.model_free.dqn import DQN
from mobrl.envs.gym_env import make
from mobrl.envs.env_spec import EnvSpec
from mobrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mobrl.test.tests.test_setup import TestTensorflowSetup
from mobrl.common.util.recorder import Recorder


class Foo(Basic):
    def __init__(self):
        super().__init__()
        self.loss = 1.0
        self.recorder = Recorder()

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


class TestLogger(TestWithLogSet):
    def test_register(self):
        obj = Foo()

        a = Recorder()
        a.register_logging_attribute_by_record(obj=obj, attr_name='val', get_method=lambda x: x['obj'].get_val,
                                               static_flag=False)
        a.register_logging_attribute_by_record(obj=obj, attr_name='loss', static_flag=True)
        a.record()
        print(a._obj_log)
        self.assertTrue('val' in a._obj_log[obj])
        self.assertTrue('loss' in a._obj_log[obj])
        obj.loss = 10.0
        a.record()

        b = Recorder()
        b.register_logging_attribute_by_record(obj=obj, attr_name='val', get_method=lambda x: x['obj'].get_val,
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
        obj = Foo()
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

        obj = Foo()
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


class TesTLoggerWithDQN(TestTensorflowSetup, TestWithLogSet):

    def setUp(self):
        self.assertFalse(ConsoleLogger().inited_flag)
        self.assertFalse(Logger().inited_flag)
        TestWithLogSet.setUp(self)
        TestTensorflowSetup.setUp(self)

        self.assertTrue(ConsoleLogger().inited_flag)
        self.assertTrue(Logger().inited_flag)

    def test_integration_with_dqn(self):
        env = make('Acrobot-v1')
        env_spec = EnvSpec(obs_space=env.observation_space,
                           action_space=env.action_space)

        mlp_q = MLPQValueFunction(env_spec=env_spec,
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
                  adaptive_learning_rate=True,
                  config_or_config_dict=dict(REPLAY_BUFFER_SIZE=1000,
                                             GAMMA=0.99,
                                             BATCH_SIZE=10,
                                             Q_NET_L1_NORM_SCALE=0.001,
                                             Q_NET_L2_NORM_SCALE=0.001,
                                             LEARNING_RATE=0.001,
                                             TRAIN_ITERATION=1,
                                             DECAY=0.5),
                  value_func=mlp_q)
        dqn.init()
        st = env.reset()
        from mobrl.common.sampler.sample_data import TransitionData
        a = TransitionData(env_spec)
        res = []
        for i in range(100):
            ac = dqn.predict(obs=st, sess=self.sess, batch_flag=False)
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
            dqn.append_to_memory(a)
        res.append(dqn.train(batch_data=a, train_iter=10, sess=None, update_target=True)['average_loss'])
        res.append(dqn.train(batch_data=None, train_iter=10, sess=None, update_target=True)['average_loss'])
        self.assertTrue(dqn in dqn.recorder._obj_log)
        self.assertTrue('average_loss' in dqn.recorder._obj_log[dqn])
        self.assertTrue(len(dqn.recorder._obj_log[dqn]['average_loss']) == 2)
        self.assertTrue(
            np.equal(np.array(res), [x['log_val'] for x in dqn.recorder._obj_log[dqn]['average_loss']]).all())

        self.assertTrue(len(Logger()._registered_recorders) > 0)
        self.assertTrue(dqn.recorder in Logger()._registered_recorders)

        self.assertTrue('dqn_adaptive_learning_rate' in dqn.recorder._obj_log[dqn])
        self.assertTrue(len(dqn.recorder._obj_log[dqn]['dqn_adaptive_learning_rate']) == 2)
        print(dqn.recorder._obj_log[dqn]['dqn_adaptive_learning_rate'])

        Logger().flush_recorder()

    def test_console_logger(self):
        self.assertTrue(ConsoleLogger().inited_flag)
        logger = ConsoleLogger()
        self.assertTrue(logger.inited_flag)
        self.assertTrue(logger.inited_flag)
        logger.print('info', 'this is for test %s', 'args')

        logger2 = ConsoleLogger()
        self.assertEqual(id(logger), id(logger2))
        logger.flush()

    def tearDown(self):
        TestTensorflowSetup.tearDown(self)
        TestWithLogSet.tearDown(self)


if __name__ == '__main__':
    import unittest

    unittest.main()
