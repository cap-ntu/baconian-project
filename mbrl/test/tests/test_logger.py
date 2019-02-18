from mbrl.test.tests.test_setup import BaseTestCase
from mbrl.common.util.logger import Logger, record_return_decorator, reset_global_memo, global_logger
import numpy as np
from mbrl.core.basic import Basic
from mbrl.test.tests.test_setup import TestTensorflowSetup
from mbrl.algo.rl.model_free.dqn import DQN
from mbrl.envs.gym_env import make
from mbrl.envs.env_spec import EnvSpec
from mbrl.algo.rl.value_func.mlp_q_value import MLPQValueFunction
from mbrl.test.tests.test_setup import TestTensorflowSetup


class Foo(Basic):
    def __init__(self, use_global=False):
        super().__init__()
        self.loss = 1.0
        self.logger = Logger(log_path='/home/dls/tmp/test/', config_or_config_dict=dict(), log_level=0,
                             use_global_memo=use_global)

    def get_status(self):
        return dict(x=1)

    def get_val(self):
        return np.random.random()

    @record_return_decorator(which_logger='self')
    def get_by_return(self, res, num=2, *args, **kwargs):
        return dict(val=res * num, val2=res)

    @property
    def name(self):
        return 'foo'


class TestLogger(BaseTestCase):
    def test_register(self):
        reset_global_memo()
        obj = Foo()

        a = Logger(log_path='/home/dls/tmp/test/', config_or_config_dict=dict(), log_level=0)
        a.register_logging_attribute_by_record(obj=obj, obj_name='foo', attr_name='val', get_method_name='get_val',
                                               static_flag=False)
        a.register_logging_attribute_by_record(obj=obj, obj_name='foo', attr_name='loss', static_flag=True)
        a.record()
        print(a._global_obj_log)
        self.assertTrue('val' in a._global_obj_log[obj])
        self.assertTrue('loss' in a._global_obj_log[obj])
        obj.loss = 10.0
        a.record()
        # print(a._global_obj_log)

        b = Logger(log_path='/home/dls/tmp/test/', config_or_config_dict=dict(), log_level=0, use_global_memo=False)
        b.register_logging_attribute_by_record(obj=obj, obj_name='foo', attr_name='val', get_method_name='get_val',
                                               static_flag=False)
        b.register_logging_attribute_by_record(obj=obj, obj_name='foo', attr_name='loss', static_flag=True)

        b.record()
        # print(b._global_obj_log)
        self.assertTrue('val' in b._global_obj_log[obj])
        self.assertTrue('loss' in b._global_obj_log[obj])
        obj.loss = 10.0
        b.record()
        # print(b._global_obj_log)
        # global _global_obj_log
        # global _registered_log_file_dict
        self.assertTrue(b._global_obj_log is not a._global_obj_log)
        self.assertTrue(b._registered_log_attr_by_get_dict is not a._registered_log_attr_by_get_dict)
        from mbrl.common.util.logger import _registered_log_attr_by_get_dict, _global_obj_log
        self.assertTrue(a._global_obj_log is _global_obj_log)
        self.assertTrue(a._registered_log_attr_by_get_dict is _registered_log_attr_by_get_dict)

    def test_return_record(self):
        reset_global_memo()
        from mbrl.common.util.logger import _registered_log_attr_by_get_dict, _global_obj_log
        self.assertEqual(len(_global_obj_log), 0)
        obj = Foo(use_global=True)
        obj.get_by_return(res=10, num=2)
        obj.get_by_return(res=1, num=2)
        obj.get_by_return(res=2, num=4)
        print(obj.logger._global_obj_log)
        self.assertEqual(len(obj.logger._global_obj_log), 1)
        self.assertTrue(obj in obj.logger._global_obj_log)
        self.assertTrue('val' in obj.logger._global_obj_log[obj])
        self.assertTrue(len(obj.logger._global_obj_log[obj]['val']) == 3)
        self.assertTrue(obj.logger._global_obj_log[obj]['val'][0]['value'] == 20)
        self.assertTrue(obj.logger._global_obj_log[obj]['val'][1]['value'] == 2)
        self.assertTrue(obj.logger._global_obj_log[obj]['val'][2]['value'] == 8)

        self.assertTrue('val2' in obj.logger._global_obj_log[obj])
        self.assertTrue(len(obj.logger._global_obj_log[obj]['val2']) == 3)
        self.assertTrue(obj.logger._global_obj_log[obj]['val2'][0]['value'] == 10)
        self.assertTrue(obj.logger._global_obj_log[obj]['val2'][1]['value'] == 1)
        self.assertTrue(obj.logger._global_obj_log[obj]['val2'][2]['value'] == 2)

        obj = Foo(use_global=True)
        obj.get_by_return(res=10, num=2)
        obj.get_by_return(res=1, num=2)
        obj.get_by_return(res=2, num=4)
        print(obj.logger._global_obj_log)
        self.assertEqual(len(obj.logger._global_obj_log), 2)
        self.assertTrue(obj in obj.logger._global_obj_log)
        self.assertTrue('val' in obj.logger._global_obj_log[obj])
        self.assertTrue(len(obj.logger._global_obj_log[obj]['val']) == 3)
        self.assertTrue(obj.logger._global_obj_log[obj]['val'][0]['value'] == 20)
        self.assertTrue(obj.logger._global_obj_log[obj]['val'][1]['value'] == 2)
        self.assertTrue(obj.logger._global_obj_log[obj]['val'][2]['value'] == 8)

        self.assertTrue('val2' in obj.logger._global_obj_log[obj])
        self.assertTrue(len(obj.logger._global_obj_log[obj]['val2']) == 3)
        self.assertTrue(obj.logger._global_obj_log[obj]['val2'][0]['value'] == 10)
        self.assertTrue(obj.logger._global_obj_log[obj]['val2'][1]['value'] == 1)
        self.assertTrue(obj.logger._global_obj_log[obj]['val2'][2]['value'] == 2)

        print(_global_obj_log)


class TesTLoggerWithDQN(TestTensorflowSetup):
    def test_init(self):
        reset_global_memo()
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
        from mbrl.common.sampler.sample_data import TransitionData
        a = TransitionData(env_spec)
        res = []
        for i in range(100):
            ac = dqn.predict(obs=st, sess=self.sess, batch_flag=False)
            st_new, re, done, _ = env.step(action=ac)
            a.append(state=st, new_state=st_new, action=ac, done=done, reward=re)
            dqn.append_to_memory(a)
        res.append(dqn.train(batch_data=a, train_iter=10, sess=None, update_target=True)['average_loss'])
        res.append(dqn.train(batch_data=None, train_iter=10, sess=None, update_target=True)['average_loss'])
        self.assertTrue(dqn in global_logger._global_obj_log)
        self.assertTrue('average_loss' in global_logger._global_obj_log[dqn])
        self.assertTrue(len(global_logger._global_obj_log[dqn]['average_loss']) == 2)
        self.assertTrue(
            np.equal(np.array(res), [x['value'] for x in global_logger._global_obj_log[dqn]['average_loss']]).all())


if __name__ == '__main__':
    import unittest

    unittest.main()
