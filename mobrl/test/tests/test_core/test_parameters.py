from mobrl.config.global_config import GlobalConfig
from mobrl.test.tests.set_up.setup import TestWithLogSet
import numpy as np


class TestParam(TestWithLogSet):
    def test_basic(self):
        a, locals = self.create_parameters()
        a.save(save_path=GlobalConfig.DEFAULT_LOG_PATH + '/param_path',
               name=a.name,
               global_step=0)
        or_val = a._source_config.config_dict['var1']
        or_param = a('param3').copy()
        a._source_config.config_dict['var1'] = 100
        a._parameters['param3'] = 1000
        self.assertNotEqual(a._source_config.config_dict['var1'], or_val)
        self.assertFalse(np.equal(a._parameters['param3'], or_param).all())
        a.load(load_path=GlobalConfig.DEFAULT_LOG_PATH + '/param_path',
               name=a.name,
               global_step=0
               )
        self.assertEqual(a._source_config.config_dict['var1'], or_val)
        self.assertTrue(np.equal(a._parameters['param3'], or_param).all())
