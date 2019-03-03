from baconian.test.tests.set_up.setup import TestWithAll
from baconian.algo.rl.misc.sample_processor import SampleProcessor


class TestSampleProcessor(TestWithAll):
    def test_init(self):
        ppo = self.create_ppo()
