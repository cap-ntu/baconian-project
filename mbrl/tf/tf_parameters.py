import tensorflow as tf
from mbrl.core.parameters import Parameters
from mbrl.config.global_config import GlobalConfig
from overrides.overrides import overrides
from typeguard import typechecked
import numpy as np


class TensorflowParameters(Parameters):

    @typechecked
    def __init__(self, tf_var_list: list, rest_parameters: dict, name: str,
                 max_to_keep=GlobalConfig.DEFAULT_MAX_TF_SAVER_KEEP,
                 auto_init=False,
                 source_config=None,
                 to_ph_parameter_dict=None,
                 require_snapshot=False):
        para_dict = {**dict(tf_var_list=tf_var_list), **rest_parameters}

        super(TensorflowParameters, self).__init__(parameters=para_dict,
                                                   auto_init=False,
                                                   name=name,
                                                   source_config=source_config)

        self.snapshot_var = []
        self.save_snapshot_op = []
        self.load_snapshot_op = []
        self.saver = None
        self.max_to_keep = max_to_keep
        self.require_snapshot = require_snapshot

        self._registered_tf_ph_dict = dict()
        if to_ph_parameter_dict:
            for key, val in to_ph_parameter_dict.items():
                self.to_tf_ph(key, ph=val)
        if auto_init is True:
            self.init()

    @overrides
    def init(self):
        sess = tf.get_default_session()
        sess.run(tf.variables_initializer(var_list=self._parameters['tf_var_list']))
        if self.require_snapshot is True:
            if len(self.snapshot_var) == 0:
                # add the snapshot op after the init
                sess = tf.get_default_session()
                with tf.variable_scope('snapshot'):
                    for var in self._parameters['tf_var_list']:
                        snap_var = tf.Variable(initial_value=sess.run(var),
                                               expected_shape=var.get_shape().as_list(),
                                               name=str(var.name).split(':')[0])
                        self.snapshot_var.append(snap_var)
                        self.save_snapshot_op.append(tf.assign(var, snap_var))
                        self.load_snapshot_op.append(tf.assign(snap_var, var))
            sess.run(tf.variables_initializer(var_list=self.snapshot_var))
            sess.run(self.save_snapshot_op)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep,
                                    var_list=self._parameters['tf_var_list'])

    @typechecked
    def return_tf_parameter_feed_dict(self) -> dict:
        # todo tuning or adaptive para setting here
        res = dict()
        for key, val in self._registered_tf_ph_dict.items():
            res[val] = self(key, require_true_value=True)
        return res

    def save_snapshot(self):
        sess = tf.get_default_session()
        if len(self.save_snapshot_op) == 0:
            with tf.variable_scope('snapshot'):
                for var in self._parameters['tf_var_list']:
                    snap_var = tf.Variable(initial_value=sess.run(var),
                                           expected_shape=var.get_shape().as_list(),
                                           name=var.name)
                    self.snapshot_var.append(snap_var)
                    self.save_snapshot_op.append(tf.assign(var, snap_var))
        sess.run(self.save_snapshot_op)

    def load_snapshot(self):
        sess = tf.get_default_session()
        if len(self.load_snapshot_op) == 0:
            with tf.variable_scope('snapshot'):
                for var in self._parameters['tf_var_list']:
                    snap_var = tf.Variable(initial_value=sess.run(var),
                                           expected_shape=var.get_shape().as_list(),
                                           name=var.name)
                    self.snapshot_var.append(snap_var)
                    self.load_snapshot_op.append(tf.assign(snap_var, var))
        sess.run(self.load_snapshot_op)

    def save_to_tf_model(self, sess, save_path, global_step):
        self.saver.save(sess=sess,
                        save_path=save_path + '/' + self.name + '/',
                        global_step=global_step)

    def load_from_tf_model(self, path_to_model, global_step):
        self.saver = tf.train.import_meta_graph(path_to_model + self.name + '/' + '-' + str(global_step) + '.meta')
        self.saver.recover_last_checkpoints(path_to_model + self.name + '/checkpoints')
        self.saver.restore(sess=tf.get_default_session(),
                           save_path=path_to_model + self.name + '/')

    def save_to_h5py(self, var_list, sess):
        raise NotImplementedError

    def load_from_h5py(self, path_to_h5py):
        raise NotImplementedError

    def __call__(self, key=None, require_true_value=False):
        if key in self._registered_tf_ph_dict.keys():
            if require_true_value is True:
                # todo debug here
                return super().__call__(key)
            else:
                return self._registered_tf_ph_dict[key]
        else:
            return super().__call__(key)

    def set(self, key, new_val):
        if not isinstance(new_val, type(self(key))):
            raise TypeError('new value of parameters {} should be type {} instead of {}'.format(key, type(self(key)),
                                                                                                type(new_val)))
        else:
            if key in self._parameters:
                self._parameters[key] = new_val
            else:
                self._source_config[key] = new_val

    def set_tf_var_list(self, tf_var_list: list):
        temp_var_list = list(set(tf_var_list))
        if len(temp_var_list) < len(tf_var_list):
            raise ValueError('Redundant tf variable in tf_var_list')
        for var in tf_var_list:
            assert isinstance(var, (tf.Tensor, tf.Variable))
        self._parameters['tf_var_list'] += tf_var_list

    @typechecked
    def to_tf_ph(self, key, ph: tf.Tensor):
        # call the parameters first to make sure it have a init value
        self(key)
        self._registered_tf_ph_dict[key] = ph

    @typechecked
    @overrides
    def copy_from(self, source_parameter: Parameters):
        tmp_op_list = []
        for t_para, s_para in zip(self._parameters['tf_var_list'], source_parameter._parameters['tf_var_list']):
            tmp_op_list.append(tf.assign(t_para, s_para))
        sess = tf.get_default_session()
        sess.run(tmp_op_list)
        del tmp_op_list

    def update(self, *args, **kwargs):
        # todo the adaptive strategy of params goes here
        self.set('LEARNING_RATE', new_val=self('LEARNING_RATE') * 0.99)
        pass

# class PlaceholderParameter(object):
#     """
#     Create a placeholder by passing into a init value, the shape and type will be inferred
#     """
#
#     def __init__(self, init_value, var_scope, name):
#         with tf.variable_scope(var_scope):
#             if isinstance(init_value, (float, int)) or np.isscalar(init_value):
#                 shape = [1]
#                 type = tf.float32
#             else:
#                 shape = np.array(init_value).shape
#                 np_type = np.array(init_value).dtype
