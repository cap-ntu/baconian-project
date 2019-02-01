import tensorflow as tf
from src.core.parameters import Parameters
from src.core.global_config import GlobalConfig
from overrides.overrides import overrides
from typeguard import typechecked


class TensorflowParameters(Parameters):

    @typechecked
    def __init__(self, tf_var_list: list, rest_parameters: dict, name: str,
                 max_to_keep=GlobalConfig.DEFAULT_MAX_TF_SAVER_KEEP,
                 auto_init=False,
                 source_config=None,
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

        if require_snapshot is True:
            sess = tf.get_default_session()
            with tf.variable_scope('snapshot'):
                for var in self._parameters['tf_var_list']:
                    snap_var = tf.Variable(initial_value=sess.run(var),
                                           expected_shape=var.get_shape().as_list(),
                                           name=var.name)
                    self.snapshot_var.append(snap_var)
                    self.save_snapshot_op.append(tf.assign(var, snap_var))
                    self.load_snapshot_op.append(tf.assign(snap_var, var))
        if auto_init is True:
            self.init()

    @overrides
    def init(self):
        sess = tf.get_default_session()
        sess.run(tf.variables_initializer(var_list=self._parameters['tf_var_list']))
        if self.require_snapshot is True:
            sess.run(tf.variables_initializer(var_list=self.snapshot_var))
            sess.run(self.save_snapshot_op)
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep,
                                    var_list=self._parameters['tf_var_list'])

    @typechecked
    def return_tf_parameter_feed_dict(self) -> dict:
        # todo tuning or adaptive para setting here
        return dict()

    def save_snapshot(self):
        sess = tf.get_default_session()
        if len(self.save_snapshot_op) == 0:
            with tf.variable_scope('snapshot'):
                for var in self._parameters:
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
                for var in self._parameters:
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

    def __call__(self, key=None):
        return super().__call__(key)

    def set_tf_var_list(self, tf_var_list: list):
        for var in tf_var_list:
            assert isinstance(var, (tf.Tensor, tf.Variable))
        self._parameters['tf_var_list'] = tf_var_list

    @typechecked
    @overrides
    def copy_from(self, source_parameter: Parameters):
        tmp_op_list = []
        for t_para, s_para in zip(self._parameters['tf_var_list'], source_parameter._parameters['tf_var_list']):
            tmp_op_list.append(tf.assign(t_para, s_para))
        sess = tf.get_default_session()
        sess.run(tmp_op_list)
        del tmp_op_list
