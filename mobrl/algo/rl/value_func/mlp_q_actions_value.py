# import tensorflow as tf
#
# from mobrl.core.parameters import Parameters
# from mobrl.envs.env_spec import EnvSpec
#
# """
# to be done in the v0.2 maybe
# """
#
#
# class MLPQValueOnActions(PlaceholderInputValueFunction):
#     # todo get the code from intel tutor dqn for reference
#     def __init__(self, name: str, env_spec: EnvSpec, parameters: Parameters = None, input: tf.Tensor = None):
#         super().__init__(name=name, env_spec=env_spec, parameters=parameters, input=input)
#
#     def copy(self, obj) -> bool:
#         return super().copy(obj)
#
#     def forward(self, *args, **kwargs):
#         pass
#
#     def update(self, *args, **kwargs):
#         pass
