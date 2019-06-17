def _get_copy_arg_with_tf_reuse(obj, kwargs: dict):
    # kwargs = deepcopy(kwargs)
    if 'reuse' in kwargs:
        if kwargs['reuse'] is True:
            if 'name_scope' in kwargs and kwargs['name_scope'] != obj.name_scope:
                raise ValueError('If reuse, the name scope should be same instead of : {} and {}'.format(
                    kwargs['name_scope'], obj.name_scope))
            else:
                kwargs.update(name_scope=obj.name_scope)
        else:
            if 'name_scope' in kwargs and kwargs['name_scope'] == obj.name:
                raise ValueError(
                    'If not reuse, the name scope should be different instead of: {} and {}'.format(
                        kwargs['name_scope'], obj.name_scope))
    return kwargs
