_global_obj_arg_dict = {}
_global_name_dict = {}

assert id(_global_obj_arg_dict) == id(globals()['_global_obj_arg_dict']) == id(locals()['_global_obj_arg_dict'])
assert id(_global_name_dict) == id(globals()['_global_name_dict']) == id(locals()['_global_name_dict'])


def reset_all():
    globals()['_global_obj_arg_dict'] = {}
    globals()['_global_name_dict'] = {}


def reset(key: str):
    globals()[key] = {}


def get_all() -> dict:
    return dict(
        _global_obj_arg_dict=globals()['_global_obj_arg_dict'],
        _global_name_dict=globals()['_global_name_dict']
    )

