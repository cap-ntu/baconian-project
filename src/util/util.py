__all__ = [
    'type_check'
]


def type_check(o, t):
    if not isinstance(o, t):
        raise TypeError("Except type %s, but get %s" % (str(t), str(o.__class_)))
