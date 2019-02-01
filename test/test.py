class Foo(object):
    def __getitem__(self, item):
        return item * 2


if __name__ == '__main__':
    a = Foo()
    print(a[2])
