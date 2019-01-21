import numpy as np
from transitions import Machine
from transitions import Machine


class Matter(object):
    def __init__(self):
        self.c = 0
        self.tc = 0

    def init(self):
        print("init!!!")

    def on_enter_training(self):
        print('into state!!!{}'.format('training'))
        self.c += 1
        print("train count {}".format(self.c))

    def on_exit_training(self):
        print('exit tate!!!{}'.format(self.state))

    def on_enter_testing(self):
        print('into state!!!{}'.format('testing'))

    def on_exit_testing(self):
        print('exit test!!!{}'.format(self.state))
        self.tc += 1
        print("test count {}".format(self.tc))

    def into_test(self):
        if self.c % 4 == 0:
            return True
        else:
            return False


def main():
    states = ['inited', 'training', 'testing']
    lump = Matter()
    transitions = [
        ['train', ['testing', 'inited', 'training'], 'training']
    ]
    machine = Machine(model=lump, states=states, initial='inited', transitions=transitions)
    machine.add_transition('test', ['training', 'inited'], 'testing', conditions='into_test')
    for i in range(10):
        print(i)
        lump.trigger('train')
        print(lump.state)
        lump.trigger('test')
        print(lump.state)


if __name__ == '__main__':
    main()
