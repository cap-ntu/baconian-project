class ExplorationStrategy(object):
    def __init__(self):
        self.parameters = None

    def predict(self, **kwargs):
        raise NotImplementedError
