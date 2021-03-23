class ValueNetwork:
    def __init__(self, model_path):
        if model_path is not None:
            # Here load model from file
            raise NotImplementedError
        else:
            # Here construct model
            raise NotImplementedError

    def predict_value(self, state):
        raise NotImplementedError