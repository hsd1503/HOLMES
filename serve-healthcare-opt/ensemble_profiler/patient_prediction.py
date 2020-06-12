class PytorchPredictorECG:
    """
    Data is fired every 8 ms (125Hz).
    But the prediction is made every 30 seconds.
    The data is saved uptill 30 seconds.
    Args:
        model(torch.nn.Module): a pytorch model for prediction.
        cuda(bool): to use_gpu or not.
    """

    def __init__(self, model, cuda=False):
        self.cuda = cuda
        self.model = model
        if cuda:
            self.model = self.model.cuda()

    def __call__(self, flask_request, data):
        if self.cuda:
            data = data.cuda()
        # do the prediction
        result = self.model(data)
        return result.data.cpu().numpy().argmax().item()
