import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = updates

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000)))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def state_dict(self):
        return self.ema.state_dict()