from base_model import BaseConditionalGenerationOracle
from gan_model import Ge

class GANModel(BaseConditionalGenerationOracle):
    def __init__(self):
        self._generator =

    def loss(self, x, condition):
        return compute_loss(self._model, data=x, condition=condition)

    def fit(self, x, condition, epochs=400, lr=1e-3):
        trainable_parameters = list(self._model.parameters())
        optimizer = torch.optim.Adam(trainable_parameters, lr=lr)

        for epoch in range(epochs):
            loss = self.loss(x, condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def generate(self, condition):
        n = len(condition)
        z = torch.randn(n, self._dim_x).to(self.device)
        return self._sample_fn(z, condition)

    def log_density(self, x, condition):
        return self._density_fn(x, condition)