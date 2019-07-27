from base_model import BaseConditionalGenerationOracle
from gan_nets import Generator, Discriminator
from gan_nets import GANLosses
import torch
import torch.utils.data as pytorch_data_utils


class GANModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 noise_dim,
                 psi_dim,
                 x_dim,
                 batch_size=64,
                 task="KL",
                 grad_penalty=False,
                 zero_centered_grad_penalty=False,
                 instance_noise_std=None,
                 iters_discriminator=1,
                 iters_generator=5,
                 epochs=5,
                 lr=1e-3,
                 wasserstein=True):
        super(BaseConditionalGenerationOracle, self).__init__()
        self._noise_dim = noise_dim
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._grad_penalty = grad_penalty
        self._zero_centered_grad_penalty = zero_centered_grad_penalty
        self._instance_noise_std = instance_noise_std
        self._iters_discriminator = iters_discriminator
        self._iters_generator = iters_generator
        self._ganloss = GANLosses(task=task)
        self._generator = Generator(noise_dim=self._noise_dim,
                                    out_dim=self._x_dim)
        self._discriminator = Discriminator(in_dim=self._x_dim,
                                            wasserstein=wasserstein)

    @staticmethod
    def instance_noise(data, std):
        device = data.device
        return data + torch.distributions.Normal(0, std).sample(data.shape).to(device)

    def loss(self, x, condition):
        return self._discriminator(x, condition)

    def fit(self, x, condition):
        g_optimizer = torch.optim.Adam(self._generator.parameters(), lr=self._lr, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self._discriminator.parameters(), lr=self._lr, betas=(0.5, 0.999))

        dataset = torch.utils.data.TensorDataset(x, condition)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self._batch_size,
                                                 shuffle=True)

        for epoch in range(self._epochs):
            for x_batch, cond_batch in dataloader:
                for _ in range(self._iters_discriminator):
                    x_gen = self.generate(condition=cond_batch)
                    if self._instance_noise_std:
                        x_batch = self.instance_noise(x_batch, self._instance_noise_std)
                        x_gen = self.instance_noise(x_gen, self._instance_noise_std)
                    loss = self._ganloss.d_loss(self.loss(x_gen, cond_batch),
                                                self.loss(x_batch, cond_batch))
                    if self._grad_penalty:
                        loss += self._ganloss.calc_gradient_penalty(self._discriminator,
                                                                    x_gen.data,
                                                                    x_batch.data,
                                                                    condition.data)
                    if self._zero_centered_grad_penalty:
                        loss -= self._ganloss.calc_zero_centered_GP(self._discriminator,
                                                                    x_gen.data,
                                                                    x_batch.data,
                                                                    condition.data)

                    d_optimizer.zero_grad()
                    loss.backward()
                    d_optimizer.step()

                for _ in range(self._iters_generator):
                    x_gen = self.generate(cond_batch)
                    if self._instance_noise_std:
                        x_batch = self.instance_noise(x_batch, self._instance_noise_std)
                    loss = self._ganloss.g_loss(self.loss(x_gen, cond_batch))
                    g_optimizer.zero_grad()
                    loss.backward()
                    g_optimizer.step()

        return self

    def generate(self, condition):
        n = len(condition)
        z = torch.randn(n, self._noise_dim).to(self.device)
        return self._generator(z, condition)

    def log_density(self, x, condition):
        return None