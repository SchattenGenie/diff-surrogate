from base_model import BaseConditionalGenerationOracle
from gan_nets import Generator, Discriminator
from gan_nets import GANLosses
import torch
import torch.utils.data as pytorch_data_utils


class GANModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 noise_dim: int,
                 psi_dim: int,
                 y_dim: int,
                 x_dim: int,
                 batch_size: int,
                 task: str,
                 epochs: int,
                 lr: float,
                 iters_discriminator: int,
                 iters_generator: int,
                 grad_penalty: bool = False,
                 zero_centered_grad_penalty: bool = False,
                 instance_noise_std: float = None,
                 logger=None):
        super(GANModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        if task == 'WASSERSTEIN':
            wasserstein = True
        else:
            wasserstein = False
        self._noise_dim = noise_dim
        self._psi_dim = psi_dim
        self._y_dim = y_dim
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
                                    out_dim=self._y_dim,
                                    psi_dim=self._psi_dim)
        self._discriminator = Discriminator(in_dim=self._y_dim,
                                            wasserstein=wasserstein,
                                            psi_dim=self._psi_dim)
        self.logger = logger

    @staticmethod
    def instance_noise(data, std):
        device = data.device
        return data + torch.distributions.Normal(0, std).sample(data.shape).to(device)

    def loss(self, y, condition):
        return self._discriminator(y, condition)

    def fit(self, y, condition):
        g_optimizer = torch.optim.Adam(self._generator.parameters(), lr=self._lr, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self._discriminator.parameters(), lr=self._lr, betas=(0.5, 0.999))

        dataset = torch.utils.data.TensorDataset(y, condition)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self._batch_size,
                                                 shuffle=True)

        for epoch in range(self._epochs):
            dis_epoch_loss = []
            gen_epoch_loss = []
            for y_batch, cond_batch in dataloader:
                for _ in range(self._iters_discriminator):
                    y_gen = self.generate(condition=cond_batch)
                    if self._instance_noise_std:
                        y_batch = self.instance_noise(y_batch, self._instance_noise_std)
                        y_gen = self.instance_noise(y_gen, self._instance_noise_std)
                    loss = self._ganloss.d_loss(self.loss(y_gen, cond_batch),
                                                self.loss(y_batch, cond_batch))
                    if self._grad_penalty:
                        loss += self._ganloss.calc_gradient_penalty(self._discriminator,
                                                                    y_gen.data,
                                                                    y_batch.data,
                                                                    condition.data)
                    if self._zero_centered_grad_penalty:
                        loss -= self._ganloss.calc_zero_centered_GP(self._discriminator,
                                                                    x_gen.data,
                                                                    y_batch.data,
                                                                    condition.data)

                    d_optimizer.zero_grad()
                    loss.backward()
                    d_optimizer.step()
                dis_epoch_loss.append(loss.item())

                for _ in range(self._iters_generator):
                    y_gen = self.generate(cond_batch)
                    if self._instance_noise_std:
                        y_batch = self.instance_noise(y_batch, self._instance_noise_std)
                    loss = self._ganloss.g_loss(self.loss(y_gen, cond_batch))
                    g_optimizer.zero_grad()
                    loss.backward()
                    g_optimizer.step()
                gen_epoch_loss.append(loss.item())

            if self.logger:
                self.logger.log_losses([dis_epoch_loss, gen_epoch_loss])

        return self

    def generate(self, condition):
        n = len(condition)
        z = torch.randn(n, self._noise_dim).to(self.device)
        return self._generator(z, condition)

    def log_density(self, y, condition):
        return None
