from base_model import BaseConditionalGenerationOracle
from RegressionNN.regression_nets import RegressionNet, RegressionLosses
import torch
import torch.utils.data as pytorch_data_utils
from tqdm import trange, tqdm

class RegressionModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 psi_dim: int,
                 y_dim: int,
                 x_dim: int,
                 batch_size: int,
                 epochs: int,
                 lr: float,
                 predict_risk: bool,
                 logger=None):
        super(RegressionModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size

        self._predict_risk = predict_risk
        self._output_dim = 1 if self._predict_risk else y_dim
        self._net = RegressionNet(self._output_dim, psi_dim, x_dim)
        self.logger = logger
        self._losses = RegressionLosses()
        self.attention_net = None

    def fit(self, y, condition, weights=None):
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)


        dataset = torch.utils.data.TensorDataset(y, condition)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self._batch_size,
                                                 shuffle=True)
        for epoch in trange(self._epochs):
            loss_history = []
            for y_batch, cond_batch in dataloader:
                y_gen = self.generate(cond_batch)

                loss = self._losses.loss(y_batch, y_gen)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_history.append(loss.item())

            if self.logger:
                if self.attention_net:
                    self.logger._experiment.log_metric("GAMMA", self.attention_net.gamma.item(), step=self.logger._epoch)
                self.logger.log_losses([loss_history])
                if not self._predict_risk:
                    self.logger.log_validation_metrics(self._y_model, y, condition, self,
                                                       (condition[:, :self._psi_dim].min(dim=0)[0].view(-1),
                                                        condition[:, :self._psi_dim].max(dim=0)[0].view(-1)),
                                                       batch_size=1000)
                self.logger.add_up_epoch()

    def generate(self, condition):
        if not self._predict_risk:
            return self._net(condition)
        else:
            raise NotImplementedError

    def log_density(self, y, condition):
        return None

    def loss(self):
        return None