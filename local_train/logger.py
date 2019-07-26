from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class BaseLogger(ABC):
    @abstractmethod
    def log(self, optimizer):
        # raise NotImplementedError('optimization step is not implemented.')


class SimpleLogger(BaseLogger):
    def __init__(self):
        super(SimpleLogger, self).__init__()

    def log(self, optimizer):
        history = optimizer._history
        figure = plt.figure(figsize=(18, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history['func'])
        plt.grid()
        plt.ylabel("Loss", fontsize=19)
        plt.xlabel("iter", fontsize=19)
        plt.plot((movingaverage(losses, 50)), c='r')

        xs = np.array(history['x'])
        plt.subplot(1, 2, 2)
        for i in range(m_vals.shape[1]):
            plt.plot(xs[:, i], label=i)
        plt.grid()
        plt.ylabel("$\mu$", fontsize=19)
        plt.xlabel("iter", fontsize=19)
        plt.legend()

        return figure

class CometLogger(SimpleLogger):
    def __init__(self, experiment):
        super(CometLogger, self).__init__()
        self._experiment = experiment


    def log(self, optimizer):
        figure = super().log(optimizer=optimizer)
        experiment.log_figure("psi_dynamic", f, overwrite=True)
