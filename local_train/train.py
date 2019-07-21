import os
import torch
from pyro import distributions as dist
import torch.optim as optim
from utils import sample_noise, iterate_minibatches, generate_local_data
import numpy as np
from gan.metrics import Metrics


metric_calc = Metrics((-10, 10), 100)


class GANTrainingUtils(object):
    def __init__(self, GANLosses,
                 task, device, hyper_params, experiment, y_sampler, path, instance_noise):
        self.GANLosses = GANLosses
        self.TASK = task
        self.device = device
        self.hyper_params = hyper_params
        self.experiment = experiment
        self.y_sampler = y_sampler
        self.PATH = path
        self.INSTANCE_NOISE = instance_noise

    def calculate_validation_metrics(self, data, inputs, train_fixed_noise, generator, mu_range, epoch, points_size=100, sample_size=2000):
        js = []
        ks = []

        if (epoch + 0) % 5 == 0:
            mu = dist.Uniform(*mu_range).sample([points_size])
            x = self.y_sampler.x_dist.sample([points_size, self.hyper_params['x_dim']])
            print(mu.shape, x.shape)
            inputs_mu_x = torch.cat([mu, x], dim=1).to(self.device)
            for index in range(points_size):
                noise = torch.Tensor(sample_noise(sample_size, self.hyper_params['NOISE_DIM'])).to(self.device)
                sample_inputs = inputs_mu_x[index, :].reshape(1, -1).repeat([sample_size, 1])
                gen_samples = generator(noise, sample_inputs).detach().cpu()

                self.y_sampler.make_condition_sample({'mu': sample_inputs[:, :self.hyper_params["mu_dim"]],
                                                 'X': sample_inputs[:, self.hyper_params["mu_dim"]:]})
                true_samples = self.y_sampler.condition_sample().cpu()
                js.append(metric_calc.compute_JS(true_samples, gen_samples).item())
                ks.append(metric_calc.compute_KSStat(true_samples.numpy(), gen_samples.numpy()).item())
            self.experiment.log_metric("average_mu_JS", np.mean(js), step=epoch)
            self.experiment.log_metric("average_mu_KS", np.mean(ks), step=epoch)

        train_data_js = metric_calc.compute_JS(data.cpu(), generator(train_fixed_noise, inputs).detach().cpu())
        train_data_ks = metric_calc.compute_KSStat(data.cpu().numpy(),
                                                   generator(train_fixed_noise, inputs).detach().cpu().numpy())

        self.experiment.log_metric("train_data_JS", train_data_js, step=epoch)
        self.experiment.log_metric("train_data_KS", train_data_ks, step=epoch)
        for order in range(1, 4):
            moment_of_true = metric_calc.compute_moment(data.cpu(), order)
            moment_of_generated = metric_calc.compute_moment(generator(train_fixed_noise, inputs).detach().cpu(),
                                                     order)
            metric_diff = moment_of_true - moment_of_generated

            self.experiment.log_metric("train_data_diff_order_" + str(order), metric_diff)
            self.experiment.log_metric("train_data_gen_order_" + str(order), moment_of_generated)

    def train_gan(self, generator, discriminator, current_psi, total_epoch_counter):
        # ===========================
        # IMPORTANT PARAMETER:
        # Number of D updates per G update
        # ===========================
        k_d, k_g = self.hyper_params["n_d_train"], 1
        step = 0.5
        random_step_std = 1

        g_optimizer = optim.Adam(generator.parameters(), lr=self.hyper_params['learning_rate'], betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=self.hyper_params['learning_rate'], betas=(0.5, 0.999))
        gan_losses = self.GANLosses(self.TASK, self.device)

        data, inputs = generate_local_data(self.y_sampler, self.device,
                                           n_samples_per_dim=self.hyper_params["n_samples_per_dim"],
                                           step=step, current_psi=
                                           current_psi, x_dim=1, std=random_step_std)

        train_fixed_noise = torch.Tensor(sample_noise(data.shape[0], self.hyper_params['NOISE_DIM'])).to(self.device)

        for epoch in range(self.hyper_params['num_epochs']):
            dis_epoch_loss = []
            gen_epoch_loss = []

            for input_data, inputs_batch in iterate_minibatches(data, self.hyper_params['batch_size'], y=inputs):
                # Optimize D
                for _ in range(k_d):
                    # Sample noise
                    noise = torch.Tensor(sample_noise(len(input_data), self.hyper_params['NOISE_DIM'])).to(self.device)


                    # Do an update
                    data_gen = generator(noise, inputs_batch)

                    if self.INSTANCE_NOISE:
                        inp_data += torch.distributions.Normal(0,self.hyper_params['INST_NOISE_STD']).\
                                    sample(inp_data.shape).to(self.device)
                        data_gen += torch.distributions.Normal(0, self.hyper_params['INST_NOISE_STD']).\
                                    sample(data_gen.shape).to(self.device)

                    loss = gan_losses.d_loss(discriminator(data_gen, inputs_batch),
                                             discriminator(input_data, inputs_batch))
                    if self.TASK == 4:
                        grad_penalty = gan_losses.calc_gradient_penalty(discriminator,
                                                                        data_gen.data,
                                                                        inputs_batch.data,
                                                                        input_data.data)
                        loss += grad_penalty

                    if self.TASK == 5:
                        grad_penalty = gan_losses.calc_zero_centered_GP(discriminator,
                                                                        data_gen.data,
                                                                        inputs_batch.data,
                                                                        inp_data.data)
                        loss -= grad_penalty

                    d_optimizer.zero_grad()
                    loss.backward()
                    d_optimizer.step()

                    if self.TASK == 3:
                        for p in discriminator.parameters():
                            p.data.clamp_(clamp_lower, clamp_upper)
                dis_epoch_loss.append(loss.item())

                # Optimize G
                for _ in range(k_g):
                    # Sample noise
                    noise = torch.Tensor(sample_noise(len(input_data), self.hyper_params['NOISE_DIM'])).to(self.device)

                    # Do an update
                    data_gen = generator(noise, inputs_batch)
                    if self.INSTANCE_NOISE:
                        data_gen += torch.distributions.Normal(0, self.hyper_params['INST_NOISE_STD']).\
                                    sample(data_gen.shape).to(self.device)
                    loss = gan_losses.g_loss(discriminator(data_gen, inputs_batch))
                    g_optimizer.zero_grad()
                    loss.backward()
                    g_optimizer.step()
                gen_epoch_loss.append(loss.item())

            self.experiment.log_metric("d_loss", np.mean(dis_epoch_loss), step=total_epoch_counter[0])
            self.experiment.log_metric("g_loss", np.mean(gen_epoch_loss), step=total_epoch_counter[0])

            self.calculate_validation_metrics(data, inputs, train_fixed_noise, generator, (current_psi.view(-1) - step, current_psi.view(-1) + step), total_epoch_counter[0])
            total_epoch_counter[0] += 1

            if epoch % 20 == 0:
            #     mu_range = self.hyper_params["mu_range"]
            #     f = dist_plotter.draw_conditional_samples(mu_range)
            #     self.experiment.log_figure("conditional_samples_{}".format(epoch), f)
            #     plt.closmu_rangee(f)
            #
            #     mu_range = self.hyper_params["mu_range"]
            #     f = dist_plotter.draw_mu_samples(mu_range)
            #     self.experiment.log_figure("mu_samples_{}".format(epoch), f)
            #     plt.close(f)
            #
            #     x_range = (-10,10)
            #     f = dist_plotter.draw_X_samples(x_range)
            #     self.experiment.log_figure("x_samples_{}".format(epoch), f)
            #     plt.close(f)

    #                     f, g = dist_plotter.plot_means_diff(self.hyper_params["mu_range"], x_range)
    #                     self.experiment.log_figure("mean_diff_x_{}".format(epoch), f)
    #                     self.experiment.log_figure("mean_diff_{}".format(epoch), g)
    #                     plt.close(f)
    #                     plt.close(g)

    #                     f = dist_plotter.draw_mu_2d_samples(self.hyper_params["mu_range"])
    #                     self.experiment.log_figure("mu_samples_2d_{}".format(epoch), f)
    #                     plt.close(f)

                snapshot_path = os.path.join(self.PATH, "{}.tar".format(epoch))
                torch.save({
                    'gen_state_dict': generator.state_dict(),
                    'dis_state_dict': discriminator.state_dict(),
                    'genopt_state_dict': g_optimizer.state_dict(),
                    'disopt_state_dict': d_optimizer.state_dict(),
                    'epoch': epoch
                    }, snapshot_path)
                self.experiment.log_asset(snapshot_path, overwrite=True)
