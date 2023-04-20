import torch.optim
import torch.utils.data as torch_data
from typing import Dict
from train_eval.initialization import initialize_adms_model,\
    initialize_adms_dataset, initialize_metric
import torch
import time
import math
import os
import train_eval.utils as u
from datasets.adms_dataset import adms_collate
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')
import global_var

global_var._init()

# Initialize device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """
    Trainer class for running train loops
    """

    def __init__(self, cfg: Dict, data_root: str, data_dir: str, checkpoint_path=None, just_weights=False, writer=None):
        """
        Initialize trainer object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        :param just_weights: Load just weights from checkpoint
        :param writer: Tensorboard summary writer
        """

        # TODO:Initialize datasets use data_root, data_dir:
        dataset = initialize_adms_dataset(cfg['dataset'], cfg['datafile'])
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_len = 1000
        val_len = 100
        if dataset_size < train_len + val_len:
            train_indices, val_indices = indices[split:], indices[:split]
        else:
            train_indices, val_indices = indices[:train_len], indices[train_len:train_len + val_len]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Initialize data loaders. for train, shuffle is true.
        self.tr_dl = torch_data.DataLoader(
            dataset,
            cfg['batch_size'],
            #    True,
            num_workers=cfg['num_workers'],
            sampler=train_sampler,
            drop_last=True,
            collate_fn=adms_collate)
        # batch size 1 for eval
        self.val_dl = torch_data.DataLoader(
            dataset,
            1,
            # False,
            num_workers=cfg['num_workers'],
            sampler=valid_sampler,
            drop_last=True,
            collate_fn=adms_collate)

        print("Train dataset length: ", len(self.tr_dl) * cfg['batch_size'], ", val dataset length: ", len(self.val_dl))

        # Initialize model
        self.model = initialize_adms_model(cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
                                           cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.model = self.model.float().to(device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['optim_args']['lr'])

        # Initialize epochs
        self.current_epoch = 0

        # TODO:Initialize losses
        self.losses = [initialize_metric(cfg['losses'][i], cfg['loss_args'][i]) for i in range(len(cfg['losses']))]
        self.loss_weights = cfg['loss_weights']
        #
        # TODO:Initialize metrics
        self.train_metrics = [
            initialize_metric(cfg['tr_metrics'][i], cfg['tr_metric_args'][i]) for i in range(len(cfg['tr_metrics']))
        ]
        self.val_metrics = [
            initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i]) for i in range(len(cfg['val_metrics']))
        ]
        self.val_metric = math.inf
        self.min_val_metric = math.inf

        # Print metrics after these many mini-batches to keep track of training
        self.log_period = len(self.tr_dl) // cfg['log_freq']
        if not self.log_period:
            self.log_period = 1

        # Initialize tensorboard writer
        self.writer = writer
        self.tb_iters = 0
        self.has_add_graph = False

        # Load checkpoint if checkpoint path is provided
        if checkpoint_path is not None:
            print()
            print("Loading checkpoint from " + checkpoint_path + " ...", end=" ")
            self.load_checkpoint(checkpoint_path, just_weights=just_weights)
            print("Done")

    def train(self, num_epochs: int, output_dir: str):
        """
        Main function to train model
        :param num_epochs: Number of epochs to run training for
        :param output_dir: Output directory to store tensorboard logs and checkpoints
        :return:
        """

        # Run training, validation for given number of epochs
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, start_epoch + num_epochs):

            # Set current epoch
            self.current_epoch = epoch
            print()
            print('Epoch (' + str(self.current_epoch + 1) + '/' + str(start_epoch + num_epochs) + ')')

            # self.learning_rate_decay(epoch, self.optimizer)

            # Train
            train_epoch_metrics, _, _ = self.run_epoch('train', self.tr_dl)
            self.print_metrics(train_epoch_metrics, self.tr_dl, mode='train')
            self.log_tensorboard_metrics(train_epoch_metrics, mode='train')

            # Validate
            with torch.no_grad():
                val_epoch_metrics, prediction, gt = self.run_epoch('val', self.val_dl)
            self.print_metrics(val_epoch_metrics, self.val_dl, mode='val')
            self.log_tensorboard_metrics(val_epoch_metrics, mode='val')

            # Update validation metric using first metric
            self.val_metric = val_epoch_metrics[str(self.val_metrics[0])[:-2]] / val_epoch_metrics['minibatch_count']
            # self.val_metric = val_epoch_metrics['rel'] / val_epoch_metrics['minibatch_count']

            # save best checkpoint when applicable
            if self.val_metric < self.min_val_metric:
                self.min_val_metric = self.val_metric
                self.save_checkpoint(os.path.join(output_dir, 'checkpoints', 'best.tar'))
                self.save_model(os.path.join(output_dir, 'saved_model', 'ori_best_adms_model.pth'))
                np.save(os.path.join(output_dir, 'saved_model', 'ori_prediction.npy'), prediction)
                np.save(os.path.join(output_dir, 'saved_model', 'ori_gt.npy'), gt)
                # np.save(os.path.join(output_dir, 'saved_model', 'per5_' + str(self.val_metric)), gt)

            # Save checkpoint every epoch.
            self.save_checkpoint(os.path.join(output_dir, 'checkpoints', str(self.current_epoch) + '.tar'))

            # tensorboard global step
            self.tb_iters += 1

        # self.save_model(os.path.join(output_dir, 'saved_model', 'adms_model_' + time.strftime("%Y%m%d_%H%M%S") + '.pth'))
        # self.save_model(os.path.join(output_dir, 'saved_model', 'adms_model.pth'))

    def run_epoch(self, mode: str, dl: torch_data.DataLoader):
        """
        Runs an epoch for a given dataloader
        :param mode: 'train' or 'val'
        :param dl: Dataloader object
        """
        if mode == 'val':
            self.model.eval()
        else:
            self.model.train()

        # Initialize epoch metrics
        epoch_metrics = self.initialize_metrics_for_epoch(mode)

        # Main loop
        st_time = time.time()

        # cache prediction and gt of val
        val_len = len(self.val_dl)
        prediction_val = np.empty(val_len)
        label_val = np.empty(val_len)

        for i, data in enumerate(dl):
            # Load data
            data = u.send_to_device(u.convert_double_to_float(data))

            (veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features, polylines, polynum,
             attention_mask) = self.model.preprocess_inputs(data['inputs'])

            # if mode == 'val':
            #     polylines_expand = torch.zeros([256, 19, 128], device=polylines.device)
            #     mask_expand = torch.ones([256, 19, 64],  device=attention_mask.device)*(-10000.0)
            #     polylines_expand[:polynum[0],:,:] = polylines
            #     mask_expand[:polynum[0],:,:] = attention_mask
            #     polylines = polylines_expand
            #     attention_mask = mask_expand
            # Forward pass
            predictions = self.model(veh_cate_features, veh_dense_features, driver_cate_features, driver_dense_features,
                                     polylines, polynum, attention_mask)

            # # visualize model
            # if not self.has_add_graph:
            #     self.writer.add_graph(self.model, (veh_cate_features, veh_dense_features, driver_cate_features,
            #                                        driver_dense_features, polylines, polynum, attention_mask))
            #     self.has_add_graph = True

            # Compute loss and backpropagation if training
            if mode == 'train':
                loss = self.compute_loss(predictions, data['label'])
                # loss = torch.div(torch.abs(predictions - data['label']), data['label']).mean()
                self.back_prop(loss)
                # if i == 0:
                #     pre_np = predictions.detach().cpu().numpy().squeeze()
                #     label_np = data['label'].detach().cpu().numpy().squeeze()
                #     self.log_tensorboard_fig(pre_np, label_np, mode)

                #     # debug
                #     fig, ax = plt.subplots()
                #     x = range(128)
                #     # plt.scatter(x, global_var.get_value('veh_vector')[0], color="r", label="veh_vector")
                #     plt.scatter(x, global_var.get_value('driver_vector')[0], color="b", label="driver_vector")
                #     plt.scatter(x, global_var.get_value('global_vector')[0], color="y", label="global_vector")
                #     plt.legend()
                #     self.writer.add_figure(mode + "/vectors.fig", fig, self.tb_iters)
                #     # fig, ax = plt.subplots()
                #     # for i in range(2,debug_vec.shape[1]):
                #     #     plt.scatter(x, debug_vec[0][i])
                #     # self.writer.add_figure(mode + "/obj_vectors.fig", fig, self.tb_iters)

                #     # debug input
                #     # self.log_tensorboard_input_data(data)

            elif mode == "val":
                prediction_val[i] = predictions.detach().cpu().numpy().squeeze()
                label_val[i] = data['label'].detach().cpu().numpy().squeeze()

            # Keep time
            minibatch_time = time.time() - st_time
            st_time = time.time()

            # Aggregate metrics
            minibatch_metrics, epoch_metrics = self.aggregate_metrics(epoch_metrics, minibatch_time, predictions, data['label'],
                                                                      mode)

            # Display metrics at a predefined frequency
            if i % self.log_period == self.log_period - 1 and len(dl) - i != 1:
                self.print_metrics(epoch_metrics, dl, mode)

        # # log val fig
        # val_fig_len = 64
        # # if (len(dl) < val_fig_len):
        # prediction_val_fig = prediction_val[:val_fig_len]
        # label_val_fig = label_val[:val_fig_len]
        # self.log_tensorboard_fig(prediction_val_fig, label_val_fig, "val")
        return epoch_metrics, prediction_val, label_val

    def learning_rate_decay(self, i_epoch, optimizer):
        if i_epoch > 20 and i_epoch % 50 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5

    def compute_loss(self, model_outputs: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Computes loss given model outputs and ground truth labels
        """
        # TODO: Implement the calculation of the multi-loss
        loss_vals = [loss(model_outputs, ground_truth) for loss in self.losses]
        total_loss = torch.tensor(0).float().to(device)
        for n in range(len(loss_vals)):
            total_loss += self.loss_weights[n] * loss_vals[n]

        return total_loss

    def compute_metric(self, model_outputs: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Computes loss given model outputs and ground truth labels
        """
        # TODO: Implement the calculation of different train or val metric
        raise NotImplementedError()

    def back_prop(self, loss: torch.Tensor, grad_clip_thresh=10):
        """
        Backpropagation loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def initialize_metrics_for_epoch(self, mode: str):
        """
        Initialize metrics for epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        epoch_metrics = {'minibatch_count': 0, 'time_cost': 0}
        for metric in metrics:
            epoch_metrics[str(metric)[:-2]] = 0

        return epoch_metrics

    def aggregate_metrics(self, epoch_metrics: Dict, minibatch_time: float, model_outputs: torch.Tensor,
                          ground_truth: torch.Tensor, mode: str):
        """
        Aggregates metrics by minibatch for the entire epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics

        minibatch_metrics = {}

        # TODO: Compute different metrics
        for metric in metrics:
            minibatch_metrics[str(metric)[:-2]] = metric(model_outputs, ground_truth).item()

        epoch_metrics['minibatch_count'] += 1
        epoch_metrics['time_cost'] += minibatch_time
        for metric in metrics:
            epoch_metrics[str(metric)[:-2]] += minibatch_metrics[str(metric)[:-2]]

        # tmp for debug
        minibatch_metrics['rel'] = torch.div(torch.abs(model_outputs - ground_truth), ground_truth).mean().item()
        if 'rel' not in epoch_metrics.keys():
            epoch_metrics['rel'] = 0
        epoch_metrics['rel'] += minibatch_metrics['rel']

        return minibatch_metrics, epoch_metrics

    def print_metrics(self, epoch_metrics: Dict, dl: torch_data.DataLoader, mode: str):
        """
        Prints aggregated metrics
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        minibatches_left = len(dl) - epoch_metrics['minibatch_count']
        eta = (epoch_metrics['time_cost'] / epoch_metrics['minibatch_count']) * minibatches_left
        epoch_progress = int(epoch_metrics['minibatch_count'] / len(dl) * 100)

        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print('\rTraining:  ' if mode == 'train' else '\rValidating:', end=" ")
        print(progress_bar, str(epoch_progress) if epoch_progress == 100 else (" " + str(epoch_progress)), '%', end=", ")
        print('ETA:', int(eta), end="s, ")
        print('Metrics', end=": { ")
        for metric in metrics:
            metric_val = epoch_metrics[str(metric)[:-2]] / epoch_metrics['minibatch_count']
            print(str(metric)[:-2] + ':', format(metric_val, '.2f'), end=", ")
        print('\b\b }     ', end="\n" if minibatches_left == 0 else "")

    def load_checkpoint(self, checkpoint_path, just_weights=False):
        """
        Loads checkpoint from given path
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not just_weights:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.val_metric = checkpoint['val_metric']
            self.min_val_metric = checkpoint['min_val_metric']

    def save_checkpoint(self, checkpoint_path):
        """
        Saves checkpoint to given path
        """
        torch.save(
            {
                'epoch': self.current_epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_metric': self.val_metric,
                'min_val_metric': self.min_val_metric
            }, checkpoint_path)

    def save_model(self, model_path):
        """
        Saves checkpoint to given path
        """
        torch.save(self.model, model_path)

    def log_tensorboard_train(self, minibatch_metrics: Dict):
        """
        Logs mini-batch metrics during training
        """
        for metric_name, metric_val in minibatch_metrics.items():
            self.writer.add_scalar('train/' + metric_name, metric_val, self.tb_iters)
        self.tb_iters += 1

    def log_tensorboard_val(self, epoch_metrics: Dict):
        """
        Logs epoch metrics for validation set
        """
        for metric_name, metric_val in epoch_metrics.items():
            if metric_name != 'minibatch_count' and metric_name != 'time_cost':
                res = metric_val / epoch_metrics['minibatch_count']
                self.writer.add_scalar('val/' + metric_name, res, self.tb_iters)

    def log_tensorboard_metrics(self, epoch_metrics: Dict, mode: str):
        """
        Logs epoch metrics for validation set
        """
        for metric_name, metric_value in epoch_metrics.items():
            if metric_name != 'minibatch_count' and metric_name != 'time_cost':
                res = metric_value / epoch_metrics['minibatch_count']
                self.writer.add_scalar(mode + "/" + metric_name, res, self.tb_iters)

    def log_tensorboard_fig(self, predictions: np.array, labels: np.array, mode: str):
        fig, ax = plt.subplots()
        x = range(len(predictions))
        for i in range(len(predictions)):
            plt.plot([i, i], [predictions[i], labels[i]], color="blue")
        plt.scatter(x, predictions, color="r", label="prediction")
        plt.scatter(x, labels, color='black', label="gt")
        plt.legend()
        self.writer.add_figure(mode + "/pred_label.fig", fig, self.tb_iters)

    def log_tensorboard_input_data(self, data):

        veh_cate_features = data['inputs']['veh_cate_features'].detach().cpu().numpy()
        veh_dense_features = data['inputs']['veh_dense_features'].detach().cpu().numpy()
        driver_cate_features = data['inputs']['driver_cate_features'].detach().cpu().numpy()
        driver_dense_features = data['inputs']['driver_dense_features'].detach().cpu().numpy()
        polylines = data['inputs']['polylines'].detach().cpu().numpy()
        attention_mask = data['inputs']['attention_mask'].detach().cpu().numpy()
        polynum = data['inputs']['polynum'].detach().cpu().numpy()
        # label = data["label"]

        # obj traj
        fig, ax = plt.subplots()
        # for i in range(19):
        for i in range(polynum[0]):
            plt.plot(polylines[i][:][0], polylines[i][:][1])
        # plt.savefig("output/test.png")
        self.writer.add_figure("train" + "/obj_traj.fig", fig, self.tb_iters)
