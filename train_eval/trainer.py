import time
import math
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from typing import Dict
from tqdm import tqdm as tqdm_
from functools import partial

matplotlib.use('Agg')

import torch
import torch.optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as torch_data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler

import train_eval.utils as u
from train_eval import logger
from datasets.adms_dataset import adms_collate
from train_eval.initialization import initialize_adms_model,\
    initialize_adms_dataset, initialize_metric
from train_eval.utils import batch_list_to_batch_tensors, is_main_device
import global_var

global_var._init()
tqdm = partial(tqdm_, dynamic_ncols=True)

# Initialize device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """
    Trainer class for running train loops
    """
    def __init__(self,
                 cfg: Dict,
                 data_root: str,
                 data_dir: str,
                 checkpoint_path=None,
                 just_weights=False,
                 writer=None):
        """
        Initialize trainer object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        :param just_weights: Load just weights from checkpoint
        :param writer: Tensorboard summary writer
        """

        # Init params
        self.distribute_training: int = cfg['distribute_training']
        self.optim_args: Dict = cfg['optim_args']
        self.batch_size: int = cfg['batch_size']
        self.world_size: int = cfg['world_size']
        self.output_dir = 'output/test_av/'
        self.num_epoch = 200

        # Initialize datasets:
        self.dataset = initialize_adms_dataset(cfg['dataset'], cfg['datafile'])

        # Initialize model
        self.model = initialize_adms_model(
            cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
            cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.model = self.model.float()

        # Initialize epochs
        self.current_epoch = 0

        self.losses = [
            initialize_metric(cfg['losses'][i], cfg['loss_args'][i])
            for i in range(len(cfg['losses']))
        ]
        self.loss_weights = cfg['loss_weights']

        self.train_metrics = [
            initialize_metric(cfg['tr_metrics'][i], cfg['tr_metric_args'][i])
            for i in range(len(cfg['tr_metrics']))
        ]
        self.val_metrics = [
            initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i])
            for i in range(len(cfg['val_metrics']))
        ]
        self.val_metric = math.inf
        self.min_val_metric = math.inf

        # Print metrics after these many mini-batches to keep track of training
        self.log_period = len(self.dataset) // cfg['log_freq']
        if not self.log_period:
            self.log_period = 1

        # Initialize tensorboard writer
        # self.writer = writer
        self.tb_iters = 0
        self.has_add_graph = False

        # Load checkpoint if checkpoint path is provided
        if checkpoint_path is not None:
            logger.info()
            logger.info("Loading checkpoint from " + checkpoint_path + " ...")
            self.load_checkpoint(checkpoint_path, just_weights=just_weights)
            logger.info("Done")

    def train(self):
        """
        Main function to train model
        :return:
        """
        if self.distribute_training:
            spawn_context = mp.spawn(self._train,
                                     args=(self.distribute_training),
                                     nprocs=self.distribute_training,
                                     join=False)
            while not spawn_context.join():
                pass
        else:
            logger.error(
                'Please set "--distributed_training 1" to use single gpu')

    def _train(self, rank: int, world_size: int):
        output_dir = self.output_dir
        num_epochs = self.num_epoch
        if world_size > 0:
            print(f"Running DDP on rank {rank}.")

            def setup(rank, world_size):
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = 12907

                # initialize the process group
                dist.init_process_group(
                    "nccl", rank=rank, world_size=world_size
                )  # block process until processes joined

            setup(rank, world_size)
            self.model.to(rank)
            model = DDP(self.model,
                        device_ids=[rank],
                        find_unused_parameters=True)
        else:
            model = self.model.to(rank)
            # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.optim_args['lr'])
        if self.distributed_training:
            dist.barrier()

        train_sampler = DistributedSampler(self.dataset, shuffle=True)
        assert self.batch_size == 64, 'The optimal total batch size for training is 64'
        assert self.batch_size % world_size == 0

        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=train_sampler,
            batch_size=self.batch_size // world_size,
            collate_fn=batch_list_to_batch_tensors)

        # Run training, validation for given number of epochs
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, start_epoch + num_epochs):

            # Set current epoch
            self.learning_rate_decay(epoch, optimizer)
            self.current_epoch = epoch
            # if rank == 0:
                # logger.info()
                # logger.info('Epoch (' + str(self.current_epoch + 1) + '/' +
                #             str(start_epoch + num_epochs) + ')')
                # logger.info('Learning Rate = %5.8f' %
                #             optimizer.state_dict()['param_groups'][0]['lr'])
            train_sampler.set_epoch(epoch - start_epoch)

            if rank == 0:
                iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
            else:
                iter_bar = train_dataloader

            # Train
            train_epoch_metrics, _, _ = self.run_epoch(model,
                                                       mode='train',
                                                       iter_bar=iter_bar,
                                                       device=rank,
                                                       optimizer=optimizer)
            self.print_metrics(train_epoch_metrics, self.tr_dl, mode='train')
            # self.log_tensorboard_metrics(train_epoch_metrics, mode='train')

            # Validate
            with torch.no_grad():
                val_epoch_metrics, prediction, gt = self.run_epoch(
                    'val', self.val_dl)
            self.print_metrics(val_epoch_metrics, self.val_dl, mode='val')
            # self.log_tensorboard_metrics(val_epoch_metrics, mode='val')

            # Update validation metric using first metric
            self.val_metric = val_epoch_metrics[str(
                self.val_metrics[0]
            )[:-2]] / val_epoch_metrics['minibatch_count']
            # self.val_metric = val_epoch_metrics['rel'] / val_epoch_metrics['minibatch_count']

            # save best checkpoint when applicable
            if self.val_metric < self.min_val_metric:
                self.min_val_metric = self.val_metric
                self.save_checkpoint(
                    os.path.join(output_dir, 'checkpoints', 'best.tar'))
                self.save_model(
                    os.path.join(output_dir, 'saved_model',
                                 'ori_best_adms_model.pth'))
                np.save(
                    os.path.join(output_dir, 'saved_model',
                                 'ori_prediction.npy'), prediction)
                np.save(os.path.join(output_dir, 'saved_model', 'ori_gt.npy'),
                        gt)
                # np.save(os.path.join(output_dir, 'saved_model', 'per5_' + str(self.val_metric)), gt)

            # Save checkpoint every epoch.
            self.save_checkpoint(
                os.path.join(output_dir, 'checkpoints',
                             str(self.current_epoch) + '.tar'))

            # tensorboard global step
            self.tb_iters += 1

        # self.save_model(os.path.join(output_dir, 'saved_model', 'adms_model_' + time.strftime("%Y%m%d_%H%M%S") + '.pth'))
        # self.save_model(os.path.join(output_dir, 'saved_model', 'adms_model.pth'))

    def run_epoch(self, model, mode: str, iter_bar, device, optimizer):
        """
        Runs an epoch for a given dataloader
        :param mode: 'train' or 'val'
        :param dl: Dataloader object
        """
        if self.distribute_training:
            assert dist.get_world_size() == self.distribute_training

        if mode == 'val':
            model.eval()
        else:
            model.train()

        # Initialize epoch metrics
        epoch_metrics = self.initialize_metrics_for_epoch(mode)

        # Main loop
        st_time = time.time()

        # cache prediction and gt of val
        val_len = len(self.val_dl)
        prediction_val = np.empty(val_len)
        label_val = np.empty(val_len)

        for i, batch in enumerate(iter_bar):
            # Load data
            # data = u.send_to_device(u.convert_double_to_float(data))

            # (veh_cate_features, veh_dense_features, driver_cate_features,
            #  driver_dense_features, polylines, polynum,
            #  attention_mask) = self.model.preprocess_inputs(data['inputs'])

            predictions = model(batch, device)

            # Compute loss and backpropagation if training
            if mode == 'train':
                loss = self.compute_loss(predictions, batch, device)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if is_main_device(device):
                iter_bar.set_description(f'loss={loss.item():.3f}')

            # Keep time
            minibatch_time = time.time() - st_time
            st_time = time.time()

            # Aggregate metrics
            minibatch_metrics, epoch_metrics = self.aggregate_metrics(
                epoch_metrics, minibatch_time, predictions, data['label'],
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

    def compute_loss(self, model_outputs: torch.Tensor, ground_truth,
                     device) -> torch.Tensor:
        """
        Computes loss given model outputs and ground truth labels
        """
        # TODO: Implement the calculation of the multi-loss
        loss_vals = [
            loss(model_outputs, ground_truth, device) for loss in self.losses
        ]
        total_loss = torch.tensor(0).float().to(device)
        for n in range(len(loss_vals)):
            total_loss += self.loss_weights[n] * loss_vals[n]

        return total_loss

    def compute_metric(self, model_outputs: torch.Tensor,
                       ground_truth: torch.Tensor) -> torch.Tensor:
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

    def aggregate_metrics(self, epoch_metrics: Dict, minibatch_time: float,
                          model_outputs: torch.Tensor,
                          ground_truth: torch.Tensor, mode: str):
        """
        Aggregates metrics by minibatch for the entire epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics

        minibatch_metrics = {}

        # Compute different metrics
        for metric in metrics:
            minibatch_metrics[str(metric)[:-2]] = metric(
                model_outputs, ground_truth).item()

        epoch_metrics['minibatch_count'] += 1
        epoch_metrics['time_cost'] += minibatch_time
        for metric in metrics:
            epoch_metrics[str(metric)[:-2]] += minibatch_metrics[str(metric)
                                                                 [:-2]]

        # tmp for debug
        minibatch_metrics['rel'] = torch.div(
            torch.abs(model_outputs - ground_truth),
            ground_truth).mean().item()
        if 'rel' not in epoch_metrics.keys():
            epoch_metrics['rel'] = 0
        epoch_metrics['rel'] += minibatch_metrics['rel']

        return minibatch_metrics, epoch_metrics

    def print_metrics(self, epoch_metrics: Dict, dl: torch_data.DataLoader,
                      mode: str):
        """
        Prints aggregated metrics
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        minibatches_left = len(dl) - epoch_metrics['minibatch_count']
        eta = (epoch_metrics['time_cost'] /
               epoch_metrics['minibatch_count']) * minibatches_left
        epoch_progress = int(epoch_metrics['minibatch_count'] / len(dl) * 100)

        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print('\rTraining:  ' if mode == 'train' else '\rValidating:', end=" ")
        print(progress_bar,
              str(epoch_progress) if epoch_progress == 100 else
              (" " + str(epoch_progress)),
              '%',
              end=", ")
        print('ETA:', int(eta), end="s, ")
        print('Metrics', end=": { ")
        for metric in metrics:
            metric_val = epoch_metrics[str(
                metric)[:-2]] / epoch_metrics['minibatch_count']
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
            self.writer.add_scalar('train/' + metric_name, metric_val,
                                   self.tb_iters)
        self.tb_iters += 1

    def log_tensorboard_val(self, epoch_metrics: Dict):
        """
        Logs epoch metrics for validation set
        """
        for metric_name, metric_val in epoch_metrics.items():
            if metric_name != 'minibatch_count' and metric_name != 'time_cost':
                res = metric_val / epoch_metrics['minibatch_count']
                self.writer.add_scalar('val/' + metric_name, res,
                                       self.tb_iters)

    def log_tensorboard_metrics(self, epoch_metrics: Dict, mode: str):
        """
        Logs epoch metrics for validation set
        """
        for metric_name, metric_value in epoch_metrics.items():
            if metric_name != 'minibatch_count' and metric_name != 'time_cost':
                res = metric_value / epoch_metrics['minibatch_count']
                self.writer.add_scalar(mode + "/" + metric_name, res,
                                       self.tb_iters)

    def log_tensorboard_fig(self, predictions: np.array, labels: np.array,
                            mode: str):
        fig, ax = plt.subplots()
        x = range(len(predictions))
        for i in range(len(predictions)):
            plt.plot([i, i], [predictions[i], labels[i]], color="blue")
        plt.scatter(x, predictions, color="r", label="prediction")
        plt.scatter(x, labels, color='black', label="gt")
        plt.legend()
        self.writer.add_figure(mode + "/pred_label.fig", fig, self.tb_iters)

    def log_tensorboard_input_data(self, data):

        veh_cate_features = data['inputs']['veh_cate_features'].detach().cpu(
        ).numpy()
        veh_dense_features = data['inputs']['veh_dense_features'].detach().cpu(
        ).numpy()
        driver_cate_features = data['inputs']['driver_cate_features'].detach(
        ).cpu().numpy()
        driver_dense_features = data['inputs']['driver_dense_features'].detach(
        ).cpu().numpy()
        polylines = data['inputs']['polylines'].detach().cpu().numpy()
        attention_mask = data['inputs']['attention_mask'].detach().cpu().numpy(
        )
        polynum = data['inputs']['polynum'].detach().cpu().numpy()
        # label = data["label"]

        # obj traj
        fig, ax = plt.subplots()
        # for i in range(19):
        for i in range(polynum[0]):
            plt.plot(polylines[i][:][0], polylines[i][:][1])
        # plt.savefig("output/test.png")
        self.writer.add_figure("train" + "/obj_traj.fig", fig, self.tb_iters)
