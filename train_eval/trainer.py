import time
import math
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from typing import Dict, Optional
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
from train_eval.utils import batch_list_to_batch_tensors, is_main_device,\
    show_heatmaps, get_max_st_from_spans, get_from_mapping
import global_var

global_var._init()
tqdm = partial(tqdm_, dynamic_ncols=True)

# Initialize device:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.optim_args: Dict = cfg['optim_args']
        self.batch_size: int = int(cfg['batch_size'])
        self.output_dir: str = cfg['output_dir']
        self.num_epoch: int = int(cfg['num_epochs'])
        self.viz: bool = bool(cfg['viz'])
        self.use_cuda: bool = bool(cfg['use_cuda'])
        self.multi_gpu: bool = bool(cfg['use_multi_gpu'])
        self.world_size: int = int(
            os.environ["WORLD_SIZE"]) if 'WORLD_SIZE' in os.environ else 0
        self.past_length = int(cfg['past_length'])
        self.pred_length = int(cfg['pred_length'])

        # cuda or cpu
        self.main_device: int = int(cfg['main_device'])
        self.cuda_id = int(os.environ["LOCAL_RANK"]) if (
            'LOCAL_RANK' in os.environ) and self.use_cuda else self.main_device
        self.device = torch.device(
            "cuda:{}".format(self.cuda_id)
            if torch.cuda.is_available() and self.use_cuda else "cpu")
        if 'WORLD_SIZE' in os.environ and self.multi_gpu:
            self.multi_gpu = True if int(
                os.environ['WORLD_SIZE']) > 1 else False
        else:
            self.multi_gpu = False

        if self.multi_gpu:
            torch.cuda.set_device(self.device)
            dist.init_process_group('nccl', init_method='env://')
            if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                print(
                    f"CUDA: {self.cuda_id} - RANK: {os.environ['LOCAL_RANK']} / WORLD_SIZE: {os.environ['WORLD_SIZE']}"
                )

        # Initialize datasets:
        self.train_dataset = initialize_adms_dataset(cfg['dataset'],
                                                     cfg['train_datafile'])
        self.val_dataset = initialize_adms_dataset(cfg['dataset'],
                                                   cfg['val_datafile'],
                                                   mode='val')

        if cfg['encoder_type'] == 'wayformer_enc' and (
                'max_temporal' not in cfg['encoder_args'].keys()
                or 'max_spatial' not in cfg['encoder_args'].keys()):
            train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=batch_list_to_batch_tensors)

            val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                collate_fn=batch_list_to_batch_tensors)

            max_spatial_num, max_vector_num = 0, 0
            for mapping in train_dataloader:
                polyline_spans = get_from_mapping(mapping, 'polyline_spans')
                spatial_num, slice_num_list = get_max_st_from_spans(
                    polyline_spans)
                if max(spatial_num) > max_spatial_num:
                    max_spatial_num = max(spatial_num)
                if max(max(row) for row in slice_num_list) > max_vector_num:
                    max_vector_num = max(max(row) for row in slice_num_list)
            for mapping in val_dataloader:
                polyline_spans = get_from_mapping(mapping, 'polyline_spans')
                spatial_num, slice_num_list = get_max_st_from_spans(
                    polyline_spans)
                if max(spatial_num) > max_spatial_num:
                    max_spatial_num = max(spatial_num)
                if max(max(row) for row in slice_num_list) > max_vector_num:
                    max_vector_num = max(max(row) for row in slice_num_list)
            cfg['encoder_args'].update(
                dict(max_temporal=max_vector_num, max_spatial=max_spatial_num))
            print(max_spatial_num, max_vector_num)

        if self.multi_gpu:
            # use distributed data sampler
            self.train_sampler = DistributedSampler(self.train_dataset,
                                                    shuffle=True)

            self.val_sampler = DistributedSampler(self.val_dataset,
                                                  shuffle=False)

            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size // self.world_size,
                sampler=self.train_sampler,
                shuffle=False,
                # collate_fn=batch_list_to_batch_tensors
            )
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size // self.world_size,
                sampler=self.val_sampler,
                shuffle=False,
                # collate_fn=batch_list_to_batch_tensors
            )
        else:
            self.train_sampler = RandomSampler(self.train_dataset)

            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                sampler=self.train_sampler,
                batch_size=self.batch_size,
                # collate_fn=batch_list_to_batch_tensors
            )

            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                # collate_fn=batch_list_to_batch_tensors
            )

        # Initialize model
        self.model = initialize_adms_model(
            cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
            cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.model = self.model.float().to(self.device)

        if self.multi_gpu:
            self.model = DDP(self.model,
                             device_ids=[self.cuda_id],
                             output_device=self.cuda_id,
                             find_unused_parameters=True)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.optim_args['lr'])

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
        self.log_period = len(self.train_dataloader) // cfg['log_freq']
        if not self.log_period:
            self.log_period = 1

        # Initialize tensorboard writer
        if not self.multi_gpu or (self.multi_gpu and is_main_device(
                self.cuda_id, self.main_device)):
            self.writer = writer
        self.tb_iters = 0
        self.has_add_graph = False

        # Load checkpoint if checkpoint path is provided
        if checkpoint_path is not None:
            logger.info("Loading checkpoint from " + checkpoint_path + " ...")
            self.load_checkpoint(checkpoint_path, just_weights=just_weights)
            logger.info("Done")

    def train(self):
        """
        Main function to train model
        :return:
        """
        # Run training, validation for given number of epochs
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, start_epoch + self.num_epoch):

            # Set current epoch
            self.learning_rate_decay(epoch, self.optimizer)
            self.current_epoch = epoch
            if is_main_device(self.cuda_id, self.main_device):
                logger.info(" ")
                logger.info('Epoch (' + str(self.current_epoch + 1) + '/' +
                            str(start_epoch + self.num_epoch) + ')')
                logger.info(
                    'Learning Rate = %5.8f' %
                    self.optimizer.state_dict()['param_groups'][0]['lr'])
            if self.multi_gpu:
                self.train_sampler.set_epoch(epoch - start_epoch)

            if is_main_device(self.cuda_id, self.main_device):
                iter_bar = tqdm(
                    self.train_dataloader,
                    desc='Iter (loss=X.XXX / minfde6=X.XXX / mr=X.XXX)')
            else:
                iter_bar = self.train_dataloader

            # Train
            train_epoch_metrics = self.run_epoch(mode='train',
                                                 iter_bar=iter_bar)

            if is_main_device(self.cuda_id, self.main_device):
                self.print_metrics(train_epoch_metrics,
                                   self.train_dataloader,
                                   mode='train')
            # self.log_tensorboard_metrics(train_epoch_metrics, mode='train')

            # Validate
            with torch.no_grad():
                if is_main_device(self.cuda_id, self.main_device):
                    val_iter_bar = tqdm(self.val_dataloader, desc='Val: ')
                else:
                    val_iter_bar = self.val_dataloader
                val_epoch_metrics = self.run_epoch('val', val_iter_bar)

            if is_main_device(self.cuda_id, self.main_device):
                self.print_metrics(val_epoch_metrics,
                                   self.val_dataloader,
                                   mode='val')

            # Update validation metric using first metric
            self.val_metric = val_epoch_metrics[str(
                self.val_metrics[4])] / val_epoch_metrics['minibatch_count']

            # save best checkpoint when applicable
            if is_main_device(self.cuda_id, self.main_device
                              ) and self.val_metric < self.min_val_metric:
                self.min_val_metric = self.val_metric
                self.save_checkpoint(
                    os.path.join(self.output_dir, 'checkpoints', 'best.tar'))
                # self.save_model(
                #     os.path.join(self.output_dir, 'saved_model',
                #                  'ori_best_adms_model.pth'))
            if self.multi_gpu:
                dist.barrier()

            # Save checkpoint every epoch.
            if is_main_device(self.cuda_id, self.main_device):
                self.save_checkpoint(
                    os.path.join(self.output_dir, 'checkpoints',
                                 str(self.current_epoch) + '.tar'))

            # tensorboard global step
            self.tb_iters += 1

        # self.save_model(os.path.join(output_dir, 'saved_model', 'adms_model_' + time.strftime("%Y%m%d_%H%M%S") + '.pth'))
        # self.save_model(os.path.join(output_dir, 'saved_model', 'adms_model.pth'))

    def run_epoch(self, mode: str, iter_bar):
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

        for i, batch in enumerate(iter_bar):
            # Load data
            batch = u.send_to_device(u.convert_double_to_float(batch),
                                     device=self.device)
            batch['target_history'] = batch['target_history'][:, :self.
                                                              past_length]
            batch['target_future'] = batch['target_future'][:, :self.
                                                            pred_length]
            # batch = batch['target_history']
            # self.model.eval()
            # traced_model = torch.jit.trace(self.model, batch)
            # traced_model.save("mlp_resmlp_2_3.pt")
            
            predictions = self.model(batch, self.device)

            # Compute loss and backpropagation if training
            if mode == 'train':
                loss = self.compute_loss(predictions, batch, self.device)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Keep time
            minibatch_time = time.time() - st_time
            st_time = time.time()

            # Aggregate metrics
            minibatch_metrics, epoch_metrics = self.aggregate_metrics(
                epoch_metrics, minibatch_time, predictions, batch, mode)

            if mode == 'train' and is_main_device(self.cuda_id,
                                                  self.main_device):
                # minfde_1 = (minibatch_metrics['min_fde_1'])
                # mr = (minibatch_metrics['miss_rate_1'])
                # iter_bar.set_description(
                #     f'loss={loss.item():.3f} / minfde1={minfde_1:.3f} / mr={mr:.3f}'
                # )
                minfde = (minibatch_metrics['minfde'])
                minade = (minibatch_metrics['minade'])
                iter_bar.set_description(
                    f'loss={loss.item():.3f} / minade={minade:.3f} / minfde={minfde:.3f}'
                )

            # Log minibatch metrics to tensorboard during training
            if mode == 'train' and is_main_device(self.cuda_id,
                                                  self.main_device):
                self.log_tensorboard_train(minibatch_metrics)

                # # Display metrics at a predefined frequency
                # if i % self.log_period == self.log_period - 1 and len(
                #         iter_bar) - i != 1 and is_main_device(
                #             self.cuda_id, self.main_device):
                #     self.print_metrics(epoch_metrics, iter_bar, mode)

        # Log val metrics for the complete epoch to tensorboard
        if mode == 'val' and is_main_device(self.cuda_id, self.main_device):
            self.log_tensorboard_val(epoch_metrics)

        # if is_main_device(self.cuda_id, self.main_device) and self.viz:
        #     self.log_tensorboard_att_scores()

        return epoch_metrics

    def learning_rate_decay(self, i_epoch, optimizer):
        if i_epoch > 15 and i_epoch % 10 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9

    def compute_loss(self, model_outputs: torch.Tensor, ground_truth,
                     device) -> torch.Tensor:
        """
        Computes loss given model outputs and ground truth labels
        """
        # TODO: Implement the calculation of the multi-loss
        loss_vals = [
            loss(model_outputs, ground_truth, device)
            if str(loss) not in ['MSELoss()', 'SmoothL1Loss()'] else loss(
                model_outputs[0].squeeze(1).reshape(
                    model_outputs[0].shape[0],
                    -1), ground_truth['fut_traj'].reshape(
                        model_outputs[0].shape[0], -1)) for loss in self.losses
        ]
        # total_loss = torch.tensor(0).float().to(device)
        # for n in range(len(loss_vals)):
        #     total_loss += self.loss_weights[n] * loss_vals[n]

        # return total_loss
        return loss_vals[0][0][-1]

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
            epoch_metrics[str(metric)] = 0

        return epoch_metrics

    def aggregate_metrics(self, epoch_metrics: Dict, minibatch_time: float,
                          model_outputs: torch.Tensor, batch, mode: str):
        """
        Aggregates metrics by minibatch for the entire epoch
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics

        minibatch_metrics = {}

        # Compute different metrics
        for metric in metrics:
            if str(metric) in ['MSELoss()', 'SmoothL1Loss()']:
                minibatch_metrics[str(metric)] = metric(
                    model_outputs[0].squeeze(1).reshape(
                        model_outputs[0].shape[0], -1),
                    batch['fut_traj'].reshape(model_outputs[0].shape[0],
                                              -1)).item()
            elif str(metric) == "motion_loss":
                metrics = ['motion_loss', 'cls_loss', 'reg_loss', 'minade', 'minfde']
                out_metric = metric(
                    model_outputs, batch, self.device)
                for i, metric in enumerate(metrics):
                    minibatch_metrics[metric] = out_metric[i][-1] 
            else:
                minibatch_metrics[str(metric)] = metric(
                    model_outputs, batch, self.device).item()

        epoch_metrics['minibatch_count'] += 1
        epoch_metrics['time_cost'] += minibatch_time
        for metric in metrics:
            try:
                epoch_metrics[str(metric)] += minibatch_metrics[str(metric)]
            except:
                 epoch_metrics[str(metric)] = minibatch_metrics[str(metric)]

        return minibatch_metrics, epoch_metrics

    def print_metrics(self, epoch_metrics: Dict, dl: torch_data.DataLoader,
                      mode: str):
        """
        Prints aggregated metrics
        """
        metrics = self.train_metrics if mode == 'train' else self.val_metrics
        minibatches_left = len(dl) - epoch_metrics['minibatch_count']
        # eta = (epoch_metrics['time_cost'] /
        #        epoch_metrics['minibatch_count']) * minibatches_left
        # epoch_progress = int(epoch_metrics['minibatch_count'] / len(dl) * 100)

        # progress_bar = '['
        # for i in range(20):
        #     if i < epoch_progress // 5:
        #         progress_bar += '='
        #     else:
        #         progress_bar += ' '
        # progress_bar += ']'
        # print('\rTraining:  ' if mode == 'train' else '\rValidating:', end=" ")
        # print(progress_bar,
        #       str(epoch_progress) if epoch_progress == 100 else
        #       (" " + str(epoch_progress)),
        #       '%',
        #       end=", ")
        # print('ETA:', int(eta), end="s, ")
        print('Metrics', end=": { ")
        for metric in metrics:
            metric_val = epoch_metrics[str(
                metric)] / epoch_metrics['minibatch_count']
            print(str(metric) + ':', format(metric_val, '.2f'), end=", ")
        print('\b\b }     ', end="\n" if minibatches_left == 0 else "")

    def load_checkpoint(self, checkpoint_path, just_weights=False):
        """
        Loads checkpoint from given path
        """
        if self.multi_gpu:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.cuda_id}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            checkpoint = {("module." + k) if not k.startswith("module") else k:
                          v
                          for k, v in checkpoint.items()}
        else:
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
        state_dict = self.model.state_dict()
        if self.multi_gpu:
            state_dict = {
                k[len("module."):]: v
                for k, v in state_dict.items() if k.startswith("module")
            }
        torch.save(
            {
                'epoch': self.current_epoch + 1,
                'model_state_dict': state_dict,
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

    def log_tensorboard_att_scores(self, viz_num: Optional[int] = 2):
        if self.multi_gpu:
            temple_att_scores = self.model.module.encoder.layers[
                -1].attention_probs.cpu()
            spatial_att_scores = self.model.module.aggregator.global_graph.attention_probs.cpu(
            )
        else:
            temple_att_scores = self.model.encoder.layers[
                -1].attention_probs.cpu()
            spatial_att_scores = self.model.aggregator.global_graph.attention_probs.cpu(
            )
        self.writer.add_figure(
            'train/temporal_att_scores',
            show_heatmaps(temple_att_scores[:viz_num, ...],
                          xlabel='Key positions',
                          ylabel='Query positions',
                          titles=[
                              'Head %d' % i
                              for i in range(1, temple_att_scores.shape[1] + 1)
                          ],
                          figsize=(7, 3.5)), self.tb_iters)
        self.writer.add_figure(
            'train/spatial_att_scores',
            show_heatmaps(spatial_att_scores[:viz_num, ...],
                          xlabel='Key positions',
                          ylabel='Query positions',
                          titles=[
                              'Head %d' % i
                              for i in range(1, spatial_att_scores.shape[1] +
                                             1)
                          ],
                          figsize=(7, 3.5)), self.tb_iters)
        self.tb_iters += 1

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
