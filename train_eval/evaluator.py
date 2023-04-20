import torch.utils.data as torch_data
from typing import Dict
from train_eval.initialization import initialize_adms_model, initialize_metric, \
    initialize_adms_dataset
import torch
import os
import train_eval.utils as u
import numpy as np
import json
from datasets.adms_dataset import adms_collate

# Initialize device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    """
    Class for evaluating trained models
    """

    def __init__(self, cfg: Dict, data_root: str, data_dir: str, checkpoint_path: str):
        """
        Initialize evaluator object
        :param cfg: Configuration parameters
        :param data_root: Root directory with data
        :param data_dir: Directory with extracted, pre-processed data
        :param checkpoint_path: Path to checkpoint with trained weights
        """

        # TODO: Initialize test dataset
        test_set = initialize_adms_dataset(cfg['dataset'])

        # Initialize dataloader
        self.dl = torch_data.DataLoader(test_set, cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'],
                                        collate_fn=adms_collate)

        # Initialize model
        self.model = initialize_adms_model(cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
                                           cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
        self.model = self.model.float().to(device)
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Initialize metrics
        self.metrics = [initialize_metric(cfg['val_metrics'][i], cfg['val_metric_args'][i])
                        for i in range(len(cfg['val_metrics']))]

    def evaluate(self, output_dir: str):
        """
        Main function to evaluate trained model
        :param output_dir: Output directory to store results
        """

        # Initialize aggregate metrics
        agg_metrics = self.initialize_aggregate_metrics()

        prediction_res = []
        label_res = []

        with torch.no_grad():
            for i, data in enumerate(self.dl):
                # Load data
                data = u.send_to_device(u.convert_double_to_float(data))

                # Forward pass
                predictions = self.model(data['inputs'])

                # Aggregate metrics
                agg_metrics = self.aggregate_metrics(agg_metrics, predictions.squeeze(-1), data['label'])

                self.print_progress(i)

                prediction_res.extend(predictions.squeeze(-1).cpu().tolist())
                label_res.extend(data['label'].cpu().tolist())
        
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot(prediction_res, color='red', label='prediction')
        plt.plot(label_res, color='purple', label='label')
        plt.legend()
        plt.savefig("./result/result_for_debug/predicton_label_" + str(i) + ".png")
        plt.cla()
        plt.plot(np.array(prediction_res) - np.array(label_res))
        plt.savefig("./result/result_for_debug/diff_" + str(i) + ".png")


        # compute and print average metrics
        self.print_progress(len(self.dl))
        with open(os.path.join(output_dir, 'results', "results.txt"), "w") as out_file:
            for metric in self.metrics:
                avg_metric = agg_metrics[str(metric)[:-2]] / agg_metrics['sample_count']
                output = str(metric)[:-2] + ': ' + format(avg_metric, '0.2f')
                print(output)
                out_file.write(output + '\n')

    def initialize_aggregate_metrics(self):
        """
        Initialize aggregate metrics for test set.
        """
        agg_metrics = {'sample_count': 0}
        for metric in self.metrics:
            agg_metrics[str(metric)[:-2]] = 0

        return agg_metrics

    def aggregate_metrics(self, agg_metrics: Dict, model_outputs: Dict, ground_truth: Dict):
        """
        Aggregates metrics for evaluation
        """
        minibatch_metrics = {}
        for metric in self.metrics:
            minibatch_metrics[str(metric)[:-2]] = metric(model_outputs, ground_truth)

        batch_size = ground_truth.shape[0]
        agg_metrics['sample_count'] += batch_size

        for metric in self.metrics:
            agg_metrics[str(metric)[:-2]] += minibatch_metrics[str(metric)[:-2]] * batch_size

        return agg_metrics

    def print_progress(self, minibatch_count: int):
        """
        Prints progress bar
        """
        epoch_progress = minibatch_count / len(self.dl) * 100
        print('\rEvaluating:', end=" ")
        progress_bar = '['
        for i in range(20):
            if i < epoch_progress // 5:
                progress_bar += '='
            else:
                progress_bar += ' '
        progress_bar += ']'
        print(progress_bar, format(epoch_progress, '0.2f'), '%', end="\n" if epoch_progress == 100 else " ")
