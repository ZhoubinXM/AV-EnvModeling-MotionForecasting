import torch.nn

from models.model import ADMS
from models.encoders.deepfm_subgraph import DeppFMSubGraph
from models.aggregators.global_graph_aggregator import GlobalGraphAggregator
from models.decoders.mlp import MLP

from datasets.adms_dataset import ADMSDataset
from typing import Dict, Union


# TODO:
def initialize_adms_dataset(dataset_type, data_file) -> torch.utils.data.Dataset:
    """
    Helper function to initialize adms dataset by dataset type string
    """
    datasets = {'adms_dataset': ADMSDataset}
    return datasets[dataset_type](data_file)


# Models
def initialize_adms_model(encoder_type: str, aggregator_type: str,
                          decoder_type: str, encoder_args: Dict,
                          aggregator_args: Dict, decoder_args: Dict):
    """
    Helper function to initialize appropriate encoder and decoder models
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = ADMS(encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        'deepfm_subgraph': DeppFMSubGraph
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        'global_graph': GlobalGraphAggregator
    }

    return aggregator_mapping[aggregator_type](aggregator_args)


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        'mlp': MLP
    }

    return decoder_mapping[decoder_type](decoder_args)

def initialize_metric(metric_type: str, metric_args: Dict):
    """
    Initialize appropriate metric by type.
    :return:
    """
    # TODO: Implementation of Model Evaluation Metrics
    metric_mapping = {
        'mse': torch.nn.MSELoss
    }
    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()
