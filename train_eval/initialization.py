import torch.nn

from models.model import ADMS
from models.encoders.deepfm_subgraph import DeppFMSubGraph
from models.encoders.new_subgraph import NewSubGraph
from models.encoders.wayformer_enc import WayFormerEncoder
from models.encoders.img_bev import ImgBevEnc
from models.encoders.motionformer import MotionFormer
from models.aggregators.global_graph_aggregator import GlobalGraphAggregator
from models.aggregators.densetnt_spatial_transformer_agg import SpatialTransformerEncoder
from models.decoders.mlp import MLPdecx, MLPDecoder
from models.decoders.resduial_concat_mlp import DecoderResCat
from models.decoders.motion_mtp import MotionMTP

from datasets.adms_dataset import ADMSDataset
from datasets.argoverse_dataset import ArgoverseDataset
from datasets.prediction_dataset import PredictionDataset
from datasets.motion_dataset import MotionDataset

from metric.variety_loss import VarietyLoss
from metric.minfde_k import MinFDEK
from metric.minade_k import MinADEK
from metric.miss_rate import MissRateK
from metric.motion_loss import MotionLoss
from typing import Dict, Union, Optional


# Adapt more dataset.
def initialize_adms_dataset(
        dataset_type: str,
        data_file: Union[Dict, str],
        mode: Optional[str] = 'train') -> torch.utils.data.Dataset:
    """
    Helper function to initialize adms dataset by dataset type string
    """
    datasets = {
        'adms_dataset': ADMSDataset,
        'argoverse_dataset': ArgoverseDataset,
        'img_dataset': PredictionDataset,
        'motion_dataset': MotionDataset
    }
    return datasets[dataset_type](data_file, mode)


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
    # Update as we add more encoder types
    encoder_mapping = {
        'deepfm_subgraph': DeppFMSubGraph,
        'new_subgraph': NewSubGraph,
        'wayformer_enc': WayFormerEncoder,
        'img_bev_enc': ImgBevEnc,
        'motionformer': MotionFormer,
    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict,
                                                                       None]):
    """
    Initialize appropriate aggregator by type.
    """
    # Update as we add more aggregator types
    aggregator_mapping = {
        'global_graph': GlobalGraphAggregator,
        'spatial_aggregator': SpatialTransformerEncoder
    }

    return aggregator_mapping[aggregator_type](aggregator_args)


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # Update as we add more decoder types
    decoder_mapping = {'mlp': MLPdecx, 'residual_concat_mlp': DecoderResCat,
                       'mmlp': MLPDecoder, 'motion_mtp': MotionMTP}

    return decoder_mapping[decoder_type](decoder_args)


def initialize_metric(metric_type: str, metric_args: Dict):
    """
    Initialize appropriate metric by type.
    :return:
    """
    # Implementation of Model Evaluation Metrics
    metric_mapping = {
        'mse': torch.nn.MSELoss,
        'variety_loss': VarietyLoss,
        'min_fde': MinFDEK,
        'min_ade': MinADEK,
        'miss_rate': MissRateK,
        'smooth_l1': torch.nn.SmoothL1Loss,
        'motion_loss': MotionLoss,
    }
    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()
