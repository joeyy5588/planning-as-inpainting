from .gridworld import GridDataset
from .simple import SimpleDataset
from .simple_2d import Simple2DDataset
from .gridworld_2d import Grid2DDataset
from .gridworld_condition import GridConditionDataset
from .gridworld_repr import GridRepresentationDataset
from .gridworld_heatmap import GridHeatMapDataset
from .gridworld_sequence import GridSequenceDataset
from .kuka import KukaDataset
from .alfred import AlfredDataset, AlfredMergeDataset, AlfredReprDataset

def build_dataset(config):
    dataset_name = config.name
    dataset = eval(dataset_name)(config)

    return dataset
