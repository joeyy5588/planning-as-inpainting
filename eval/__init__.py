from .grid_eval import GridEvaluator, Grid2DEvaluator, Grid2DReprEvaluator, POMDPReprEvaluator
from .simple_eval import SimpleEvaluator, Simple2DEvaluator
from .pipeline import PlanningPipeline
from .conditional_pipeline import Conditional2DPipeline, KukaPipeline, AlfredPipeline
from .kuka_eval import KukaEvaluator
from .alfred_eval import AlfredEvaluator

def build_evaluator(config):
    dataset_name = config.name
    if dataset_name == "GridDataset":
        evaluator = GridEvaluator()
    elif dataset_name == "SimpleDataset":
        evaluator = SimpleEvaluator()
    elif dataset_name == "Simple2DDataset":
        evaluator = Simple2DEvaluator()
    elif dataset_name == "Grid2DDataset":
        evaluator = Grid2DEvaluator()
    elif dataset_name == "GridConditionDataset":
        evaluator = Grid2DEvaluator()
    elif dataset_name == "GridRepresentationDataset":
        evaluator = Grid2DReprEvaluator()
    elif dataset_name == "GridHeatMapDataset":
        evaluator = Grid2DReprEvaluator()
    elif dataset_name == "GridSequenceDataset":
        evaluator = POMDPReprEvaluator()
    elif dataset_name == "KukaDataset":
        evaluator = KukaEvaluator()
    elif "Alfred" in dataset_name:
        evaluator = AlfredEvaluator()
        

    return evaluator

def build_pipeline(config, checkpoint, device="cuda"):
    dataset_name = config.name
    if dataset_name == "GridDataset":
        pipeline = PlanningPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif dataset_name == "SimpleDataset":
        pipeline = PlanningPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif dataset_name == "Simple2DDataset":
        pipeline = PlanningPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif dataset_name == "Grid2DDataset":
        pipeline = PlanningPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif dataset_name == "GridConditionDataset":
        pipeline = Conditional2DPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif dataset_name == "GridRepresentationDataset":
        pipeline = Conditional2DPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif dataset_name == "GridHeatMapDataset":
        pipeline = Conditional2DPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif dataset_name == "GridSequenceDataset":
        pipeline = Conditional2DPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif dataset_name == "KukaDataset":
        pipeline = KukaPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)
    elif "Alfred" in dataset_name:
        pipeline = AlfredPipeline.from_pretrained(checkpoint, use_safetensors=True).to(device)

    return pipeline