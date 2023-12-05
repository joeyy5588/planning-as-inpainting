# Planning as In-Painting
[Cheng-Fu Yang](https://joeyy5588.github.io/chengfu-yang/), Haoyang Xu, Te-Lin Wu, Xiaofeng Gao, Kai-Wei Chang, Feng Gao

This is the official implementation of the paper: [Planning as In-Painting: A Diffusion-Based Embodied Task Planning Framework for Environments under Uncertainty](https://arxiv.org/abs/2312.01097)

### Installation
```
conda create -n diffuser python=3.8
conda activate diffuser
```

```
pip install diffusers["torch"] transformers
pip install -r requirements.txt
```

```
# If previous command fails, try this
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers accelerate
pip install -r requirements.txt
```

### General Usage
```
# For single GPU training
python train.py --config <config_file> --output_dir <output_dir>

# For multi-GPU training
# First configure your gpu settings
accelerate config
accelerate launch train.py --config <config_file> --output_dir <output_dir>

# For evaluation
python evaluate.py --checkpoint <checkpoint_dir>
```

### CompILE
```
# For 1O1G
python train.py --config config/simple2d.yaml --output_dir output/simple2d
python evaluate.py --checkpoint output/simple2d

# For MO1G
python train.py --config config/grid_heatmap.yaml --output_dir output/grid_heatmap
python evaluate.py --checkpoint output/grid_heatmap

# For P-Mo2G
python train.py --config config/grid_sequence.yaml --output_dir output/grid_sequence
python evaluate.py --checkpoint output/grid_sequence
```

### Kuka
```
# For stacking
python train.py --config config/kuka_stacking.yaml
python evaluate.py --checkpoint output/kuka_stacking

# For rearrangement
python train.py --config config/kuka_rearrangement.yaml
python evaluate.py --checkpoint output/kuka_rearrangement

# We prepare to release the inverse kinematics code soon
```

### ALFRED
```
# For training
python train.py --config config/alfred_repr.yaml --output_dir output/alfred_repr

# We prepare to release the rollout and evaluation code on ALFRED soon
```
### Citation
If you find this repository useful, please consider cite our work:
```bibtex
@misc{yang2023planning,
      title={Planning as In-Painting: A Diffusion-Based Embodied Task Planning Framework for Environments under Uncertainty}, 
      author={Cheng-Fu Yang and Haoyang Xu and Te-Lin Wu and Xiaofeng Gao and Kai-Wei Chang and Feng Gao},
      year={2023},
      eprint={2312.01097},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
