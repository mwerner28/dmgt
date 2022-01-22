## Paper
```
@article{mwerner2022dmgt,
  title={Online Active Learning with Dynamic Marginal Gain Thresholding},
  author={Werner, Mariel A and Angelopoulos, Anastasios N and Bates, Stephen and Jordan, Michael I},
  journal={arXiv preprint arXiv:?},
  year={2022}
}
```
## Overview
Given any stream of data, any assessment of its value, and any formulation of its selection cost, our method DMGT extracts the most valuable subset of the stream up to a constant factor in an online fashion. The procedure is simple (selecting each point if its marginal value given the currently selected set exceeds a threshold decided by the analyst at that time) and memory-efficient (storing only the selected subset in memory). The figure below illustrates an example in which high value is assigned to class-balanced sets. Given a class-imbalanced stream from ImageNet, DMGT selects a class-balanced subset of the stream onlin. 
<p align="center">
  <img src="plots/outputs/figure1.svg">
</p>

## Usage
You can reproduce the experiments in our paper by running:
```
git clone https://github.com/mwerner28/dmgt
cd experiments
conda env create -f environment.yml
conda activate dmgt
python run_dmgt.py(or run_fed_dmgt.py for federated version) --dataset_name 'imagenet(or mnist)' --train_path 'path/to/imagenet(or mnist)/train/' --val_path 'path/to/imagenet(or mnist)/val/'
```
