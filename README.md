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
Given any stream of data, any assessment of its value, and any formulation of its selection cost, our method, DMGT, extracts the most valuable subset of the stream up to a constant factor. The procedure is simple (selecting a point if its marginal gain under the chosen value function exceeds a given threshold chosen by the analyst at that time) and memory-efficient (storing only the selected subset in memory). The figure below illustrates its performance on ImageNet for a particular example: if high value is assigned to class-balanced sets, given a class-imbalanced stream, DMGT selects a class-balanced subset of that stream. 
<p align="center">
  <img src="plots/outputs/figure1.svg">
</p>

## Usage
You can reproduce our experiments by running:
```
git clone https://github.com/mwerner28/dmgt
cd experiments
conda env create -f environment.yml
conda activate dmgt
python run_dmgt.py --dataset_name='imagenet(or mnist)' --train_path='path/to/imagenet(or mnist)/train/' --val_path='path/to/imagenet(or mnist)/val/'
```
