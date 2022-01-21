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
Given any stream of data, any assessment of its value, and any formulation of its selection cost, this codebase extracts the most valuable subset of the stream up to a constant factor. Our method is simple (selects a point if its marginal gain under the chosen value function exceeds a given threshold chosen by the analyst at that time) and is memory-efficient (storing only the selected subset in memory). For instance, on ImageNet, if high value is assigned to class-balanced sets, given a class-imbalanced stream, our method selects a class-balanced subset of that stream. 
<p align="center">
  <img src="plots/outputs/figure1.svg">
</p>

## Usage
You can reproduce our experiments by running the following:
```
git clone https://github.com/mwerner28/dmgt
cd experiments/dmgt (if running dmgt)
cd experiments/fed_dmgt (if funning fed_dmgt)
conda env create -f environment.yml
conda activate dmgt
python run_exp.py 'path/to/imagenet(or mnist)/train' 'path/to/imagenet(or mnist)/test'
```
