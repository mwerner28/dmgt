## Paper
## Overview
Given any stream of data, any assessment of its value, and any formulation of its selection cost, this codebase extracts the most valuable subset of the stream up to a constant factor. Our method is simple (selects a point if its marginal gain under the chosen value function exceeds a given threshold chosen by the analyst at that time) and is memory-efficient (storing only the selected subset in memory).  
For instance, on ImageNet, if high value is assigned to class-balanced sets, given a class-imbalanced stream, our method selects a class-balanced subset of that stream. 
![](<plots/outputs/figure1.pdf>)
<img src="plots/outputs/figure1.pdf">
## Usage
