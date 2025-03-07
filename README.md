# Routing-aware training for optimal memory usage: a case study

This repo refers to this paper:

[Hardware architecture and routing-aware training for optimal memory usage: a case study](https://arxiv.org/pdf/2412.01575)

Authors: *Jimmy Weber, Theo Ballet, Melika Payvand*

Abstract—Efficient deployment of neural networks on resource-constrained hardware demands optimal use of on- chip memory. In event-based processors, this is particularly critical for routing architectures, where substantial memory is dedicated to managing network connectivity. While prior work has focused on optimizing event routing during hardware design, optimizing memory utilization for routing during network training remains underexplored. Key challenges include: (i) integrating routing into the loss function, which often intro- duces non-differentiability, and (ii) computational expense in evaluating network mappability to hardware. We propose a hardware-algorithm co-design approach to train routing-aware neural networks. To address challenge (i), we extend the DeepR training algorithm, leveraging dynamic pruning and random re-assignment to optimize memory use. For challenge (ii), we introduce a proxy-based approximation of the mapping function to incorporate placement and routing constraints efficiently. We demonstrate our approach by optimizing a network for the Spiking Heidelberg Digits (SHD) dataset using a small-world connectivity-based hardware architecture as a case study. The resulting network, trained with our routing-aware methodology, is fully mappable to the hardware, achieving 5% more accuracy using the same number of parameters, and iso-accuracy with 10x less memory usage, compared to non-routing-aware training methods. This work highlights the critical role of co-optimizing algorithms and hardware to enable efficient and scalable solutions for constrained environments.
Index Terms—HW-aware training, Dynamic architecture search, Non-differentiable constraints optimization, Routing.

# Requirements
see requirements.txt

# Installation
see Makefile

# Usage
To reproduce the results in the paper, run the notebook in the ```experiments``` folder.
