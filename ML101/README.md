# ML101

![Cover][cover_pic]

```A simple list of around 101 ML papers and corresponding links.
These paper appeared in several renowned conferences around the world with
an impact of making inference faster in deep learning terminology.
Ideas from these paper have been extracted and presented in a simplified manner.
```
## Papers yet to be published

## Papers from 2017

### More is Less: A More Complicated Network with Less Inference Complexity

```
Problem solved: With each convolution layer, reduce
the computation of convolution that results in zero
activation.
```
* Hints that the computation will lead to zero activation
is given by a parallel light-weight convolution's (either spatial or depth wise) activation.
* LCCL (low cost collaborative layer) are parallel low cost convolutions providing hints to main convolution that output gonna be zero
* To increase the sparsity, methods such as using RelU activation and Batch Normalization is explored.
* Some lights on GEMM and GEMV has been also put to argue about theoretical and practical acceleration improvements.

![more_is_less][more_is_less]
>[LINK](https://arxiv.org/abs/1703.08651)

### Learning Efficient Convolutional Networks through Network Slimming

```
Problem solved: Reduction of input input channels that have near
zero activation by enforcing L1 norm to be zero along with regularization
effect. Lesser the input channels, faster the inference.
```

* Input channel are pruned during training by enforcing L1 norm to become zero.
* L1 regularization on scaling factor of batch normalization layers per channel does not require any architectural changes.
* Zero L1 enforcement per channel is achieved by addition of L1 norm of batch in the total cost function which then is needed to be minimized.
* A little bit in drop of accuracy can be restored by retraining while multi-pass scheme can be applied to push more channel pruning

![network_slimming][network_slimming]
>[LINK](https://arxiv.org/abs/1708.06519v1)

### Pruning Filters For Efficient Convnets

```
Problem solved: Reduction in number of filters with low L1 norm.
These filters are assumed to have insignificant contribution to
output channels. With reduced filters, both complexity of computation
of convolution at current layer and next layer decreases.
```

* Filters in a given layer of a trained model are compared against each other based on L1 norm, and removed with lease (proportionally)
* Removal of filters is done in a single shot for whole network, after which training can be done to restore accuracy
* Pruning filters across multiple layer can be done in either independent or greedy way.
* Sensitivity of pruning filters in a given layer is measured by the rate at which accuracy decreases with respect to number of pruned filter. Highly sensitive layers are left out for pruning.

![filter_pruning][filter_pruning]
>[LINK](https://arxiv.org/abs/1608.08710v3)

### Data-Driven Sparse Structure Selection for Deep Neural Networks

```
Problem solved: A systematic approach to prune structures is a
deep network including filters or neurons, resnet blocks and grouped
convolution (cardinality) is presented. Additional parameters
in cost function is enforced to zero in order to realize structure
pruning.
```

* A general end-to-end framework for pruning structures at once without the need for retraining.
* Scaling factors is not needed to be necessarily associated to network weights (so does the gradients), these are generalized parameters and gradients are computed in additions to weight gradients
* Accelerated Proximal Gradient is adopted for scaling factor training

![sparse_structure][sparse_structure]
>[LINK](https://arxiv.org/abs/1707.01213v2)

### Pruning Convolutional Neural Networks For Resource Efficient Inference

```
Problem solved: Evaluates/proposes the greedy criterion for pruning
feature maps with an objective of minimizing the cost difference
before and after pruning along with L1 norm constraints on pruned
weights. With FLOPs criteria, it achieves resource efficient inference.
```
* It formalizes the gated pruning (i.e. if this feature channel be pruned or not) based on greedy criteria such as minimized weight, mean, deviation of activation, information gain, first order taylor expansion etc.
* System has two steps, pruning in which based on criteria, the importance of neurons is evaluated and pruned
* Training in which it trains the network with constrained cost object in consideration.

![criteria_pruning][criteria_pruning]
>[LINK](https://arxiv.org/abs/1611.06440v2)

### CondenseNet: An Efficient DenseNet using Learned Group Convolutions

```
Problem solved: Improves the dense-net by introducing the concept
of learned group convolution. Condensing and optimization
(both being the part of training phase) helps prune filters per
logical groups which gets re-arranged by indexing to formulate
normal group convolution. Better design strategy for dense-net
is also proposed.
```
* Exponential growth rate of input channel is compensated by using depth wise separable convolution in case of dense net. Condense net improvises this by group convolution that learns the grouping.
* Method used allows groups to automatically select appropriate channels or not use some at all. Since this architecture follows dense-net, non-used channels will be useful in another layer.
* Condense stage involves pruning per group by sparsity regularizer and optimization stage (second half of training stage) involves re-arranging and indexing of input filters for group convolution.

![condense_net][condense_net]
>[LINK](https://arxiv.org/abs/1711.09224v1)

### DelugeNets: Deep Networks with Efficient and Flexible Cross-layer Information Inflows

```
Problem solved: Effectively propagating cross layers parameters
as compared to residual blocks but at the same time being more
efficient that dense-nets. Through cross-layer depthwise convolution
the efficient information flow is possible whilst providing more
flexibility than resnets.
```

* Per block, composite network is used (bottle neck design) having same map dimentions.
* In one block, inputs from each preceding layers are convolved by point wise convolution channel wise. In other words, for a given channel c, all the filter maps at channel c from preceding layers are segregated and convolved depthwise to get filter map of channel c in the input layer. Same process to all channels.
* For block transition (where dimention changes), 3X3 strided filters are applied to match the input dimentions.
* Whole idea is to have cross layer separable convolution to reduce computation complexity.

![deluge_net][deluge_net]
>[LINK](https://arxiv.org/abs/1611.05552v5)

## Papers from 2016

### EIE: Efficient Inference Engine on Compressed Deep Neural Network

```
Problem solved: To fit whole network model in cache
so that DRAM access is minimized, models are Compressed
by involving techniques such as pruning and weight sharing.
Once model are in cached, computation requires indirect
access to weights(exploit static sparsity) and activations
(exploit dynamic sparsity) which costs inefficiency.
Thus compressed models are expanded.
EIE develops a hardware design that directly works on
compressed model using CSC format for inference.
```
* CSC(compressed sparse column) format employees two vectors. First encodes shared weights and second encodes relative distance in cache memory.
* This design effectively works on fully connected layers.
* Hardware design includes description of all logical units and queuing/scheduling algorithms.

![EIE][EIE]
>[LINK](https://arxiv.org/abs/1602.01528v2)

## Papers from 2015


[cover_pic]: https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/cover.jpg
[more_is_less]: https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/more_is_less.jpg
[EIE]: https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/EIE.jpg
[network_slimming]:
https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/network_slimming.jpg
[filter_pruning]:
https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/filter_pruning.jpg
[sparse_structure]:
https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/sparse_structure.jpg
[criteria_pruning]:
https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/criteria_pruning.jpg
[condense_net]:
https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/condense_net.jpg
[deluge_net]:
https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/deluge_net.jpg
