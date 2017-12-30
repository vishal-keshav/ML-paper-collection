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

## Papers from 2016

## Papers from 2015


[cover_pic]: https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/cover.jpg
[more_is_less]: https://github.com/bulletcross/ML-paper-collection/blob/master/ML101/more_is_less.jpg
