# Stochastic Depth (Sample-Wise)

Tags: `Method`, `Networks with Residual Connections`, `Method`, `Regularization`, `Increased Accuracy`

## TL;DR

Sample-wise stochastic depth is a regularization technique for networks with residual connections that probabilistically drops samples after the transformation function in each residual block. This means that different samples go through different combinations of blocks.

## Attribution

[EfficientNet model in the TPU Github repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) from Google

[EfficientNet model in gen-efficientnet-pytorch Github repository](https://github.com/rwightman/gen-efficientnet-pytorch) by Ross Wightman

## Hyperparameters

- `stochastic_method` - Specifies the version of the stochastic depth method to use. `stochastic_method=sample` applies stochastic dropping to samples. `stochastic_method=block` applies block-wise stochastic depth, which we address in a separate method card.
- `target_layer_name` - The reference name for the module that will be replaced with a functionally equivalent sample-wise stochastic block. For example, `target_layer_name=ResNetBottleNeck` will replace modules in the model named `BottleNeck`.
- `drop_rate` - The probability of dropping a sample within a residual block.
- `drop_distribution` - How the `drop_rate` is distributed across the model's blocks. The two possible values are `uniform` and `linear`. `uniform` assigns a single `drop_rate` across all blocks. `linear` linearly increases the drop rate according to the block's depth, starting from 0 at the first block and ending with `drop_rate` at the last block.

## Applicable Settings

Sample-wise stochastic depth requires models to have residual blocks since the method relies on skip connections to allow samples to skip blocks of the network.

## Example Effects

For both ResNet-50 and ResNet-101 on ImageNet, we measure a +0.4% absolute accuracy improvement when using `drop_rate=0.1` and `drop_distribution=linear`. The training wall-clock time is approximately 5% longer when using sample-wise stochastic depth.

## Implementation Details

When training, samples are dropped after the transformation function in a residual block by multiplying the batch by a binary vector. The binary vector is generated by sampling independent Bernoulli distributions with probability (1 - `drop_rate`). After the samples are dropped, the skip connection is added as usual. During inference, no samples are dropped, but the batch of samples is scaled by (1 - `drop_rate`) to compensate for the drop frequency when training.

## Suggested Hyperparameters

We observed that `drop_rate=0.1` and `drop_distribution=linear` yielded maximum accuracy improvements on both ResNet-50 and ResNet-101.

## Considerations

Because sample-wise stochastic depth randomly drops samples within each residual block, a shallow model may exhibit instability due to insufficient transformation on some samples. When using a shallow model, it is best to use a small drop rate or avoid sample-wise stochastic depth entirely. 

In addition, there may be instability when training on smaller batch sizes since a significant proportion of the batch may be dropped even at low drop rates.

## Composability

Combining several regularization methods may have diminishing returns, and can even degrade accuracy. This may hold true when combining sample-wise stochastic depth with other regularization methods.

--------

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.stochastic_depth.StochasticDepth
    :members: match, apply
    :noindex:

.. autofunction:: composer.algorithms.stochastic_depth.apply_stochastic_depth
    :noindex:
```