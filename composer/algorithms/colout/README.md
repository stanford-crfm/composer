# ColOut

[\[How to Use\]](#how-to-use) &middot; [\[Suggested Hyperparameters\]](#suggested-hyperparameters) &middot; [\[Technical Details\]](#technical-details) &middot; [\[Attribution\]](#attribution)

`Computer Vision`

ColOut is a data augmentation technique that drops a fraction of the rows or columns of an input image for a computer vision model.
If the fraction of rows/columns isn't too large, the image content is not significantly altered but the image size is reduced, speeding up training.
This modification modestly reduces accuracy, but it is a worthwhile tradeoff for the improved speed.

| ![ColOut](col_out.png) |
|:--:
|*Several instances of an image of an apple from the CIFAR-100 dataset with ColOut applied. ColOut randomly removes different rows and columns each time it is applied.*|

## How to Use

### Functional Interface

```python
def training_loop(model, train_loader):
  opt = torch.optim.Adam(model.parameters())
  loss_fn = F.cross_entropy
  model.train()
  
  for epoch in range(num_epochs):
      for X, y in train_loader:
          y_hat = model(X)
          loss = loss_fn(y_hat, y)
          loss.backward()
          opt.step()
          opt.zero_grad()
```

### Composer Trainer

## Suggested Hyperparameters

We found that setting `p_row = 0.15` and `p_col = 0.15` strike a good balance between improving training throughput and limiting the negative impact on model accuracy. Setting `batch = True` also yields slightly lower accuracy, but we found that - in contexts that were CPU-bottlenecked - this reduction was offset by a large increase in throughput (~11% for ResNet-50 on ImageNet) because ColOut is only called once per batch and its operations are offloaded onto the GPU.

## Technical Details

ColOut reduces the size of images, reducing the number of operations per training step and consequently the total time to train the network.
The variability induced by randomly dropping rows and columns can negatively affect generalization performance. In our testing, we saw a decrease in accuracy of ~0.2% in some models on ImageNet and a decrease in accuracy of ~1% on CIFAR-10.

> 🚧 Quality/Speed Tradeoff
> 
> In our experiments, ColOut presents a tradeoff in that it increases training speed at the cost of lower model quality.
> On ResNet-50 on ImageNet and ResNet-56 on CIFAR-10, we found this tradeoff to be worthwhile: it is a pareto improvement over the standard versions of those benchmarks.
> We also found it to be worthwhile in composition with other methods.
> We recommend that you carefully evaluate whether ColOut is also a pareto improvement in the context of your application.

ColOut currently has two implementations.
One implementation acts as an additional data augmentation for use in PyTorch dataloaders. It runs on the CPU and applies ColOut independently to each training example.
A second implementation runs immediately before the training example is provided to the model. It runs on the GPU and drops the same rows and columns for all training examples in a mini-batch.
The GPU-based, batch-wise implementation suffers a drop in validation accuracy compared to the CPU-based example-wise implementation (0.2% on CIFAR-10 and 0.1% on ImageNet)

> 🚧 CPU/GPU Tradeoff
> 
> If the workload is CPU heavy, it may make sense to run ColOut batch-wise on GPU so that it does not bottleneck training on the CPU. If the workload is GPU-bottlenecked, it will make sense to run ColOut sample-wise on the CPU, avoiding the accuracy reduction of running it batch-wise and improving GPU throughput.

ColOut will show diminishing returns when composed with other methods that change the size of images, such as Progressive Resizing and Selective Backdrop with downsampling. In addition, to the extent that ColOut serves as a form of regularization, combining regularization-based methods can lead to sublinear improvements in accuracy.

## Attribution


*This method and the accompanying documentation were created and implemented by Cory Stephenson at MosaicML.*