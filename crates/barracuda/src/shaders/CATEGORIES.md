# Shader Category Index

Quick reference for finding shaders by name or purpose.

---

## Activations (40)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `gelu.wgsl` | Gaussian Error Linear Unit | Transformers, BERT |
| `silu.wgsl` | Sigmoid Linear Unit (Swish) | EfficientNet, Mobile |
| `relu.wgsl` | Rectified Linear Unit | General CNN |
| `leaky_relu.wgsl` | Leaky ReLU (negative slope) | GANs, Deep nets |
| `mish.wgsl` | Self-gated smooth activation | State-of-art models |
| `hardswish.wgsl` | Efficient approximation of Swish | Mobile, quantized |
| `sigmoid.wgsl` | Logistic function | Binary classification |
| `tanh.wgsl` | Hyperbolic tangent | RNNs, normalization |
| `softmax.wgsl` | Probability distribution | Classification output |
| `log_softmax.wgsl` | Log of softmax (numerically stable) | NLL loss |

---

## Attention (19)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `attention_matmul.wgsl` | Q @ K^T | Standard attention |
| `attention_softmax.wgsl` | Softmax over attention scores | Attention weights |
| `attention_apply.wgsl` | Attention @ V | Weighted sum |
| `causal_attention_softmax.wgsl` | Masked softmax for autoregressive | GPT, decoder |
| `cross_attention_matmul.wgsl` | Cross attention Q @ K^T | Encoder-decoder |
| `local_attention_softmax.wgsl` | Sparse local attention | Long sequences |
| `flash_attention.wgsl` | Memory-efficient fused attention | Large models |
| `gqa_matmul.wgsl` | Grouped query attention | Llama, efficient |

---

## Linear Algebra (32)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `cholesky.wgsl` | Cholesky decomposition A = L L^T | Positive definite |
| `eigh.wgsl` | Eigenvalue decomposition (symmetric) | PCA, physics |
| `linsolve.wgsl` | Solve Ax = b (Gaussian elimination) | Linear systems |
| `triangular_solve.wgsl` | Solve Lx = b or Ux = b | Cholesky back-sub |
| `inverse.wgsl` | Matrix inversion | Small matrices |
| `determinant.wgsl` | Matrix determinant | 2x2, 3x3 |
| `matrix_power.wgsl` | A^n | Markov chains |
| `matrix_rank.wgsl` | Matrix rank estimation | Singularity |
| `trace.wgsl` | Sum of diagonal | Regularization |
| `diag.wgsl` | Extract/create diagonal | Diagonal matrices |

---

## Loss Functions (34)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `focal_loss.wgsl` | Focal loss (class imbalance) | Object detection |
| `dice_loss.wgsl` | Dice coefficient loss | Segmentation |
| `giou_loss.wgsl` | Generalized IoU loss | Object detection |
| `iou_loss.wgsl` | Intersection over Union loss | Bounding boxes |
| `binary_cross_entropy.wgsl` | BCE for binary classification | Binary output |
| `cross_entropy.wgsl` | Cross-entropy for multi-class | Classification |
| `mse_loss.wgsl` | Mean squared error | Regression |
| `mae_loss.wgsl` | Mean absolute error | Robust regression |
| `huber_loss.wgsl` | Smooth L1 loss | Outlier robust |
| `kl_divergence.wgsl` | KL divergence | VAE, distribution |
| `triplet_loss.wgsl` | Triplet loss (metric learning) | Face recognition |
| `contrastive_loss.wgsl` | Contrastive loss | Siamese networks |
| `center_loss.wgsl` | Center loss (intra-class) | Face recognition |
| `chamfer_distance.wgsl` | Point cloud distance | 3D geometry |

---

## Convolutions (11)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `conv1d.wgsl` | 1D convolution | Time series, audio |
| `conv2d.wgsl` | 2D convolution | Image, vision |
| `conv3d.wgsl` | 3D convolution | Video, medical |
| `depthwise_conv2d.wgsl` | Depthwise separable | MobileNet, efficient |
| `grouped_conv2d.wgsl` | Grouped convolution | ResNeXt, efficient |
| `dilated_conv2d.wgsl` | Atrous convolution | Large receptive field |
| `deformable_conv2d.wgsl` | Deformable convolution | Adaptive sampling |
| `gated_conv2d.wgsl` | Gated convolution | Image generation |

---

## Pooling (17)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `maxpool2d.wgsl` | Max pooling 2D | Spatial downsample |
| `avgpool2d.wgsl` | Average pooling 2D | Smooth downsample |
| `adaptive_avgpool2d.wgsl` | Adaptive average pool | Variable input size |
| `adaptive_maxpool2d.wgsl` | Adaptive max pool | Variable input size |
| `global_avgpool.wgsl` | Global average pool | Classification head |
| `global_maxpool.wgsl` | Global max pool | Feature extraction |
| `roi_pool.wgsl` | RoI pooling | Object detection |
| `roi_align.wgsl` | RoI align (interpolation) | Mask R-CNN |

---

## Normalization (27)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `batch_norm.wgsl` | Batch normalization | CNN training |
| `layer_norm.wgsl` | Layer normalization | Transformers, RNNs |
| `group_norm.wgsl` | Group normalization | Small batch size |
| `instance_norm.wgsl` | Instance normalization | Style transfer |
| `rmsnorm.wgsl` | RMS normalization | Llama, modern LLMs |
| `adaptive_instance_norm.wgsl` | AdaIN | Style transfer |
| `spectral_norm.wgsl` | Spectral normalization | GAN stability |
| `weight_norm.wgsl` | Weight normalization | RNN training |

---

## Math Operations (68)

### Trigonometric
`cos.wgsl`, `sin.wgsl`, `tan.wgsl`, `acos.wgsl`, `asin.wgsl`, `atan.wgsl`

### Hyperbolic
`cosh.wgsl`, `sinh.wgsl`, `tanh.wgsl`, `acosh.wgsl`, `asinh.wgsl`, `atanh.wgsl`

### Exponential/Log
`exp.wgsl`, `log.wgsl`, `log2.wgsl`, `log10.wgsl`, `pow.wgsl`, `sqrt.wgsl`, `rsqrt.wgsl`

### Rounding
`floor.wgsl`, `ceil.wgsl`, `round.wgsl`, `trunc.wgsl`, `frac.wgsl`

### Comparison
`min.wgsl`, `max.wgsl`, `clamp.wgsl`, `sign.wgsl`, `abs.wgsl`

### Special
`erf.wgsl` (error function), `erfc.wgsl` (complementary error), `lgamma.wgsl` (log gamma)

---

## Tensor Operations (41)

### Concatenation/Stacking
`concat.wgsl`, `stack.wgsl`, `tile.wgsl`, `repeat.wgsl`

### Splitting
`chunk.wgsl`, `slice.wgsl`, `split.wgsl`, `narrow.wgsl`

### Reshaping
`reshape.wgsl`, `flatten.wgsl`, `squeeze.wgsl`, `unsqueeze.wgsl`, `view.wgsl`

### Transposition
`transpose.wgsl`, `permute.wgsl`, `movedim.wgsl`

### Indexing
`gather.wgsl`, `scatter.wgsl`, `index_select.wgsl`, `masked_select.wgsl`, `where_op.wgsl`

### Padding
`pad.wgsl`, `circular_pad.wgsl`, `reflection_pad.wgsl`, `replication_pad.wgsl`

---

## Optimizers (13)

| Shader | Description | Learning Rate |
|--------|-------------|---------------|
| `adam.wgsl` | Adaptive moment estimation | Auto-adaptive |
| `adamw.wgsl` | Adam with weight decay | Auto-adaptive + L2 |
| `sgd.wgsl` | Stochastic gradient descent | Fixed |
| `rmsprop.wgsl` | Root mean square prop | Adaptive |
| `lamb.wgsl` | Layer-wise adaptive moments | Large batch |
| `adagrad.wgsl` | Adaptive gradient | Sparse features |
| `adadelta.wgsl` | Extension of Adagrad | No learning rate |
| `nadam.wgsl` | Nesterov + Adam | Momentum + adaptive |
| `radam.wgsl` | Rectified Adam | Warmup |
| `adafactor.wgsl` | Memory-efficient Adam | Low memory |

---

## Reduction Operations (14)

| Shader | Description | Output Shape |
|--------|-------------|--------------|
| `sum_reduce.wgsl` | Sum all elements | Scalar |
| `mean_reduce.wgsl` | Average all elements | Scalar |
| `max_reduce.wgsl` | Max element | Scalar |
| `min_reduce.wgsl` | Min element | Scalar |
| `argmax.wgsl` | Index of max | Index |
| `argmin.wgsl` | Index of min | Index |
| `logsumexp.wgsl` | log(sum(exp(x))) | Scalar (stable) |
| `variance_reduce.wgsl` | Variance | Scalar |
| `std_reduce.wgsl` | Standard deviation | Scalar |
| `cumsum.wgsl` | Cumulative sum | Same shape |

---

## Special Functions (5)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `bessel_j0.wgsl` | Bessel J₀ | Cylindrical waves |
| `bessel_j1.wgsl` | Bessel J₁ | Cylindrical waves |
| `bessel_i0.wgsl` | Modified Bessel I₀ | Heat diffusion |
| `bessel_k0.wgsl` | Modified Bessel K₀ | Decaying fields |
| `spherical_harmonics.wgsl` | Y_lm (l=0..6) | Molecular orbitals |

---

## Audio/Signal (9)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `stft.wgsl` | Short-time Fourier transform | Spectrogram |
| `istft.wgsl` | Inverse STFT | Waveform synthesis |
| `mfcc.wgsl` | Mel-frequency cepstral coefficients | Speech features |
| `mel_scale.wgsl` | Mel filterbank | Perceptual scale |
| `spectrogram.wgsl` | Power spectrum | Audio visualization |
| `griffin_lim.wgsl` | Phase reconstruction | Audio from magnitude |

---

## Graph Neural Networks (6)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `gcn_conv.wgsl` | Graph convolution | Node classification |
| `gat_conv.wgsl` | Graph attention | Attention graphs |
| `gin_conv.wgsl` | Graph isomorphism | Expressive GNN |
| `sage_conv.wgsl` | GraphSAGE | Inductive learning |
| `edge_conv.wgsl` | Edge convolution | Point clouds |

---

## Detection/Vision (5)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `nms.wgsl` | Non-maximum suppression | Object detection |
| `anchor_generator.wgsl` | Generate anchor boxes | Region proposals |
| `box_iou.wgsl` | Intersection over Union | Box matching |
| `bbox_transform.wgsl` | BBox coordinate transform | Proposal refinement |

---

## Augmentation (10)

| Shader | Description | Use Case |
|--------|-------------|----------|
| `cutmix.wgsl` | CutMix augmentation | Classification |
| `mixup.wgsl` | MixUp augmentation | Classification |
| `color_jitter.wgsl` | Random color perturbation | Robustness |
| `elastic_transform.wgsl` | Elastic deformation | Medical imaging |
| `grid_mask.wgsl` | GridMask regularization | Vision models |
| `mosaic.wgsl` | Mosaic augmentation | Object detection |

---

## Miscellaneous (56)

### Matrix Operations
`matmul.wgsl`, `matmul_tiled.wgsl`, `batch_matmul.wgsl`, `dotproduct.wgsl`

### Embedding/Indexing
`embedding.wgsl`, `one_hot.wgsl`

### Quantization
`quantize.wgsl`, `dequantize.wgsl`, `fake_quantize.wgsl`

### Utilities
`u64_emu.wgsl` (64-bit emulation), `bucketize.wgsl`, `bincount.wgsl`, `unique.wgsl`, `nonzero.wgsl`

### Distance
`pairwise_distance.wgsl`, `cdist.wgsl`, `cosine_similarity.wgsl`

---

**Total: 786 shaders across 33 categories**

**Navigation**: 
- By function → Find category above
- By name → Use Ctrl+F / Cmd+F
- By use case → Check "Use Case" column

**Last Updated**: March 7, 2026
