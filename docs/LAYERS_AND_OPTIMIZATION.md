# Layers & Optimization Reference

This guide summarizes the neural network layers, `Trainable` adapters, trainers, orientation strategies, line searches, and trust-region constraints available in `MindseyeAll`.

## 1. Layer Libraries

### Common & Meta Layers (`com.simiacryptus.mindseye.layers`)

| Layer / Interface | Purpose |
|---|---|
| `WrapperLayer` | Base decorator forwarding evaluation to an inner `Layer` |
| `ValueLayer` | Constant/trainable parameter source emitting a `Tensor` |
| `PlaceholderLayer` | Named placeholder node for graph construction |
| `MonitoringWrapperLayer` | Tracks forward/backward timing and activation statistics |
| `LoggingWrapperLayer` / `LoggingLayer` | Logs intermediate tensor statistics during execution |
| `Explodable` | Interface for layers that can expand into lower-level subgraphs |
| `StochasticComponent` | Interface for layers with controllable RNG seeds (`shuffle(long)`) |

### Pure-Java CPU Layers (`com.simiacryptus.mindseye.layers.java` — 81 classes)

- **Activations:** `ReLuActivationLayer`, `GeluActivationLayer`, `SoftcapLayer`, `SigmoidActivationLayer`, `HyperbolicActivationLayer`, `SoftmaxLayer`, `LinearActivationLayer`, `ScalarScaleLayer`, `SimpleActivationLayer`, `AbsActivationLayer`, `SqActivationLayer`, `NthPowerActivationLayer`, `LogActivationLayer`, `GaussianActivationLayer`, `SinewaveActivationLayer`, `BoundedActivationLayer`, `BinaryEntropyActivationLayer`
- **Linear, Sequence & Transformer Primitives:** `FullyConnectedLayer`, `FullyConnectedReferenceLayer`, `LinearProjectionLayer`, `TokenEmbeddingLayer`, `RmsNormLayer`, `RopeLayer`, `GroupedQueryAttentionLayer`, `ChunkSelectLayer`, `BiasLayer`, `ProductLayer`, `ProductInputsLayer`, `SumInputsLayer`, `CrossProductLayer`, `CrossDifferenceLayer`, `VariableLayer`, `TargetValueLayer`

### Gemma 4 Model & GGUF Integration (`com.simiacryptus.mindseye.models.gemma4`)

- **Model & Graph Construction:** `Gemma4Model` (`toNetwork()`, `toCudaNetwork()`, GGUF loader, KV-cached autoregressive inference, and CLI), `Gemma4Network` (`buildNetwork`, `buildCudaNetwork`, `buildFromEmbeddingsNetwork`, `buildCudaFromEmbeddingsNetwork` supporting dense GeGLU, SWA + full causal GQA, dual RoPE with `rope_freqs.weight`, `V=K`, shared KV layers, per-layer input embeddings, `GemmaMoELayer`, and GPU-accelerated `CudaGemmaMoELayer`), `Gemma4Config`, `Gemma4Weights`
- **GGUF & Quantizations:** `GGUF` (v2/v3 parser and v3 binary builder), `GGMLType`, `GGUFTensor` (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `MXFP4`)
- **Tokenization, Chat & Sampling:** `GemmaTokenizer` (SentencePiece BPE + UTF-8 byte fallback), `GemmaChatFormat` (turn headers, system `<|think|>` turns, `<|channel>` thought stripping, FIM), `GemmaSampler` (`ARGMAX`, temperature categorical, `ToppSampler`)
- **Loss & Entropy:** `MeanSqLossLayer`, `EntropyLossLayer`, `EntropyLayer`, `AutoEntropyLayer`, `StaticScalarLossLayer`
- **Pooling, Reshaping & Concatenation:** `AvgPoolingLayer`, `MaxPoolingLayer`, `UnpoolingLayer`, `PhotoUnpoolingLayer`, `ReshapeLayer`, `TensorConcatLayer`, `SubBatchLayer`, `AssertDimensionsLayer`
- **Image Operations:** `ImgBandBiasLayer`, `ImgBandScaleLayer`, `ImgBandSelectLayer`, `ImgConcatLayer`, `ImgCropLayer`, `ImgZeroPaddingLayer`, `ImgReshapeLayer`, `ImgPixelGateLayer`, `ImgPixelSoftmaxLayer`, `ImgPixelSumLayer`, `ImgTileSelectLayer`, `ImgTileAssemblyLayer`, `ImgTileSubnetLayer`, `AffineImgViewLayer`, `ImgIndexMapViewLayer`, `ImgViewLayerBase`, `MaxImageBandLayer`, `RescaledSubnetLayer`, `StochasticSamplingSubnetLayer`
- **Reductions & Meta-Layers:** `SumReducerLayer`, `AvgReducerLayer`, `SignReducerLayer`, `SumMetaLayer`, `AvgMetaLayer`, `MaxMetaLayer`, `BiasMetaLayer`, `ScaleMetaLayer`, `StdDevMetaLayer`, `NormalizationMetaLayer`, `CrossDotMetaLayer`
- **Normalization, Noise & Monitoring:** `L1NormalizationLayer`, `DropoutNoiseLayer`, `MaxDropoutNoiseLayer`, `GaussianNoiseLayer`, `BinaryNoiseLayer`, `MonitoringSynapse`

### CUDA / cuDNN / cuBLAS GPU Layers (`com.simiacryptus.mindseye.layers.cudnn` & `conv` — 53 classes)

- **Transformer & Sequence Primitives (`layers.cudnn`):** `LinearProjectionLayer` (cuBLAS GEMM), `TokenEmbeddingLayer`, `RmsNormLayer` (learned & bare unit-RMS), `RopeLayer` (NeoX RoPE + `rope_freqs.weight`), `GroupedQueryAttentionLayer` (online-softmax FlashAttention-style GQA with causal & SWA), `ChunkSelectLayer`, `GeluActivationLayer`, `SoftcapLayer`, `ScalarScaleLayer` (plus `CudaGemmaMoELayer` in `models.gemma4`)
- **Convolutions & Dense (`layers.cudnn.conv`):** `ConvolutionLayer`, `SimpleConvolutionLayer`, `ExplodedConvolutionGrid`, `ExplodedConvolutionLeg`, `ConvolutionParams`, `FullyConnectedLayer`
- **Activations & Loss:** `ActivationLayer` (ReLU, Sigmoid, Tanh, Clipped ReLU, etc.), `SoftmaxActivationLayer`, `SquareActivationLayer`, `MeanSqLossLayer`
- **Pooling & Normalization:** `PoolingLayer` (max/avg pooling), `LRNLayer` (local response normalization), `GramianLayer` (style/Gram matrix)
- **Arithmetic & Reductions:** `ScaleLayer`, `ProductLayer`, `NProductLayer`, `BinarySumLayer`, `SumInputsLayer`, `GateBiasLayer`, `ValueLayer`, `SumReducerLayer`, `AvgReducerLayer`, `BandReducerLayer`, `BandAvgReducerLayer`
- **Image & Spatial Operations:** `ImgBandBiasLayer`, `ImgBandDynamicBiasLayer`, `ImgBandSelectLayer`, `ImgConcatLayer`, `ImgCropLayer`, `ImgPaddingLayer`, `ImgZeroPaddingLayer`, `SpatialReflectionPadding`, `ImgModulusCropLayer`, `ImgModulusPaddingLayer`, `ImgModulusPaddingSubnetLayer`, `ImgMinSizeLayer`, `ImgTileSelectLayer`, `ImgTileAssemblyLayer`, `ImgTileCycleLayer`, `ImgTileSubnetLayer`, `ImgLinearSubnetLayer`, `RescaledSubnetLayer`, `StochasticSamplingSubnetLayer`

## 2. Evaluation & `Trainable` Adapters (`com.simiacryptus.mindseye.eval`)

Optimizers operate against the **`Trainable`** interface (`measure(TrainingMonitor)` returning a `PointSample` and `reseed(long)` for stochastic resampling):

| Class | Description |
|---|---|
| `ArrayTrainable` | Evaluates a `Layer` over an in-memory `Tensor[][]` dataset |
| `SampledArrayTrainable` | Evaluates over a random minibatch of size `N` drawn from `Tensor[][]`, reshuffled on `reseed(...)` |
| `BatchedTrainable` | Splits large evaluations into smaller sub-batches to bound memory usage |
| `CachedTrainable` / `SampledCachedTrainable` | Caches evaluated points to avoid redundant forward/backward passes |
| `TensorListTrainable` | Evaluates directly over a `TensorList[]` batch |
| `L12Normalizer` / `ConstL12Normalizer` | Wraps any `Trainable` to add L1 and/or L2 weight regularization penalties |
| `TrainableDataMask` | Masks specific parameter groups during optimization |
| `BasicTrainable` | Direct optimization of a single `Layer` with no external dataset |

## 3. Optimization Framework (`com.simiacryptus.mindseye.opt`)

### Trainers

- **`IterativeTrainer`** — Core optimization loop combining an `OrientationStrategy`, a `LineSearchStrategy` factory, a `TrainingMonitor`, iteration limits (`setMaxIterations`), timeout (`setTimeout`), and resampling cadence (`setIterationsPerSample`).
- **`ValidatingTrainer`** — Multi-phase trainer (`getRegimen()`) that tracks a separate validation `Trainable` alongside the training `Trainable`.
- **`RoundRobinTrainer`** — Alternates optimization across multiple `Trainable` objectives.
- **`LayerRateDiagnosticTrainer`** — Diagnostic trainer for inspecting per-layer step sensitivity.

### Orientation Strategies (`com.simiacryptus.mindseye.opt.orient`)

| Strategy | Description |
|---|---|
| `LBFGS` | Limited-memory BFGS quasi-Newton optimizer (default in `IterativeTrainer`) |
| `OwlQn` | Orthant-Wise Limited-memory Quasi-Newton for L1-regularized objectives |
| `QQN` | Quadratic Quasi-Newton combining L-BFGS and gradient directions |
| `GradientDescent` | Steepest descent |
| `MomentumStrategy` | Gradient descent with momentum |
| `RecursiveSubspace` | Subspace-based quasi-Newton orientation |
| `TrustRegionStrategy` | Projects search directions onto a `TrustRegion` constraint |
| `LayerReweightingStrategy` | Scales per-layer search directions |
| `ValidatingOrientationWrapper` / `QuantifyOrientationWrapper` / `DescribeOrientationWrapper` | Diagnostic and validation wrappers around any `OrientationStrategy` |

### Line Search Strategies (`com.simiacryptus.mindseye.opt.line`)

- **`ArmijoWolfeSearch`** — Backtracking/interpolation search enforcing Armijo–Wolfe conditions (default in `IterativeTrainer`)
- **`QuadraticSearch`** — Parabolic interpolation line search
- **`BisectionSearch`** — Interval bisection line search
- **`StaticLearningRate`** — Fixed step size (no line search)
- **`FailsafeLineSearchCursor`** — Wraps a cursor to recover gracefully from NaN/overflow steps

### Trust Regions & Constraints (`com.simiacryptus.mindseye.opt.region`)

Used with `TrustRegionStrategy` to constrain parameter updates:
- **Norm & Sphere Bounds:** `AdaptiveTrustSphere`, `GrowthSphere`, `DistanceConstraint`, `FixedMagnitudeConstraint`
- **Structural Constraints:** `RangeConstraint`, `LinearSumConstraint`, `OrthonormalConstraint`, `ProportionalityConstraint`, `SingleOrthant`, `MeanVarianceGradient`, `StaticConstraint`
- **Composition:** `CompoundRegion`, `LayerTrustRegionMap` (apply different constraints per layer)
