# MindsEye (`MindseyeAll`)

A JVM deep learning, differentiable computing, and numerical optimization library built on **Java 22** (with Kotlin 2.1 support), **Java Foreign Function & Memory (FFM)** bindings for NVIDIA CUDA/cuDNN, and deterministic reference-counted resource management via [`refcount-core`](../java-reference-counter/core/README.md).

## Key Features

- **Differentiable DAG Computation Graphs** — Compose arbitrary directed acyclic neural networks (`DAGNetwork`, `PipelineNetwork`, `SimpleLossNetwork`) with reverse-mode automatic differentiation (`Result`, `Delta`, `DeltaSet`, `StateSet`).
- **Java 22 FFM CUDA & cuDNN Acceleration** — Direct native bindings (`com.simiacryptus.mindseye.lang.cudnn.ffm`) to `libcudart` (CUDA 12) and `libcudnn` (cuDNN 9) using `java.lang.foreign`—no JNI wrapper JARs required.
- **Rich CPU & GPU Layer Libraries** — 70+ pure-Java CPU layers (`com.simiacryptus.mindseye.layers.java`) and 40+ cuDNN-accelerated GPU layers (`com.simiacryptus.mindseye.layers.cudnn` and `layers.cudnn.conv`).
- **Deterministic Memory Management** — Integrates `com.simiacryptus:refcount-core` and the `com.simiacryptus.refcount-autocoder` Gradle plugin to deterministically free CPU tensors and GPU device buffers without waiting for GC.
- **First- and Second-Order Optimizers** — Pluggable trainers (`IterativeTrainer`, `ValidatingTrainer`, `RoundRobinTrainer`), quasi-Newton & gradient orientation strategies (`LBFGS`, `OwlQn`, `QQN`, `MomentumStrategy`, `GradientDescent`, `TrustRegionStrategy`), line searches (`ArmijoWolfeSearch`, `QuadraticSearch`, `BisectionSearch`), and trust-region constraints.
- **Notebook & Experiment Reporting** — Built-in Markdown, HTML, and PDF report generation (`com.simiacryptus.notebook`) with embedded plots, tables, and image outputs.

## Documentation Guides

- [Architecture & Core Concepts](docs/ARCHITECTURE.md) — Tensors, layers, DAG execution, reference counting, serialization, and notebook reporting
- [CUDA & Java 22 FFM Guide](docs/CUDA_FFM.md) — Native library loading, GPU memory management, driver/toolkit installation, and `CudaSettings`
- [Layers & Optimization Reference](docs/LAYERS_AND_OPTIMIZATION.md) — Complete CPU/GPU layer catalog, `Trainable` wrappers, optimizers, line searches, and trust regions

## Prerequisites

- **JDK 22** (automatically provisioned if using Gradle toolchains via Foojay resolver)
- **Sibling `java-reference-counter` checkout** at `../java-reference-counter` (wired via `includeBuild("../java-reference-counter")` in [settings.gradle.kts](settings.gradle.kts))
- **Optional (for GPU acceleration):** NVIDIA GPU driver, **CUDA Toolkit 12.6**, and **cuDNN 9**
  - On Ubuntu/Debian Linux, you can install them with [`scripts/install-cuda-drivers.sh`](scripts/install-cuda-drivers.sh) (supports `--dry-run` and `--skip-driver`).
  - On CPU-only machines, pass `-DDISABLE_CUDNN=true` to disable GPU execution.
- **JVM native access flag:** Any JVM invoking the FFM CUDA/cuDNN bindings must pass `--enable-native-access=ALL-UNNAMED` (automatically configured for Gradle test tasks).

## Building & Testing

```bash
# Build library and run default unit tests + JaCoCo coverage report
./gradlew build

# Run default unit tests (excludes @Tag("LayerTest"), @Tag("Report"), @Tag("Manual"))
./gradlew test

# Run intensive layer test suite (@Tag("LayerTest"))
./gradlew layerTests

# Run report and dataset tests (@Tag("Report"))
./gradlew reportTests

# Run CPU-only without CUDA/cuDNN
./gradlew test -DDISABLE_CUDNN=true
```

Test task defaults configured in [build.gradle.kts](build.gradle.kts):
- Adds `--enable-native-access=ALL-UNNAMED` to all `Test` JVMs.
- Sets `TEST_REPO` to `build/reports/mindseye` unless overridden via `-DTEST_REPO=...`.
- Prepends `/usr/local/cuda-12.6/lib64` and `/usr/local/cuda/lib64` (when present) to `LD_LIBRARY_PATH`.
- Generates JaCoCo XML and HTML reports at `build/reports/jacoco/test/html/index.html` after `./gradlew test`.

### Reference-Counting Autocoder Tasks

Because `id("com.simiacryptus.refcount-autocoder")` is applied in `build.gradle.kts`, all `refcount` Gradle tasks are available directly:

```bash
./gradlew refcountVerify
./gradlew refcountCheck
./gradlew refcountInsert
```

See [refcount-autocoder/README.md](../java-reference-counter/refcount-autocoder/README.md) for all plugin options and tasks.

## Quick Start

### 1. Build and Evaluate a `PipelineNetwork`

```java
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;

PipelineNetwork network = new PipelineNetwork();
try {
    network.add(new FullyConnectedLayer(new int[]{28, 28}, new int[]{128})).freeRef();
    network.add(new BiasLayer(128)).freeRef();
    network.add(new ReLuActivationLayer()).freeRef();
    network.add(new FullyConnectedLayer(new int[]{128}, new int[]{10})).freeRef();

    Tensor input = new Tensor(28, 28);
    Result result = network.eval(input);
    try {
        Tensor output = result.getData().get(0);
        try {
            // Inspect output.getData()
        } finally {
            output.freeRef();
        }
    } finally {
        result.freeRef();
    }
} finally {
    network.freeRef();
}
```

### 2. Train with `SimpleLossNetwork` and `IterativeTrainer`

```java
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.LBFGS;

import java.util.concurrent.TimeUnit;

// trainingData: Tensor[numExamples][2] where [i][0] is input and [i][1] is target
SimpleLossNetwork supervised = new SimpleLossNetwork(network.addRef(), new EntropyLossLayer());
SampledArrayTrainable trainable = new SampledArrayTrainable(trainingData, supervised, 1000);
IterativeTrainer trainer = new IterativeTrainer(trainable);
try {
    trainer.setOrientation(new LBFGS());
    trainer.setLineSearchFactory(name -> new ArmijoWolfeSearch());
    trainer.setMaxIterations(100);
    trainer.setTimeout(5, TimeUnit.MINUTES);
    double finalLoss = trainer.run();
} finally {
    trainer.freeRef();
}
```

## Package Overview

| Package | Responsibility |
|---|---|
| `com.simiacryptus.mindseye.lang` | Core primitives: `Layer`, `LayerBase`, `Tensor`, `TensorList`, `Result`, `Delta`, `DeltaSet`, `State`, `StateSet`, `CoreSettings` |
| `com.simiacryptus.mindseye.lang.cudnn` | High-level GPU management: `CudaSystem`, `CudaDevice`, `CudnnHandle`, `CudaMemory`, `CudaTensor`, `CudaTensorList`, `CudaSettings`, `Precision`, `GemmaCudaKernels` |
| `com.simiacryptus.mindseye.lang.cudnn.ffm` | Java 22 FFM bindings to `libcudart` (`JCuda`), `libcudnn` (`JCudnn`), `libcublas` (`JCublas`), `libnvrtc` (`JNvrtc`), and `libcuda` (`JCudaDriver`) |
| `com.simiacryptus.mindseye.network` | Computation graph containers: `DAGNetwork`, `PipelineNetwork`, `SimpleLossNetwork`, `SupervisedNetwork`, `DAGNode`, `GraphEvaluationContext` |
| `com.simiacryptus.mindseye.layers` | Common/meta layers (`WrapperLayer`, `ValueLayer`, `PlaceholderLayer`, `MonitoringWrapperLayer`, `LoggingWrapperLayer`, `Explodable`) |
| `com.simiacryptus.mindseye.layers.java` | Pure-Java CPU layers (activations, linear projections, attention/RoPE/RMSNorm/embeddings, pooling, loss, image ops, noise, reductions) |
| `com.simiacryptus.mindseye.models.gemma4` | Gemma 4 model family (`E2B`, `E4B`, `31B`, `26B-A4B` MoE): GGUF parser/quantizations, CPU & CUDA differentiable `DAGNetwork` builders (`GemmaMoELayer`, `CudaGemmaMoELayer`), tokenizer, chat format, and KV-cached generator |
| `com.simiacryptus.mindseye.layers.cudnn` | CUDA/cuDNN/cuBLAS-accelerated GPU layers (activations, transformer/GQA/RoPE/RMSNorm/embeddings, pooling, softmax, LRN, Gramian, image ops, reductions) |
| `com.simiacryptus.mindseye.layers.cudnn.conv` | cuDNN convolution and fully-connected layers (`ConvolutionLayer`, `SimpleConvolutionLayer`, `ExplodedConvolutionGrid`, `FullyConnectedLayer`) |
| `com.simiacryptus.mindseye.eval` | `Trainable` adapters (`ArrayTrainable`, `SampledArrayTrainable`, `BatchedTrainable`, `CachedTrainable`, `L12Normalizer`) |
| `com.simiacryptus.mindseye.opt` | Trainers (`IterativeTrainer`, `ValidatingTrainer`, `RoundRobinTrainer`) plus `orient`, `line`, and `region` subpackages |
| `com.simiacryptus.notebook` | Markdown/HTML/PDF experiment notebooks (`NotebookOutput`, `MarkdownNotebookOutput`, `TableOutput`, `FileHTTPD`) |
| `com.simiacryptus.util` / `com.simiacryptus.math` | Serialization (JSON/Kryo/Zip), statistics, binary codecs, S3/Git helpers, monitoring, and test utilities |

## Common JVM System Properties

All settings classes (`CoreSettings`, `CudaSettings`, `RefSettings`) read **JVM system properties** (`-DKEY=VALUE`):

| Property | Scope | Description | Default |
|---|---|---|---|
| `THREADS` | `CoreSettings` / `RefSettings` | ForkJoinPool common parallelism | `64` |
| `SINGLE_THREADED` | `CoreSettings` | Disable parallel execution paths | `false` |
| `BACKPROP_AGG_SIZE` | `CoreSettings` | Backpropagation aggregation batch size | `3` |
| `DISABLE_CUDNN` | `CudaSettings` | Disable CUDA/cuDNN and run on CPU only | `false` |
| `CUDA_DEVICES` | `CudaSettings` | Comma-separated GPU device IDs (e.g., `0` or `0,1`) | `""` (all detected) |
| `CUDA_DEFAULT_PRECISION` | `CudaSettings` | Default GPU precision (`Float` or `Double`) | `Float` |
| `DEBUG_LIFECYCLE` | `RefSettings` | Record creation/`addRef`/`freeRef` stack traces to debug leaks | `false` |
| `TEST_REPO` | Tests | Output directory for generated notebook reports | `build/reports/mindseye` |

See [docs/CUDA_FFM.md](docs/CUDA_FFM.md) for the complete list of `CudaSettings` properties.

## License

Licensed under the Apache License, Version 2.0.
