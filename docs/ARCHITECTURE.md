# MindsEye Architecture & Core Concepts

This document describes the core runtime abstractions, computation graph execution model, reference-counted memory lifecycle, serialization, and notebook reporting in `MindseyeAll`.

## 1. Core Primitives (`com.simiacryptus.mindseye.lang`)

### Tensors and Batches

- **`Tensor`** — Multi-dimensional `double[]`-backed array (backed by `RecycleBin.DOUBLES` for buffer reuse) with arbitrary integer dimensions (`int[]`). Supports elementwise arithmetic, coordinate indexing (`Coordinate`), image conversion, and JSON/binary serialization (`SerialPrecision`).
- **`TensorList`** — Interface representing an ordered batch of `Tensor` instances (`length()`, `get(int)`, `stream()`, `getDimensions()`).
  - **`TensorArray`** — Standard CPU-backed `TensorList`.
  - **`CudaTensorList`** — GPU-resident `TensorList` wrapping a `CudaTensor` on device memory.
  - **`ReshapedTensorList`** — Zero-copy reshaped view over an existing `TensorList`.

### Layers and Automatic Differentiation

Every differentiable operation implements **`Layer`** (typically by extending **`LayerBase`**):

```java
Result eval(Result... input);
```

Evaluating a layer returns a **`Result`**, which pairs:
1. **Forward output** (`TensorList getData()`) — the computed batch of tensors.
2. **Backward callback** (`Result.Accumulator`) — a closure invoked during backpropagation (`result.accumulate(DeltaSet<UUID> buffer, TensorList delta)`) that:
   - Accumulates parameter gradients into the `DeltaSet<UUID>` keyed by each layer's `UUID` (`Delta<UUID>`).
   - Propagates upstream input gradients by calling `accumulate(...)` on each input `Result` that is `isAlive()`.

Supporting state containers:
- **`State<T>` / `StateSet<T>`** — Mutable parameter vectors (weights/biases) indexed by layer `UUID`, supporting vector-space operations used by optimizers (scaling, addition, dot products, norms).
- **`Delta<T>` / `DeltaSet<T>`** — Gradient accumulators matching the structure of `StateSet<T>`.
- **`PointSample`** — Pairs a `StateSet<UUID>` parameter point with its evaluated scalar objective (`sum`), gradient (`delta`), and step metadata (`rate`).

## 2. Computation Graphs (`com.simiacryptus.mindseye.network`)

- **`DAGNetwork`** — A `LayerBase` subclass representing a general directed acyclic graph of `DAGNode`s:
  - **`InputNode`** — Graph input placeholder (`getInput(int index)`).
  - **`InnerNode`** — Wraps a `Layer` and its upstream `DAGNode[]` inputs.
  - **`CountingResult`** — Reference-counted fan-out coordinator that buffers gradients from multiple downstream consumers before backpropagating once to a shared node.
  - **`GraphEvaluationContext`** — Per-evaluation cache (`LazyRefHashMap`) ensuring each node in a diamond/branching DAG is evaluated once per forward pass.
- **`PipelineNetwork`** — Convenience `DAGNetwork` for sequential or branching pipelines. Maintains a `head` node so `add(layer)` chains onto the current head, while `add(layer, nodeA, nodeB)` supports multi-input merges and residual connections.
- **`SupervisedNetwork` & `SimpleLossNetwork`** — Two-input networks (`input[0]` = features, `input[1]` = target labels) that feed `input[0]` into a `student` network and pass `(studentOutput, input[1])` into a `loss` layer (such as `EntropyLossLayer` or `MeanSqLossLayer`).

## 3. Reference-Counted Memory Management

Because `Tensor`, `TensorList`, `Result`, `Layer`, `CudaMemory`, and `CudaTensor` can hold large CPU buffers or scarce GPU VRAM allocations, MindsEye uses deterministic reference counting from [`refcount-core`](../../java-reference-counter/core/README.md):

- Every `ReferenceCountingBase` object starts with `refCount = 1`.
- Passing ownership into a method or storing a reference requires calling `addRef()`.
- Releasing ownership requires calling `freeRef()`; when `refCount` reaches `0`, `_free()` immediately releases CPU buffers (back to `RecycleBin`) or GPU allocations (back to `CudaDevice` / `cudaFree`).
- **Autocoder integration:** `MindseyeAll` applies the `com.simiacryptus.refcount-autocoder` Gradle plugin (via `includeBuild("../java-reference-counter")`), allowing `./gradlew refcountVerify` and `./gradlew refcountInsert` to statically verify or instrument reference counting across the codebase.
- **Leak diagnostics:** Run with `-DDEBUG_LIFECYCLE=true` (and optionally `-DWATCH_CREATE=true`) to capture creation, `addRef()`, and `freeRef()` stack traces whenever an object is used after free or reclaimed by GC without being freed.

## 4. Serialization (`ZipSerializable` & `JsonUtil`)

Layers and entire `DAGNetwork` graphs support two persistence formats:
- **JSON (`Layer.getJson()` / `Layer.fromJson(JsonObject)`)** — Human-readable graph topology with inline or externalized weight tensors at configurable `SerialPrecision` (`Double`, `Float`, `Half`, `Int8`, etc.).
- **ZIP archives (`Layer.writeZip(...)` / `Layer.fromZip(ZipFile)`)** — Stores the JSON graph definition alongside binary weight entries for compact model checkpoints.
- **Kryo / Java Serialization** — `Layer` extends `Serializable`, and Kryo serializers are included for fast binary snapshots.

## 5. Experiment Notebooks (`com.simiacryptus.notebook`)

The `notebook` package generates self-contained experiment reports (Markdown, HTML, and PDF via Flexmark):
- **`NotebookOutput`** — Interface for structured logging (`h1`/`h2`/`h3`, `p`, `eval`, `image`, `file`, `out`) and timing.
- **`MarkdownNotebookOutput`** — Writes reports and assets (PNG images, plots, tables via `TableOutput`) to a directory (configured via `-DTEST_REPO=...`).
- **`NullNotebookOutput`** — No-op implementation for headless/production runs.
- **`FileHTTPD` / `NanoHTTPD`** — Lightweight embedded HTTP server for viewing live notebook outputs and interactive queries (`FormQuery`, `UploadImageQuery`, etc.).
- **`NotebookTestBase`** (in `src/test/java`) — Base JUnit 5 class used by `@Tag("Report")` tests to produce reproducible benchmark and training reports.
