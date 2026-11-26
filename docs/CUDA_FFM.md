# CUDA & Java 22 Foreign Function & Memory (FFM) Guide

`MindseyeAll` accelerates deep learning workloads on NVIDIA GPUs using **Java 22's Foreign Function & Memory (FFM) API** (`java.lang.foreign`) to call `libcudart` and `libcudnn` directly—replacing legacy JNI/JCuda native wrapper libraries.

## 1. How the FFM Bindings Work (`com.simiacryptus.mindseye.lang.cudnn.ffm`)

- **`NativeLibraryLoader`** loads shared libraries into `Arena.global()` via `SymbolLookup.libraryLookup(...)` and binds C functions via `Linker.nativeLinker().downcallHandle(...)`.
- **`JCuda`** binds the NVIDIA CUDA Runtime API (`libcudart`), searching candidates in order:
  1. `libcudart.so.12`
  2. `/usr/local/cuda-12.6/lib64/libcudart.so.12`
  3. `/usr/local/cuda/lib64/libcudart.so.12`
  4. `libcudart.so`, `/usr/local/cuda/lib64/libcudart.so`
  5. `cudart64_12.dll`, `cudart` (Windows)
- **`JCudnn`** binds the NVIDIA cuDNN 9 C API (`libcudnn`), searching candidates in order:
  1. `libcudnn.so.9`
  2. `/usr/local/cuda-12.6/lib64/libcudnn.so.9`
  3. `/usr/local/cuda/lib64/libcudnn.so.9`
  4. `libcudnn.so`, `/usr/local/cuda/lib64/libcudnn.so`
  5. `cudnn64_9.dll`, `cudnn` (Windows)
- **`JCublas`**, **`JNvrtc`**, and **`JCudaDriver`** bind `libcublas` (`cublasSgemm_v2` / `cublasDgemm_v2`), `libnvrtc` (runtime CUDA C++ to PTX compilation), and `libcuda` (`cuModuleLoadData` / `cuLaunchKernel`) for transformer GEMMs, RMSNorm, RoPE, FlashAttention-style GQA, and MoE kernels (`GemmaCudaKernels`).
- **Descriptors & Handles:** `NativeHandle`, `Pointer`, `cudaStream_t`, `cudnnHandle`, `cudnnTensorDescriptor`, `cudnnFilterDescriptor`, `cudnnConvolutionDescriptor`, `cudnnPoolingDescriptor`, `cudnnActivationDescriptor`, `cudnnLRNDescriptor`, `cudnnOpTensorDescriptor`, `cudnnReduceTensorDescriptor`, and matching constant enums mirror the cuDNN C API while remaining `@RefIgnore`d by the autocoder.

### Required JVM Flag

Because FFM performs restricted native downcalls, Java 22 requires:

```bash
--enable-native-access=ALL-UNNAMED
```

This flag is configured automatically for all Gradle `Test` tasks in `build.gradle.kts`, and must be passed whenever launching `MindseyeAll` applications directly with `java`.

## 2. Installing NVIDIA Drivers, CUDA 12.6, and cuDNN 9

On Ubuntu (20.04 / 22.04 / 24.04) or Debian (`x86_64` or `aarch64`), use the bundled installer script:

```bash
# Preview commands without making changes
./scripts/install-cuda-drivers.sh --dry-run

# Install CUDA Toolkit 12.6 + cuDNN 9 (auto-installs nvidia-open driver if none detected)
./scripts/install-cuda-drivers.sh

# Install toolkit + cuDNN only (keep existing driver)
./scripts/install-cuda-drivers.sh --skip-driver
```

Supported flags:
- `--install-driver` / `--skip-driver` — Force or skip NVIDIA driver installation (default: `auto`)
- `--driver-package <pkg>` — Override driver package (default: `nvidia-open`, e.g., `cuda-drivers`)
- `--cuda-version <X-Y>` — CUDA toolkit version with dash (default: `12-6`)
- `--cudnn-package <pkg>` — cuDNN package name (default: `cudnn9-cuda-12`)
- `--dry-run` — Print commands without executing

The script registers `/etc/ld.so.conf.d/cuda-12-6.conf`, runs `ldconfig`, writes `/etc/profile.d/cuda-12-6.sh`, and verifies `libcuda.so.1`, `libcublas.so.12`, `libnvrtc.so.12`, and `libcudnn.so.9`.

## 3. High-Level GPU Resource Management (`com.simiacryptus.mindseye.lang.cudnn`)

| Class | Role |
|---|---|
| `CudaSystem` | Discovers GPUs, manages per-device thread/handle pools, and executes GPU lambdas via `CudaSystem.run(gpu -> ...)` |
| `CudaDevice` | Represents a single physical GPU, its `CudaMemory` allocator/recycle cache, `DeviceLimits`, and `DeviceMetrics` |
| `CudnnHandle` | Pooled cuDNN context + `CudaStream` bound to a specific `CudaDevice` |
| `CudaMemory` | Reference-counted device memory allocation (`MemoryType`: `Device`, `Managed`, `Host`) that recycles or frees VRAM when `refCount == 0` |
| `CudaTensor` | Reference-counted `CudaMemory` paired with a `cudnnTensorDescriptor` and `Precision` (`Float` or `Double`) |
| `CudaTensorList` | GPU-backed `TensorList` implementation enabling zero-copy layer-to-layer GPU pipelines |
| `Precision` | `Float` (FP32, 4 bytes) and `Double` (FP64, 8 bytes) with conversion helpers |
| `MultiPrecision` | Interface implemented by cuDNN layers supporting runtime precision switching |

## 4. `CudaSettings` Configuration Reference

All settings are read from **JVM system properties** (`-DKEY=VALUE`):

| System Property | Description | Default |
|---|---|---|
| `DISABLE_CUDNN` | Disable CUDA/cuDNN globally and fall back to CPU | `false` |
| `CUDA_DEVICES` | Comma-separated GPU indices to use (e.g. `0` or `0,1`) | `""` (all detected) |
| `FORCE_SINGLE_GPU` | Restrict execution to a single GPU | `false` |
| `CUDA_DEFAULT_PRECISION` | Default `Precision` (`Float` or `Double`) | `Float` |
| `CUDA_HANDLES_PER_DEVICE` | Number of pooled `CudnnHandle`s per GPU | `16` |
| `CUDA_CACHE_MODE` | `PersistanceMode` for GPU memory recycle cache (`STRONG`, `SOFT`, `WEAK`, `NULL`) | `WEAK` |
| `CUDA_CACHE_TTL` | Recycle cache TTL in seconds | `15` |
| `MAX_TOTAL_MEMORY` | Global GPU memory allocation cap (bytes) | `32 GiB` |
| `MAX_ALLOC_SIZE` | Maximum single allocation size (bytes) | `~4 GiB` (`4 * (Integer.MAX_VALUE / 2 - 1)`) |
| `MAX_IO_ELEMENTS` | Maximum I/O elements per batch | `512 MiB` |
| `MAX_FILTER_ELEMENTS` | Maximum convolution filter elements | `512 MiB` |
| `CONVOLUTION_WORKSPACE_SIZE_LIMIT` | Maximum cuDNN convolution workspace size (bytes) | `512 MiB` |
| `CUDA_MANAGED_MEM` | Enable CUDA unified/managed memory (`cudaMallocManaged`) | `false` |
| `SYNC_BEFORE_FREE` | Synchronize device before freeing GPU memory | `false` |
| `CUDA_PROFILE_MEM_IO` | Enable GPU memory I/O profiling | `false` |
| `CUDA_LOG_STACK` | Log stack traces on CUDA operations | `false` |
| `CONV_PARA_1` / `CONV_PARA_2` / `CONV_PARA_3` | Parallelization toggles for convolution backward/forward passes | `true` / `false` / `false` |

## 5. Running Without a GPU & Troubleshooting

- **CPU-only mode:** Pass `-DDISABLE_CUDNN=true` (e.g., `./gradlew test -DDISABLE_CUDNN=true`).
- **`UnsatisfiedLinkError: Failed to load native library`:** Ensure CUDA 12 (`libcudart.so.12`) and cuDNN 9 (`libcudnn.so.9`) are installed and discoverable via `ldconfig` or `LD_LIBRARY_PATH` (e.g., `/usr/local/cuda-12.6/lib64`).
- **Restricted native access warning/error:** Add `--enable-native-access=ALL-UNNAMED` to your JVM arguments.
- **Verifying FFM bindings:** Run the dedicated FFM unit tests:
  ```bash
  ./gradlew test --tests "com.simiacryptus.mindseye.lang.cudnn.ffm.FfmBindingsTest"
  ```
