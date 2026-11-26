/*
 * Copyright (c) 2026 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.lang.cudnn;

import com.simiacryptus.mindseye.lang.cudnn.ffm.JCublas;
import com.simiacryptus.mindseye.lang.cudnn.ffm.JCuda;
import com.simiacryptus.mindseye.lang.cudnn.ffm.JCudaDriver;
import com.simiacryptus.mindseye.lang.cudnn.ffm.JNvrtc;
import com.simiacryptus.mindseye.lang.cudnn.ffm.Pointer;
import com.simiacryptus.mindseye.lang.cudnn.ffm.cudaDeviceProp;
import com.simiacryptus.mindseye.lang.cudnn.ffm.cudaMemcpyKind;
import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Runtime-compiled (NVRTC -&gt; PTX -&gt; CUDA Driver API) GPU kernels and cuBLAS dispatchers
 * for Gemma 4 transformer layers in both {@link Precision#Float} and {@link Precision#Double}.
 * <p>
 * All kernel arguments use raw device {@link Pointer}s and primitives so callers manage
 * {@link CudaMemory} reference counts locally in their {@link CudaSystem#run} scopes.
 */
@RefIgnore
public final class GemmaCudaKernels {

  private static volatile byte[] compiledPtx;
  private static final ConcurrentHashMap<Long, DeviceModule> MODULES = new ConcurrentHashMap<Long, DeviceModule>();

  private GemmaCudaKernels() {
  }

  private static final class DeviceModule {
    final MemorySegment module;
    final ConcurrentHashMap<String, MemorySegment> functions = new ConcurrentHashMap<String, MemorySegment>();

    DeviceModule(final MemorySegment module) {
      this.module = module;
    }

    MemorySegment fn(final String name) {
      MemorySegment existing = functions.get(name);
      if (existing != null) {
        return existing;
      }
      MemorySegment resolved = JCudaDriver.getFunction(module, name);
      functions.putIfAbsent(name, resolved);
      return resolved;
    }
  }

  private static DeviceModule getModule(final int deviceId) {
    long key = (((long) CudaSystem.gpuGeneration.get()) << 32) | (deviceId & 0xFFFFFFFFL);
    DeviceModule mod = MODULES.get(key);
    if (mod != null) {
      JCudaDriver.ensureContext(deviceId);
      return mod;
    }
    synchronized (GemmaCudaKernels.class) {
      mod = MODULES.get(key);
      if (mod != null) {
        JCudaDriver.ensureContext(deviceId);
        return mod;
      }
      if (compiledPtx == null) {
        cudaDeviceProp prop = CudaDevice.getDeviceProperties(deviceId);
        int major = prop.major > 0 ? prop.major : 7;
        int minor = prop.minor >= 0 ? prop.minor : 0;
        String arch = "compute_" + major + minor;
        compiledPtx = JNvrtc.compileToPtx(CUDA_SOURCE, "gemma4_kernels.cu", arch);
      }
      MemorySegment loaded = JCudaDriver.loadModulePtx(deviceId, compiledPtx);
      mod = new DeviceModule(loaded);
      MODULES.put(key, mod);
      return mod;
    }
  }

  private static String suffix(final Precision precision) {
    return precision == Precision.Double ? "_f64" : "_f32";
  }

  public static long align256(final long bytes) {
    return (bytes + 255L) & ~255L;
  }

  public static void writeDoubles(final Pointer deviceDst, final double[] hostSrc) {
    if (hostSrc.length == 0) {
      return;
    }
    int st = JCuda.cudaMemcpy(deviceDst, Pointer.to(hostSrc), (long) hostSrc.length * Double.BYTES,
        cudaMemcpyKind.cudaMemcpyHostToDevice);
    if (st != JCuda.cudaSuccess) {
      throw new RuntimeException("cudaMemcpyHostToDevice(double[]) failed: " + st);
    }
  }

  public static void readDoubles(final Pointer deviceSrc, final double[] hostDst) {
    if (hostDst.length == 0) {
      return;
    }
    int st = JCuda.cudaMemcpy(Pointer.to(hostDst), deviceSrc, (long) hostDst.length * Double.BYTES,
        cudaMemcpyKind.cudaMemcpyDeviceToHost);
    if (st != JCuda.cudaSuccess) {
      throw new RuntimeException("cudaMemcpyDeviceToHost(double[]) failed: " + st);
    }
  }

  public static void readFloatsAsDoubles(final Pointer deviceSrc, final double[] hostDst) {
    if (hostDst.length == 0) {
      return;
    }
    float[] tmp = new float[hostDst.length];
    int st = JCuda.cudaMemcpy(Pointer.to(tmp), deviceSrc, (long) tmp.length * Float.BYTES,
        cudaMemcpyKind.cudaMemcpyDeviceToHost);
    if (st != JCuda.cudaSuccess) {
      throw new RuntimeException("cudaMemcpyDeviceToHost(float[]) failed: " + st);
    }
    for (int i = 0; i < tmp.length; i++) {
      hostDst[i] = tmp[i];
    }
  }

  public static void readElemAsDoubles(final Precision precision, final Pointer deviceSrc, final double[] hostDst) {
    if (precision == Precision.Double) {
      readDoubles(deviceSrc, hostDst);
    } else {
      readFloatsAsDoubles(deviceSrc, hostDst);
    }
  }

  public static void readInts(final Pointer deviceSrc, final int[] hostDst) {
    if (hostDst.length == 0) {
      return;
    }
    int st = JCuda.cudaMemcpy(Pointer.to(hostDst), deviceSrc, (long) hostDst.length * Integer.BYTES,
        cudaMemcpyKind.cudaMemcpyDeviceToHost);
    if (st != JCuda.cudaSuccess) {
      throw new RuntimeException("cudaMemcpyDeviceToHost(int[]) failed: " + st);
    }
  }

  // =========================================================================
  // GEMM (LinearProjectionLayer + MoE Router)
  // =========================================================================

  /**
   * Computes {@code Y = W^T X} in column-major (equivalently {@code y[p, o] = sum_i w[o, i] * x[p, i]}).
   */
  public static void linearForward(final int deviceId, final Precision precision,
                                   final Pointer x, final Pointer w, final Pointer y,
                                   final int totalPositions, final int inputDim, final int outputDim) {
    if (precision == Precision.Double) {
      JCublas.dgemm(deviceId, JCublas.CUBLAS_OP_T, JCublas.CUBLAS_OP_N,
          outputDim, totalPositions, inputDim,
          1.0, w, inputDim, x, inputDim, 0.0, y, outputDim);
    } else {
      JCublas.sgemm(deviceId, JCublas.CUBLAS_OP_T, JCublas.CUBLAS_OP_N,
          outputDim, totalPositions, inputDim,
          1.0f, w, inputDim, x, inputDim, 0.0f, y, outputDim);
    }
  }

  /**
   * Computes {@code dX = W * dY} in column-major ({@code dx[p, i] = sum_o w[o, i] * dy[p, o]}).
   */
  public static void linearBackwardInput(final int deviceId, final Precision precision,
                                         final Pointer dy, final Pointer w, final Pointer dx,
                                         final int totalPositions, final int inputDim, final int outputDim) {
    if (precision == Precision.Double) {
      JCublas.dgemm(deviceId, JCublas.CUBLAS_OP_N, JCublas.CUBLAS_OP_N,
          inputDim, totalPositions, outputDim,
          1.0, w, inputDim, dy, outputDim, 0.0, dx, inputDim);
    } else {
      JCublas.sgemm(deviceId, JCublas.CUBLAS_OP_N, JCublas.CUBLAS_OP_N,
          inputDim, totalPositions, outputDim,
          1.0f, w, inputDim, dy, outputDim, 0.0f, dx, inputDim);
    }
  }

  /**
   * Computes {@code dW = X * dY^T} in column-major ({@code dw[o, i] = sum_p x[p, i] * dy[p, o]}).
   */
  public static void linearBackwardWeights(final int deviceId, final Precision precision,
                                           final Pointer x, final Pointer dy, final Pointer dw,
                                           final int totalPositions, final int inputDim, final int outputDim) {
    if (precision == Precision.Double) {
      JCublas.dgemm(deviceId, JCublas.CUBLAS_OP_N, JCublas.CUBLAS_OP_T,
          inputDim, outputDim, totalPositions,
          1.0, x, inputDim, dy, outputDim, 0.0, dw, inputDim);
    } else {
      JCublas.sgemm(deviceId, JCublas.CUBLAS_OP_N, JCublas.CUBLAS_OP_T,
          inputDim, outputDim, totalPositions,
          1.0f, x, inputDim, dy, outputDim, 0.0f, dw, inputDim);
    }
  }

  // =========================================================================
  // RMSNorm
  // =========================================================================

  /**
   * Forward RMSNorm. {@code wDouble} is a device pointer to {@code normSize} FP64 weights, or {@code null} when {@code !hasWeights}.
   */
  public static void rmsNormForward(final int deviceId, final Precision precision,
                                    final Pointer x, final Pointer wDouble, final Pointer y,
                                    final int totalChunks, final int normSize, final double eps,
                                    final boolean hasWeights) {
    DeviceModule mod = getModule(deviceId);
    MemorySegment fn = mod.fn("rmsnorm_fwd" + suffix(precision));
    JCudaDriver.launchKernel(fn, totalChunks, 1, 1, 256, 1, 1, 0,
        x, wDouble, y, Integer.valueOf(totalChunks), Integer.valueOf(normSize), Double.valueOf(eps),
        Integer.valueOf(hasWeights ? 1 : 0));
  }

  /**
   * Backward RMSNorm computing {@code dx} (if non-null) and accumulating into zeroed FP64 {@code dwDouble} (if non-null).
   */
  public static void rmsNormBackward(final int deviceId, final Precision precision,
                                     final Pointer x, final Pointer dy, final Pointer wDouble,
                                     final Pointer dx, final Pointer dwDouble,
                                     final int totalChunks, final int normSize, final double eps,
                                     final boolean hasWeights) {
    DeviceModule mod = getModule(deviceId);
    MemorySegment fn = mod.fn("rmsnorm_bwd" + suffix(precision));
    JCudaDriver.launchKernel(fn, totalChunks, 1, 1, 256, 1, 1, 0,
        x, dy, wDouble, dx, dwDouble,
        Integer.valueOf(totalChunks), Integer.valueOf(normSize), Double.valueOf(eps),
        Integer.valueOf(hasWeights ? 1 : 0));
  }

  // =========================================================================
  // GELU, Softcap, ScalarScale
  // =========================================================================

  public static void geluForward(final int deviceId, final Precision precision,
                                 final Pointer x, final Pointer y, final int totalElements) {
    DeviceModule mod = getModule(deviceId);
    int grid = (totalElements + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("gelu_fwd" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        x, y, Integer.valueOf(totalElements));
  }

  public static void geluBackward(final int deviceId, final Precision precision,
                                  final Pointer x, final Pointer dy, final Pointer dx, final int totalElements) {
    DeviceModule mod = getModule(deviceId);
    int grid = (totalElements + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("gelu_bwd" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        x, dy, dx, Integer.valueOf(totalElements));
  }

  public static void softcapForward(final int deviceId, final Precision precision,
                                    final Pointer x, final Pointer y, final double cap, final int totalElements) {
    DeviceModule mod = getModule(deviceId);
    int grid = (totalElements + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("softcap_fwd" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        x, y, Double.valueOf(cap), Integer.valueOf(totalElements));
  }

  public static void softcapBackward(final int deviceId, final Precision precision,
                                     final Pointer x, final Pointer dy, final Pointer dx,
                                     final double cap, final int totalElements) {
    DeviceModule mod = getModule(deviceId);
    int grid = (totalElements + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("softcap_bwd" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        x, dy, dx, Double.valueOf(cap), Integer.valueOf(totalElements));
  }

  public static void scalarScaleForward(final int deviceId, final Precision precision,
                                        final Pointer src, final Pointer dst,
                                        final double scale, final int totalElements) {
    DeviceModule mod = getModule(deviceId);
    int grid = (totalElements + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("scalar_scale" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        src, dst, Double.valueOf(scale), Integer.valueOf(totalElements));
  }

  /**
   * Accumulates {@code sum_i (a[i] * b[i])} into zero-initialized single-FP64 {@code outDouble}.
   */
  public static void dotReduceToDouble(final int deviceId, final Precision precision,
                                       final Pointer a, final Pointer b, final Pointer outDouble,
                                       final int totalElements) {
    DeviceModule mod = getModule(deviceId);
    int grid = Math.min(1024, (totalElements + 255) / 256);
    JCudaDriver.launchKernel(mod.fn("dot_reduce" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        a, b, outDouble, Integer.valueOf(totalElements));
  }

  // =========================================================================
  // TokenEmbedding, RoPE, ChunkSelect
  // =========================================================================

  public static void tokenEmbeddingForward(final int deviceId, final Precision precision,
                                           final Pointer tokenIds, final Pointer table, final Pointer out,
                                           final Pointer errFlagInt,
                                           final int totalTokens, final int vocabSize,
                                           final int embeddingDim, final double scale) {
    DeviceModule mod = getModule(deviceId);
    int total = totalTokens * embeddingDim;
    int grid = (total + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("token_emb_fwd" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        tokenIds, table, out, errFlagInt,
        Integer.valueOf(totalTokens), Integer.valueOf(vocabSize), Integer.valueOf(embeddingDim), Double.valueOf(scale));
  }

  public static void tokenEmbeddingBackward(final int deviceId, final Precision precision,
                                            final Pointer tokenIds, final Pointer dy, final Pointer dTableDouble,
                                            final int totalTokens, final int vocabSize,
                                            final int embeddingDim, final double scale) {
    DeviceModule mod = getModule(deviceId);
    int total = totalTokens * embeddingDim;
    int grid = (total + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("token_emb_bwd" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        tokenIds, dy, dTableDouble,
        Integer.valueOf(totalTokens), Integer.valueOf(vocabSize), Integer.valueOf(embeddingDim), Double.valueOf(scale));
  }

  public static void ropeApply(final int deviceId, final Precision precision,
                               final Pointer src, final Pointer dst, final Pointer freqFactorsDouble,
                               final int batchSize, final int seqLen, final int numHeads, final int headSize,
                               final double ropeTheta, final int startPos, final boolean inverse) {
    DeviceModule mod = getModule(deviceId);
    int halfHead = headSize / 2;
    int totalPairs = batchSize * seqLen * numHeads * halfHead;
    int grid = (totalPairs + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("rope_apply" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        src, dst, freqFactorsDouble,
        Integer.valueOf(batchSize), Integer.valueOf(seqLen), Integer.valueOf(numHeads), Integer.valueOf(headSize),
        Double.valueOf(ropeTheta), Integer.valueOf(startPos), Integer.valueOf(inverse ? 1 : 0));
  }

  public static void chunkSelectForward(final int deviceId, final Precision precision,
                                        final Pointer src, final Pointer dst,
                                        final int totalPositions, final int chunkSize,
                                        final int totalChunks, final int chunkIndex) {
    DeviceModule mod = getModule(deviceId);
    int total = totalPositions * chunkSize;
    int grid = (total + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("chunk_select_fwd" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        src, dst, Integer.valueOf(totalPositions), Integer.valueOf(chunkSize),
        Integer.valueOf(totalChunks), Integer.valueOf(chunkIndex));
  }

  public static void chunkSelectBackward(final int deviceId, final Precision precision,
                                         final Pointer dy, final Pointer dxZeroed,
                                         final int totalPositions, final int chunkSize,
                                         final int totalChunks, final int chunkIndex) {
    DeviceModule mod = getModule(deviceId);
    int total = totalPositions * chunkSize;
    int grid = (total + 255) / 256;
    JCudaDriver.launchKernel(mod.fn("chunk_select_bwd" + suffix(precision)), grid, 1, 1, 256, 1, 1, 0,
        dy, dxZeroed, Integer.valueOf(totalPositions), Integer.valueOf(chunkSize),
        Integer.valueOf(totalChunks), Integer.valueOf(chunkIndex));
  }

  // =========================================================================
  // Grouped-Query Attention (FlashAttention-style online softmax)
  // =========================================================================

  /**
   * Forward GQA. {@code statsDouble} holds {@code 2 * batchSize * qSeqLen * numHeads} FP64 values {@code (m, l)}
   * for exact backward reconstruction without materializing {@code [qSeqLen, kvSeqLen]} matrices.
   */
  public static void gqaForward(final int deviceId, final Precision precision,
                                final Pointer q, final Pointer k, final Pointer v,
                                final Pointer out, final Pointer statsDouble,
                                final int batchSize, final int kBatch, final int vBatch,
                                final int qSeqLen, final int kvSeqLen,
                                final int numHeads, final int numKvHeads, final int headSize,
                                final double scale, final boolean causal, final int slidingWindow) {
    DeviceModule mod = getModule(deviceId);
    int sharedBytes = (64 + headSize) * Double.BYTES;
    JCudaDriver.launchKernel(mod.fn("gqa_fwd" + suffix(precision)),
        qSeqLen, numHeads, batchSize,
        64, 1, 1,
        sharedBytes,
        q, k, v, out, statsDouble,
        Integer.valueOf(batchSize), Integer.valueOf(kBatch), Integer.valueOf(vBatch),
        Integer.valueOf(qSeqLen), Integer.valueOf(kvSeqLen),
        Integer.valueOf(numHeads), Integer.valueOf(numKvHeads), Integer.valueOf(headSize),
        Double.valueOf(scale), Integer.valueOf(causal ? 1 : 0), Integer.valueOf(slidingWindow));
  }

  /**
   * Backward GQA computing {@code dQ}, {@code dK} (zero-initialized before call), and {@code dV} (zero-initialized before call).
   */
  public static void gqaBackward(final int deviceId, final Precision precision,
                                 final Pointer q, final Pointer k, final Pointer v,
                                 final Pointer out, final Pointer statsDouble, final Pointer dOut,
                                 final Pointer dQ, final Pointer dKZeroed, final Pointer dVZeroed,
                                 final int batchSize, final int kBatch, final int vBatch,
                                 final int qSeqLen, final int kvSeqLen,
                                 final int numHeads, final int numKvHeads, final int headSize,
                                 final double scale, final boolean causal, final int slidingWindow) {
    DeviceModule mod = getModule(deviceId);
    int sharedBytes = (128 + headSize) * Double.BYTES;
    JCudaDriver.launchKernel(mod.fn("gqa_bwd" + suffix(precision)),
        qSeqLen, numHeads, batchSize,
        64, 1, 1,
        sharedBytes,
        q, k, v, out, statsDouble, dOut, dQ, dKZeroed, dVZeroed,
        Integer.valueOf(batchSize), Integer.valueOf(kBatch), Integer.valueOf(vBatch),
        Integer.valueOf(qSeqLen), Integer.valueOf(kvSeqLen),
        Integer.valueOf(numHeads), Integer.valueOf(numKvHeads), Integer.valueOf(headSize),
        Double.valueOf(scale), Integer.valueOf(causal ? 1 : 0), Integer.valueOf(slidingWindow));
  }

  // =========================================================================
  // Gemma 4 Mixture-of-Experts (MoE) GPU Pipeline
  // =========================================================================

  public static long moeForwardWorkspaceBytes(final Precision precision, final int totalTokens,
                                              final int dim, final int expertCount,
                                              final int topK, final int expertFF) {
    long elem = precision.size;
    long off = 0L;
    off += align256((long) totalTokens * dim * elem);               // moeInput
    off += align256((long) totalTokens * dim * elem);               // routerIn
    off += align256((long) totalTokens * expertCount * elem);       // routerLogits
    off += align256((long) totalTokens * expertCount * 8L);         // probs (double)
    off += align256((long) totalTokens * topK * 4L);                // topExperts (int)
    off += align256((long) totalTokens * topK * 8L);                // topProbs (double)
    off += align256((long) totalTokens * topK * 2L * expertFF * elem); // gateUpBuf
    off += align256((long) totalTokens * topK * expertFF * elem);   // hiddenBuf
    off += align256((long) totalTokens * topK * dim * elem);        // downBuf
    off += align256((long) totalTokens * dim * elem);               // moeOut
    return off;
  }

  public static long moeBackwardWorkspaceBytes(final Precision precision, final int totalTokens,
                                               final int dim, final int expertCount,
                                               final int topK, final int expertFF) {
    long elem = precision.size;
    long off = moeForwardWorkspaceBytes(precision, totalTokens, dim, expertCount, topK, expertFF);
    off += align256((long) totalTokens * dim * elem);               // dMoeOut
    off += align256((long) totalTokens * topK * 8L);                // dTopProbs (double)
    off += align256((long) totalTokens * topK * 2L * expertFF * elem); // dGateUpBuf
    off += align256((long) totalTokens * dim * elem);               // dMoeInput
    off += align256((long) totalTokens * expertCount * elem);       // dRouterLogits
    off += align256((long) totalTokens * dim * elem);               // dRouterIn
    return off;
  }

  public static void moeForward(final int deviceId, final Precision precision,
                                final Pointer x, final Pointer out, final Pointer workspace,
                                final Pointer preNormDouble, final Pointer gateInpScaleDouble,
                                final Pointer gateInpElem, final Pointer gateUpExpsElem,
                                final Pointer downExpsElem, final Pointer downExpsScaleDouble,
                                final Pointer postNormDouble,
                                final int totalTokens, final int dim, final int expertCount,
                                final int topK, final int expertFF, final double eps) {
    long elem = precision.size;
    long off = 0L;
    Pointer moeInput = workspace.withByteOffset(off);
    off += align256((long) totalTokens * dim * elem);
    Pointer routerIn = workspace.withByteOffset(off);
    off += align256((long) totalTokens * dim * elem);
    Pointer routerLogits = workspace.withByteOffset(off);
    off += align256((long) totalTokens * expertCount * elem);
    Pointer probs = workspace.withByteOffset(off);
    off += align256((long) totalTokens * expertCount * 8L);
    Pointer topExperts = workspace.withByteOffset(off);
    off += align256((long) totalTokens * topK * 4L);
    Pointer topProbs = workspace.withByteOffset(off);
    off += align256((long) totalTokens * topK * 8L);
    Pointer gateUpBuf = workspace.withByteOffset(off);
    off += align256((long) totalTokens * topK * 2L * expertFF * elem);
    Pointer hiddenBuf = workspace.withByteOffset(off);
    off += align256((long) totalTokens * topK * expertFF * elem);
    Pointer downBuf = workspace.withByteOffset(off);
    off += align256((long) totalTokens * topK * dim * elem);
    Pointer moeOut = workspace.withByteOffset(off);

    DeviceModule mod = getModule(deviceId);
    String sfx = suffix(precision);

    // 1. Pre-norm & router input scaling
    rmsNormForward(deviceId, precision, x, preNormDouble, moeInput, totalTokens, dim, eps, true);
    JCudaDriver.launchKernel(mod.fn("moe_router_in_fwd" + sfx), totalTokens, 1, 1, 256, 1, 1, 0,
        x, gateInpScaleDouble, routerIn, Integer.valueOf(totalTokens), Integer.valueOf(dim), Double.valueOf(eps));

    // 2. Router GEMM + top-k softmax
    linearForward(deviceId, precision, routerIn, gateInpElem, routerLogits, totalTokens, dim, expertCount);
    int tokGrid = (totalTokens + 63) / 64;
    JCudaDriver.launchKernel(mod.fn("moe_topk_softmax" + sfx), tokGrid, 1, 1, 64, 1, 1, 0,
        routerLogits, probs, topExperts, topProbs,
        Integer.valueOf(totalTokens), Integer.valueOf(expertCount), Integer.valueOf(topK));

    // 3. Expert GeGLU + Down projection + weighted sum
    int ffGrid = (expertFF + 63) / 64;
    JCudaDriver.launchKernel(mod.fn("moe_gate_up_fwd" + sfx), ffGrid, topK, totalTokens, 64, 1, 1, 0,
        moeInput, gateUpExpsElem, topExperts, gateUpBuf, hiddenBuf,
        Integer.valueOf(totalTokens), Integer.valueOf(dim), Integer.valueOf(topK), Integer.valueOf(expertFF));

    int dimGrid = (dim + 63) / 64;
    JCudaDriver.launchKernel(mod.fn("moe_down_fwd" + sfx), dimGrid, totalTokens, 1, 64, 1, 1, 0,
        hiddenBuf, downExpsElem, downExpsScaleDouble, topExperts, topProbs, downBuf, moeOut,
        Integer.valueOf(totalTokens), Integer.valueOf(dim), Integer.valueOf(topK), Integer.valueOf(expertFF));

    // 4. Post-norm
    rmsNormForward(deviceId, precision, moeOut, postNormDouble, out, totalTokens, dim, eps, true);
  }

  public static void moeBackward(final int deviceId, final Precision precision,
                                 final Pointer x, final Pointer dy, final Pointer dxOrNull,
                                 final Pointer workspaceZeroed,
                                 final Pointer preNormDouble, final Pointer gateInpScaleDouble,
                                 final Pointer gateInpElem, final Pointer gateUpExpsElem,
                                 final Pointer downExpsElem, final Pointer downExpsScaleDouble,
                                 final Pointer postNormDouble,
                                 final Pointer gPreDouble, final Pointer gGisDouble,
                                 final Pointer gGiElem, final Pointer gGueDouble,
                                 final Pointer gDeDouble, final Pointer gDesDouble,
                                 final Pointer gPostDouble,
                                 final int totalTokens, final int dim, final int expertCount,
                                 final int topK, final int expertFF, final double eps,
                                 final boolean accW) {
    long elem = precision.size;
    long off = 0L;
    Pointer moeInput = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * dim * elem);
    Pointer routerIn = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * dim * elem);
    Pointer routerLogits = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * expertCount * elem);
    Pointer probs = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * expertCount * 8L);
    Pointer topExperts = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * topK * 4L);
    Pointer topProbs = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * topK * 8L);
    Pointer gateUpBuf = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * topK * 2L * expertFF * elem);
    Pointer hiddenBuf = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * topK * expertFF * elem);
    Pointer downBuf = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * topK * dim * elem);
    Pointer moeOut = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * dim * elem);

    Pointer dMoeOut = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * dim * elem);
    Pointer dTopProbs = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * topK * 8L);
    Pointer dGateUpBuf = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * topK * 2L * expertFF * elem);
    Pointer dMoeInput = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * dim * elem);
    Pointer dRouterLogits = workspaceZeroed.withByteOffset(off);
    off += align256((long) totalTokens * expertCount * elem);
    Pointer dRouterIn = workspaceZeroed.withByteOffset(off);

    DeviceModule mod = getModule(deviceId);
    String sfx = suffix(precision);

    // Recompute forward intermediates into workspace
    rmsNormForward(deviceId, precision, x, preNormDouble, moeInput, totalTokens, dim, eps, true);
    JCudaDriver.launchKernel(mod.fn("moe_router_in_fwd" + sfx), totalTokens, 1, 1, 256, 1, 1, 0,
        x, gateInpScaleDouble, routerIn, Integer.valueOf(totalTokens), Integer.valueOf(dim), Double.valueOf(eps));
    linearForward(deviceId, precision, routerIn, gateInpElem, routerLogits, totalTokens, dim, expertCount);
    int tokGrid = (totalTokens + 63) / 64;
    JCudaDriver.launchKernel(mod.fn("moe_topk_softmax" + sfx), tokGrid, 1, 1, 64, 1, 1, 0,
        routerLogits, probs, topExperts, topProbs,
        Integer.valueOf(totalTokens), Integer.valueOf(expertCount), Integer.valueOf(topK));
    int ffGrid = (expertFF + 63) / 64;
    JCudaDriver.launchKernel(mod.fn("moe_gate_up_fwd" + sfx), ffGrid, topK, totalTokens, 64, 1, 1, 0,
        moeInput, gateUpExpsElem, topExperts, gateUpBuf, hiddenBuf,
        Integer.valueOf(totalTokens), Integer.valueOf(dim), Integer.valueOf(topK), Integer.valueOf(expertFF));
    int dimGrid = (dim + 63) / 64;
    JCudaDriver.launchKernel(mod.fn("moe_down_fwd" + sfx), dimGrid, totalTokens, 1, 64, 1, 1, 0,
        hiddenBuf, downExpsElem, downExpsScaleDouble, topExperts, topProbs, downBuf, moeOut,
        Integer.valueOf(totalTokens), Integer.valueOf(dim), Integer.valueOf(topK), Integer.valueOf(expertFF));

    // B4: Backward through postFfwNorm2
    rmsNormBackward(deviceId, precision, moeOut, dy, postNormDouble, dMoeOut, accW ? gPostDouble : null,
        totalTokens, dim, eps, true);

    // B3a: Backward through down projections & GeGLU
    JCudaDriver.launchKernel(mod.fn("moe_down_bwd" + sfx), topK, totalTokens, 1, 256, 1, 1, 0,
        dMoeOut, downBuf, hiddenBuf, gateUpBuf, downExpsElem, downExpsScaleDouble,
        topExperts, topProbs, dTopProbs, dGateUpBuf,
        accW ? gDesDouble : null, accW ? gDeDouble : null,
        Integer.valueOf(totalTokens), Integer.valueOf(dim), Integer.valueOf(topK), Integer.valueOf(expertFF));

    // B3b: Backward through gateUpExps into dMoeInput (workspaceZeroed already zeroed dMoeInput!)
    JCudaDriver.launchKernel(mod.fn("moe_gate_up_bwd" + sfx), topK, totalTokens, 1, 256, 1, 1, 0,
        dGateUpBuf, moeInput, gateUpExpsElem, topExperts, dMoeInput,
        accW ? gGueDouble : null,
        Integer.valueOf(totalTokens), Integer.valueOf(dim), Integer.valueOf(topK), Integer.valueOf(expertFF));

    // B2: Backward through router softmax & router GEMM
    JCudaDriver.launchKernel(mod.fn("moe_router_softmax_bwd" + sfx), tokGrid, 1, 1, 64, 1, 1, 0,
        probs, topExperts, topProbs, dTopProbs, dRouterLogits,
        Integer.valueOf(totalTokens), Integer.valueOf(expertCount), Integer.valueOf(topK));

    if (accW && gGiElem != null) {
      linearBackwardWeights(deviceId, precision, routerIn, dRouterLogits, gGiElem, totalTokens, dim, expertCount);
    }
    linearBackwardInput(deviceId, precision, dRouterLogits, gateInpElem, dRouterIn, totalTokens, dim, expertCount);

    // B1: Fused backward through preFfwNorm2 and router bare RMS scaling into dx, gPre, gGis
    JCudaDriver.launchKernel(mod.fn("moe_input_norms_bwd" + sfx), totalTokens, 1, 1, 256, 1, 1, 0,
        x, dMoeInput, dRouterIn, preNormDouble, gateInpScaleDouble,
        dxOrNull, accW ? gPreDouble : null, accW ? gGisDouble : null,
        Integer.valueOf(totalTokens), Integer.valueOf(dim), Double.valueOf(eps));
  }

  // =========================================================================
  // Embedded CUDA C++ Source (compiled once via NVRTC)
  // =========================================================================

  private static final String CUDA_SOURCE = ""
      + "#if __CUDA_ARCH__ < 600\n"
      + "__device__ double atomicAddDouble(double* address, double val) {\n"
      + "  unsigned long long int* address_as_ull = (unsigned long long int*)address;\n"
      + "  unsigned long long int old = *address_as_ull, assumed;\n"
      + "  do {\n"
      + "    assumed = old;\n"
      + "    old = atomicCAS(address_as_ull, assumed,\n"
      + "                    __double_as_longlong(val + __longlong_as_double(assumed)));\n"
      + "  } while (assumed != old);\n"
      + "  return __longlong_as_double(old);\n"
      + "}\n"
      + "#else\n"
      + "__device__ __forceinline__ double atomicAddDouble(double* address, double val) {\n"
      + "  return atomicAdd(address, val);\n"
      + "}\n"
      + "#endif\n"
      + "\n"
      + "__device__ __forceinline__ float atomicAddElem(float* addr, double val) {\n"
      + "  return atomicAdd(addr, (float)val);\n"
      + "}\n"
      + "__device__ __forceinline__ double atomicAddElem(double* addr, double val) {\n"
      + "  return atomicAddDouble(addr, val);\n"
      + "}\n"
      + "\n"
      + "__device__ __forceinline__ double geluVal(double x) {\n"
      + "  const double k = 0.7978845608028654;\n"
      + "  double u = k * (x + 0.044715 * x * x * x);\n"
      + "  return 0.5 * x * (1.0 + tanh(u));\n"
      + "}\n"
      + "\n"
      + "__device__ __forceinline__ double geluDeriv(double x) {\n"
      + "  const double k = 0.7978845608028654;\n"
      + "  double x2 = x * x;\n"
      + "  double u = k * (x + 0.044715 * x * x2);\n"
      + "  double t = tanh(u);\n"
      + "  double du = k * (1.0 + 3.0 * 0.044715 * x2);\n"
      + "  return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;\n"
      + "}\n"
      + "\n"
      + "#define DEFINE_KERNELS(SFX, T) \\\n"
      + "extern \"C\" __global__ void rmsnorm_fwd##SFX( \\\n"
      + "    const T* x, const double* w, T* y, \\\n"
      + "    int totalChunks, int normSize, double eps, int hasWeights) { \\\n"
      + "  int c = blockIdx.x; \\\n"
      + "  if (c >= totalChunks) return; \\\n"
      + "  __shared__ double sdata[256]; \\\n"
      + "  long base = (long)c * normSize; \\\n"
      + "  double localSS = 0.0; \\\n"
      + "  for (int i = threadIdx.x; i < normSize; i += blockDim.x) { \\\n"
      + "    double xi = (double)x[base + i]; \\\n"
      + "    localSS += xi * xi; \\\n"
      + "  } \\\n"
      + "  sdata[threadIdx.x] = localSS; \\\n"
      + "  __syncthreads(); \\\n"
      + "  for (int s = 128; s > 0; s >>= 1) { \\\n"
      + "    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  double r = 1.0 / sqrt(sdata[0] / (double)normSize + eps); \\\n"
      + "  if (hasWeights && w != 0) { \\\n"
      + "    for (int i = threadIdx.x; i < normSize; i += blockDim.x) { \\\n"
      + "      y[base + i] = (T)(w[i] * r * (double)x[base + i]); \\\n"
      + "    } \\\n"
      + "  } else { \\\n"
      + "    for (int i = threadIdx.x; i < normSize; i += blockDim.x) { \\\n"
      + "      y[base + i] = (T)(r * (double)x[base + i]); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void rmsnorm_bwd##SFX( \\\n"
      + "    const T* x, const T* dy, const double* w, T* dx, double* dw, \\\n"
      + "    int totalChunks, int normSize, double eps, int hasWeights) { \\\n"
      + "  int c = blockIdx.x; \\\n"
      + "  if (c >= totalChunks) return; \\\n"
      + "  __shared__ double sdata[512]; \\\n"
      + "  long base = (long)c * normSize; \\\n"
      + "  double localSS = 0.0; \\\n"
      + "  double localDot = 0.0; \\\n"
      + "  for (int i = threadIdx.x; i < normSize; i += blockDim.x) { \\\n"
      + "    double xi = (double)x[base + i]; \\\n"
      + "    double dyi = (double)dy[base + i]; \\\n"
      + "    double ui = (hasWeights && w != 0) ? (dyi * w[i]) : dyi; \\\n"
      + "    localSS += xi * xi; \\\n"
      + "    localDot += ui * xi; \\\n"
      + "  } \\\n"
      + "  sdata[threadIdx.x] = localSS; \\\n"
      + "  sdata[256 + threadIdx.x] = localDot; \\\n"
      + "  __syncthreads(); \\\n"
      + "  for (int s = 128; s > 0; s >>= 1) { \\\n"
      + "    if (threadIdx.x < s) { \\\n"
      + "      sdata[threadIdx.x] += sdata[threadIdx.x + s]; \\\n"
      + "      sdata[256 + threadIdx.x] += sdata[256 + threadIdx.x + s]; \\\n"
      + "    } \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  double r = 1.0 / sqrt(sdata[0] / (double)normSize + eps); \\\n"
      + "  double coeff = ((r * r) / (double)normSize) * sdata[256]; \\\n"
      + "  if (dx != 0) { \\\n"
      + "    for (int i = threadIdx.x; i < normSize; i += blockDim.x) { \\\n"
      + "      double xi = (double)x[base + i]; \\\n"
      + "      double dyi = (double)dy[base + i]; \\\n"
      + "      double ui = (hasWeights && w != 0) ? (dyi * w[i]) : dyi; \\\n"
      + "      dx[base + i] = (T)(r * (ui - coeff * xi)); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "  if (hasWeights && dw != 0) { \\\n"
      + "    for (int i = threadIdx.x; i < normSize; i += blockDim.x) { \\\n"
      + "      atomicAddDouble(&dw[i], (double)dy[base + i] * r * (double)x[base + i]); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void gelu_fwd##SFX(const T* x, T* y, int n) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  if (idx < n) y[idx] = (T)geluVal((double)x[idx]); \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void gelu_bwd##SFX(const T* x, const T* dy, T* dx, int n) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  if (idx < n) dx[idx] = (T)((double)dy[idx] * geluDeriv((double)x[idx])); \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void softcap_fwd##SFX(const T* x, T* y, double cap, int n) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  if (idx < n) { \\\n"
      + "    double xv = (double)x[idx]; \\\n"
      + "    y[idx] = (T)(cap > 0.0 ? (cap * tanh(xv / cap)) : xv); \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void softcap_bwd##SFX(const T* x, const T* dy, T* dx, double cap, int n) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  if (idx < n) { \\\n"
      + "    double dyv = (double)dy[idx]; \\\n"
      + "    if (cap > 0.0) { \\\n"
      + "      double t = tanh((double)x[idx] / cap); \\\n"
      + "      dx[idx] = (T)(dyv * (1.0 - t * t)); \\\n"
      + "    } else { \\\n"
      + "      dx[idx] = (T)dyv; \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void scalar_scale##SFX(const T* src, T* dst, double scale, int n) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  if (idx < n) dst[idx] = (T)((double)src[idx] * scale); \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void dot_reduce##SFX(const T* a, const T* b, double* out, int n) { \\\n"
      + "  __shared__ double sdata[256]; \\\n"
      + "  double sum = 0.0; \\\n"
      + "  int stride = blockDim.x * gridDim.x; \\\n"
      + "  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) { \\\n"
      + "    sum += (double)a[i] * (double)b[i]; \\\n"
      + "  } \\\n"
      + "  sdata[threadIdx.x] = sum; \\\n"
      + "  __syncthreads(); \\\n"
      + "  for (int s = 128; s > 0; s >>= 1) { \\\n"
      + "    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  if (threadIdx.x == 0) atomicAddDouble(out, sdata[0]); \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void token_emb_fwd##SFX( \\\n"
      + "    const T* tokenIds, const T* table, T* out, int* errFlag, \\\n"
      + "    int totalTokens, int vocabSize, int embeddingDim, double scale) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  int total = totalTokens * embeddingDim; \\\n"
      + "  if (idx >= total) return; \\\n"
      + "  int t = idx / embeddingDim; \\\n"
      + "  int d = idx % embeddingDim; \\\n"
      + "  int tok = (int)llround((double)tokenIds[t]); \\\n"
      + "  if (tok < 0 || tok >= vocabSize) { \\\n"
      + "    if (errFlag != 0) atomicExch(errFlag, tok == 0 ? -1 : tok); \\\n"
      + "    out[idx] = (T)0; \\\n"
      + "    return; \\\n"
      + "  } \\\n"
      + "  out[idx] = (T)((double)table[(long)tok * embeddingDim + d] * scale); \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void token_emb_bwd##SFX( \\\n"
      + "    const T* tokenIds, const T* dy, double* dTable, \\\n"
      + "    int totalTokens, int vocabSize, int embeddingDim, double scale) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  int total = totalTokens * embeddingDim; \\\n"
      + "  if (idx >= total) return; \\\n"
      + "  int t = idx / embeddingDim; \\\n"
      + "  int d = idx % embeddingDim; \\\n"
      + "  int tok = (int)llround((double)tokenIds[t]); \\\n"
      + "  if (tok >= 0 && tok < vocabSize) { \\\n"
      + "    atomicAddDouble(&dTable[(long)tok * embeddingDim + d], (double)dy[idx] * scale); \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void rope_apply##SFX( \\\n"
      + "    const T* src, T* dst, const double* freqFactors, \\\n"
      + "    int batchSize, int seqLen, int numHeads, int headSize, \\\n"
      + "    double ropeTheta, int startPos, int inverse) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  int halfHead = headSize >> 1; \\\n"
      + "  int totalPairs = batchSize * seqLen * numHeads * halfHead; \\\n"
      + "  if (idx >= totalPairs) return; \\\n"
      + "  int ic = idx % halfHead; \\\n"
      + "  int rem = idx / halfHead; \\\n"
      + "  int h = rem % numHeads; \\\n"
      + "  rem /= numHeads; \\\n"
      + "  int t = rem % seqLen; \\\n"
      + "  int b = rem / seqLen; \\\n"
      + "  double base = 1.0 / pow(ropeTheta, (2.0 * (double)ic) / (double)headSize); \\\n"
      + "  double invFreq = (freqFactors != 0) ? (base / freqFactors[ic]) : base; \\\n"
      + "  double angle = (double)(startPos + t) * invFreq; \\\n"
      + "  double fcr = cos(angle); \\\n"
      + "  double fci = inverse ? -sin(angle) : sin(angle); \\\n"
      + "  long hOff = (((long)b * seqLen + t) * numHeads + h) * headSize; \\\n"
      + "  double v0 = (double)src[hOff + ic]; \\\n"
      + "  double v1 = (double)src[hOff + ic + halfHead]; \\\n"
      + "  dst[hOff + ic] = (T)(v0 * fcr - v1 * fci); \\\n"
      + "  dst[hOff + ic + halfHead] = (T)(v0 * fci + v1 * fcr); \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void chunk_select_fwd##SFX( \\\n"
      + "    const T* src, T* dst, int totalPositions, int chunkSize, int totalChunks, int chunkIndex) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  int total = totalPositions * chunkSize; \\\n"
      + "  if (idx >= total) return; \\\n"
      + "  int p = idx / chunkSize; \\\n"
      + "  int c = idx % chunkSize; \\\n"
      + "  long stride = (long)chunkSize * totalChunks; \\\n"
      + "  dst[idx] = src[(long)p * stride + (long)chunkIndex * chunkSize + c]; \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void chunk_select_bwd##SFX( \\\n"
      + "    const T* dy, T* dx, int totalPositions, int chunkSize, int totalChunks, int chunkIndex) { \\\n"
      + "  int idx = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  int total = totalPositions * chunkSize; \\\n"
      + "  if (idx >= total) return; \\\n"
      + "  int p = idx / chunkSize; \\\n"
      + "  int c = idx % chunkSize; \\\n"
      + "  long stride = (long)chunkSize * totalChunks; \\\n"
      + "  dx[(long)p * stride + (long)chunkIndex * chunkSize + c] = dy[idx]; \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void gqa_fwd##SFX( \\\n"
      + "    const T* q, const T* k, const T* v, T* out, double* statsOut, \\\n"
      + "    int batchSize, int kBatch, int vBatch, int qSeqLen, int kvSeqLen, \\\n"
      + "    int numHeads, int numKvHeads, int headSize, double scale, int causal, int slidingWindow) { \\\n"
      + "  extern __shared__ double smem[]; \\\n"
      + "  double* sRed = smem; \\\n"
      + "  double* sVec = smem + 64; \\\n"
      + "  int t = blockIdx.x; \\\n"
      + "  int h = blockIdx.y; \\\n"
      + "  int b = blockIdx.z; \\\n"
      + "  int tid = threadIdx.x; \\\n"
      + "  int qDim = numHeads * headSize; \\\n"
      + "  int kvDim = numKvHeads * headSize; \\\n"
      + "  int kvHead = h / (numHeads / numKvHeads); \\\n"
      + "  int posOffset = kvSeqLen - qSeqLen; \\\n"
      + "  int rawPos = posOffset + t; \\\n"
      + "  if (rawPos < 0) rawPos = 0; \\\n"
      + "  if (rawPos >= kvSeqLen) rawPos = kvSeqLen - 1; \\\n"
      + "  int endS = causal ? rawPos : (kvSeqLen - 1); \\\n"
      + "  int startS = (slidingWindow > 0 && endS - slidingWindow + 1 > 0) ? (endS - slidingWindow + 1) : 0; \\\n"
      + "  long qBase = ((long)b * qSeqLen + t) * qDim + (long)h * headSize; \\\n"
      + "  long kBatchOff = (long)(kBatch == 1 ? 0 : b) * kvSeqLen * kvDim; \\\n"
      + "  long vBatchOff = (long)(vBatch == 1 ? 0 : b) * kvSeqLen * kvDim; \\\n"
      + "  for (int d = tid; d < headSize; d += 64) sVec[d] = 0.0; \\\n"
      + "  __syncthreads(); \\\n"
      + "  double m = __longlong_as_double(0xfff0000000000000ULL); \\\n"
      + "  double l = 0.0; \\\n"
      + "  for (int s = startS; s <= endS; s++) { \\\n"
      + "    long kBase = kBatchOff + (long)s * kvDim + (long)kvHead * headSize; \\\n"
      + "    long vBase = vBatchOff + (long)s * kvDim + (long)kvHead * headSize; \\\n"
      + "    double dot = 0.0; \\\n"
      + "    for (int d = tid; d < headSize; d += 64) { \\\n"
      + "      dot += (double)q[qBase + d] * (double)k[kBase + d]; \\\n"
      + "    } \\\n"
      + "    sRed[tid] = dot; \\\n"
      + "    __syncthreads(); \\\n"
      + "    for (int st = 32; st > 0; st >>= 1) { \\\n"
      + "      if (tid < st) sRed[tid] += sRed[tid + st]; \\\n"
      + "      __syncthreads(); \\\n"
      + "    } \\\n"
      + "    double sc = sRed[0] * scale; \\\n"
      + "    double mNew = (sc > m) ? sc : m; \\\n"
      + "    double alpha = (l == 0.0) ? 0.0 : exp(m - mNew); \\\n"
      + "    double p = exp(sc - mNew); \\\n"
      + "    l = l * alpha + p; \\\n"
      + "    for (int d = tid; d < headSize; d += 64) { \\\n"
      + "      sVec[d] = sVec[d] * alpha + p * (double)v[vBase + d]; \\\n"
      + "    } \\\n"
      + "    m = mNew; \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  double invL = (l > 0.0) ? (1.0 / l) : 0.0; \\\n"
      + "  for (int d = tid; d < headSize; d += 64) { \\\n"
      + "    out[qBase + d] = (T)(sVec[d] * invL); \\\n"
      + "  } \\\n"
      + "  if (tid == 0 && statsOut != 0) { \\\n"
      + "    long stIdx = (((long)b * qSeqLen + t) * numHeads + h) * 2L; \\\n"
      + "    statsOut[stIdx] = m; \\\n"
      + "    statsOut[stIdx + 1] = l; \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void gqa_bwd##SFX( \\\n"
      + "    const T* q, const T* k, const T* v, const T* out, const double* statsIn, const T* dOut, \\\n"
      + "    T* dQ, T* dK, T* dV, \\\n"
      + "    int batchSize, int kBatch, int vBatch, int qSeqLen, int kvSeqLen, \\\n"
      + "    int numHeads, int numKvHeads, int headSize, double scale, int causal, int slidingWindow) { \\\n"
      + "  extern __shared__ double smem[]; \\\n"
      + "  double* sRed = smem; \\\n"
      + "  double* sVec = smem + 128; \\\n"
      + "  int t = blockIdx.x; \\\n"
      + "  int h = blockIdx.y; \\\n"
      + "  int b = blockIdx.z; \\\n"
      + "  int tid = threadIdx.x; \\\n"
      + "  int qDim = numHeads * headSize; \\\n"
      + "  int kvDim = numKvHeads * headSize; \\\n"
      + "  int kvHead = h / (numHeads / numKvHeads); \\\n"
      + "  int posOffset = kvSeqLen - qSeqLen; \\\n"
      + "  int rawPos = posOffset + t; \\\n"
      + "  if (rawPos < 0) rawPos = 0; \\\n"
      + "  if (rawPos >= kvSeqLen) rawPos = kvSeqLen - 1; \\\n"
      + "  int endS = causal ? rawPos : (kvSeqLen - 1); \\\n"
      + "  int startS = (slidingWindow > 0 && endS - slidingWindow + 1 > 0) ? (endS - slidingWindow + 1) : 0; \\\n"
      + "  long qBase = ((long)b * qSeqLen + t) * qDim + (long)h * headSize; \\\n"
      + "  long kBatchOff = (long)(kBatch == 1 ? 0 : b) * kvSeqLen * kvDim; \\\n"
      + "  long vBatchOff = (long)(vBatch == 1 ? 0 : b) * kvSeqLen * kvDim; \\\n"
      + "  long stIdx = (((long)b * qSeqLen + t) * numHeads + h) * 2L; \\\n"
      + "  double m; \\\n"
      + "  double l; \\\n"
      + "  if (statsIn != 0) { \\\n"
      + "    m = statsIn[stIdx]; \\\n"
      + "    l = statsIn[stIdx + 1]; \\\n"
      + "  } else { \\\n"
      + "    m = __longlong_as_double(0xfff0000000000000ULL); \\\n"
      + "    l = 0.0; \\\n"
      + "    for (int s = startS; s <= endS; s++) { \\\n"
      + "      long kBase = kBatchOff + (long)s * kvDim + (long)kvHead * headSize; \\\n"
      + "      double dot = 0.0; \\\n"
      + "      for (int d = tid; d < headSize; d += 64) { \\\n"
      + "        dot += (double)q[qBase + d] * (double)k[kBase + d]; \\\n"
      + "      } \\\n"
      + "      sRed[tid] = dot; \\\n"
      + "      __syncthreads(); \\\n"
      + "      for (int st = 32; st > 0; st >>= 1) { \\\n"
      + "        if (tid < st) sRed[tid] += sRed[tid + st]; \\\n"
      + "        __syncthreads(); \\\n"
      + "      } \\\n"
      + "      double sc = sRed[0] * scale; \\\n"
      + "      double mNew = (sc > m) ? sc : m; \\\n"
      + "      double alpha = (l == 0.0) ? 0.0 : exp(m - mNew); \\\n"
      + "      l = l * alpha + exp(sc - mNew); \\\n"
      + "      m = mNew; \\\n"
      + "      __syncthreads(); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "  double invL = (l > 0.0) ? (1.0 / l) : 0.0; \\\n"
      + "  double partD = 0.0; \\\n"
      + "  for (int d = tid; d < headSize; d += 64) { \\\n"
      + "    partD += (double)dOut[qBase + d] * (double)out[qBase + d]; \\\n"
      + "    sVec[d] = 0.0; \\\n"
      + "  } \\\n"
      + "  sRed[tid] = partD; \\\n"
      + "  __syncthreads(); \\\n"
      + "  for (int st = 32; st > 0; st >>= 1) { \\\n"
      + "    if (tid < st) sRed[tid] += sRed[tid + st]; \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  double rowDot = sRed[0]; \\\n"
      + "  for (int s = startS; s <= endS; s++) { \\\n"
      + "    long kBase = kBatchOff + (long)s * kvDim + (long)kvHead * headSize; \\\n"
      + "    long vBase = vBatchOff + (long)s * kvDim + (long)kvHead * headSize; \\\n"
      + "    double qk = 0.0; \\\n"
      + "    double dov = 0.0; \\\n"
      + "    for (int d = tid; d < headSize; d += 64) { \\\n"
      + "      qk += (double)q[qBase + d] * (double)k[kBase + d]; \\\n"
      + "      dov += (double)dOut[qBase + d] * (double)v[vBase + d]; \\\n"
      + "    } \\\n"
      + "    sRed[tid] = qk; \\\n"
      + "    sRed[64 + tid] = dov; \\\n"
      + "    __syncthreads(); \\\n"
      + "    for (int st = 32; st > 0; st >>= 1) { \\\n"
      + "      if (tid < st) { \\\n"
      + "        sRed[tid] += sRed[tid + st]; \\\n"
      + "        sRed[64 + tid] += sRed[64 + tid + st]; \\\n"
      + "      } \\\n"
      + "      __syncthreads(); \\\n"
      + "    } \\\n"
      + "    double a = exp(sRed[0] * scale - m) * invL; \\\n"
      + "    double dScore = scale * a * (sRed[64] - rowDot); \\\n"
      + "    for (int d = tid; d < headSize; d += 64) { \\\n"
      + "      if (dQ != 0) sVec[d] += dScore * (double)k[kBase + d]; \\\n"
      + "      if (dK != 0) atomicAddElem(&dK[kBase + d], dScore * (double)q[qBase + d]); \\\n"
      + "      if (dV != 0) atomicAddElem(&dV[vBase + d], a * (double)dOut[qBase + d]); \\\n"
      + "    } \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  if (dQ != 0) { \\\n"
      + "    for (int d = tid; d < headSize; d += 64) dQ[qBase + d] = (T)sVec[d]; \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void moe_router_in_fwd##SFX( \\\n"
      + "    const T* x, const double* gateInpScale, T* routerIn, \\\n"
      + "    int totalTokens, int dim, double eps) { \\\n"
      + "  int t = blockIdx.x; \\\n"
      + "  if (t >= totalTokens) return; \\\n"
      + "  __shared__ double sdata[256]; \\\n"
      + "  long base = (long)t * dim; \\\n"
      + "  double localSS = 0.0; \\\n"
      + "  for (int i = threadIdx.x; i < dim; i += blockDim.x) { \\\n"
      + "    double xi = (double)x[base + i]; \\\n"
      + "    localSS += xi * xi; \\\n"
      + "  } \\\n"
      + "  sdata[threadIdx.x] = localSS; \\\n"
      + "  __syncthreads(); \\\n"
      + "  for (int s = 128; s > 0; s >>= 1) { \\\n"
      + "    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s]; \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  double rX = 1.0 / sqrt(sdata[0] / (double)dim + eps); \\\n"
      + "  double routerRms = rX / sqrt((double)dim); \\\n"
      + "  for (int i = threadIdx.x; i < dim; i += blockDim.x) { \\\n"
      + "    routerIn[base + i] = (T)((double)x[base + i] * routerRms * gateInpScale[i]); \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void moe_topk_softmax##SFX( \\\n"
      + "    const T* logits, double* probs, int* topExperts, double* topProbs, \\\n"
      + "    int totalTokens, int expertCount, int topK) { \\\n"
      + "  int t = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  if (t >= totalTokens) return; \\\n"
      + "  long lBase = (long)t * expertCount; \\\n"
      + "  double maxVal = __longlong_as_double(0xfff0000000000000ULL); \\\n"
      + "  for (int e = 0; e < expertCount; e++) { \\\n"
      + "    double v = (double)logits[lBase + e]; \\\n"
      + "    if (v > maxVal) maxVal = v; \\\n"
      + "  } \\\n"
      + "  double sumExp = 0.0; \\\n"
      + "  for (int e = 0; e < expertCount; e++) { \\\n"
      + "    double ex = exp((double)logits[lBase + e] - maxVal); \\\n"
      + "    probs[lBase + e] = ex; \\\n"
      + "    sumExp += ex; \\\n"
      + "  } \\\n"
      + "  for (int e = 0; e < expertCount; e++) { \\\n"
      + "    probs[lBase + e] /= sumExp; \\\n"
      + "  } \\\n"
      + "  long kBase = (long)t * topK; \\\n"
      + "  for (int k = 0; k < topK; k++) { \\\n"
      + "    int bestIdx = 0; \\\n"
      + "    double bestVal = __longlong_as_double(0xfff0000000000000ULL); \\\n"
      + "    for (int e = 0; e < expertCount; e++) { \\\n"
      + "      int picked = 0; \\\n"
      + "      for (int prev = 0; prev < k; prev++) { \\\n"
      + "        if (topExperts[kBase + prev] == e) { picked = 1; break; } \\\n"
      + "      } \\\n"
      + "      if (!picked && probs[lBase + e] > bestVal) { \\\n"
      + "        bestVal = probs[lBase + e]; \\\n"
      + "        bestIdx = e; \\\n"
      + "      } \\\n"
      + "    } \\\n"
      + "    topExperts[kBase + k] = bestIdx; \\\n"
      + "    topProbs[kBase + k] = bestVal; \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void moe_gate_up_fwd##SFX( \\\n"
      + "    const T* moeInput, const T* gateUpExps, const int* topExperts, \\\n"
      + "    T* gateUpBuf, T* hiddenBuf, \\\n"
      + "    int totalTokens, int dim, int topK, int expertFF) { \\\n"
      + "  int h = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  int k = blockIdx.y; \\\n"
      + "  int t = blockIdx.z; \\\n"
      + "  if (h >= expertFF || k >= topK || t >= totalTokens) return; \\\n"
      + "  int e = topExperts[(long)t * topK + k]; \\\n"
      + "  int guDim = 2 * expertFF; \\\n"
      + "  long inBase = (long)t * dim; \\\n"
      + "  long gRow = ((long)e * guDim + h) * dim; \\\n"
      + "  long uRow = ((long)e * guDim + expertFF + h) * dim; \\\n"
      + "  double gSum = 0.0; \\\n"
      + "  double uSum = 0.0; \\\n"
      + "  for (int i = 0; i < dim; i++) { \\\n"
      + "    double xi = (double)moeInput[inBase + i]; \\\n"
      + "    gSum += (double)gateUpExps[gRow + i] * xi; \\\n"
      + "    uSum += (double)gateUpExps[uRow + i] * xi; \\\n"
      + "  } \\\n"
      + "  long tkSlot = (long)t * topK + k; \\\n"
      + "  gateUpBuf[tkSlot * guDim + h] = (T)gSum; \\\n"
      + "  gateUpBuf[tkSlot * guDim + expertFF + h] = (T)uSum; \\\n"
      + "  hiddenBuf[tkSlot * expertFF + h] = (T)(geluVal(gSum) * uSum); \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void moe_down_fwd##SFX( \\\n"
      + "    const T* hiddenBuf, const T* downExps, const double* downExpsScale, \\\n"
      + "    const int* topExperts, const double* topProbs, T* downBuf, T* moeOut, \\\n"
      + "    int totalTokens, int dim, int topK, int expertFF) { \\\n"
      + "  int d = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  int t = blockIdx.y; \\\n"
      + "  if (d >= dim || t >= totalTokens) return; \\\n"
      + "  double outVal = 0.0; \\\n"
      + "  for (int k = 0; k < topK; k++) { \\\n"
      + "    long tkSlot = (long)t * topK + k; \\\n"
      + "    int e = topExperts[tkSlot]; \\\n"
      + "    double prob = topProbs[tkSlot]; \\\n"
      + "    long wRow = ((long)e * dim + d) * expertFF; \\\n"
      + "    long hBase = tkSlot * expertFF; \\\n"
      + "    double sum = 0.0; \\\n"
      + "    for (int h = 0; h < expertFF; h++) { \\\n"
      + "      sum += (double)downExps[wRow + h] * (double)hiddenBuf[hBase + h]; \\\n"
      + "    } \\\n"
      + "    downBuf[tkSlot * dim + d] = (T)sum; \\\n"
      + "    outVal += prob * downExpsScale[e] * sum; \\\n"
      + "  } \\\n"
      + "  moeOut[(long)t * dim + d] = (T)outVal; \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void moe_down_bwd##SFX( \\\n"
      + "    const T* dMoeOut, const T* downBuf, const T* hiddenBuf, const T* gateUpBuf, \\\n"
      + "    const T* downExps, const double* downExpsScale, \\\n"
      + "    const int* topExperts, const double* topProbs, \\\n"
      + "    double* dTopProbs, T* dGateUpBuf, double* gDownScale, double* gDownExps, \\\n"
      + "    int totalTokens, int dim, int topK, int expertFF) { \\\n"
      + "  int k = blockIdx.x; \\\n"
      + "  int t = blockIdx.y; \\\n"
      + "  if (k >= topK || t >= totalTokens) return; \\\n"
      + "  __shared__ double sRed[256]; \\\n"
      + "  long tkSlot = (long)t * topK + k; \\\n"
      + "  int e = topExperts[tkSlot]; \\\n"
      + "  double prob = topProbs[tkSlot]; \\\n"
      + "  double scaleE = downExpsScale[e]; \\\n"
      + "  double wFactor = prob * scaleE; \\\n"
      + "  long dOutBase = (long)t * dim; \\\n"
      + "  long downBase = tkSlot * dim; \\\n"
      + "  double partDot = 0.0; \\\n"
      + "  for (int d = threadIdx.x; d < dim; d += 256) { \\\n"
      + "    partDot += (double)dMoeOut[dOutBase + d] * (double)downBuf[downBase + d]; \\\n"
      + "  } \\\n"
      + "  sRed[threadIdx.x] = partDot; \\\n"
      + "  __syncthreads(); \\\n"
      + "  for (int s = 128; s > 0; s >>= 1) { \\\n"
      + "    if (threadIdx.x < s) sRed[threadIdx.x] += sRed[threadIdx.x + s]; \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  if (threadIdx.x == 0) { \\\n"
      + "    if (gDownScale != 0) atomicAddDouble(&gDownScale[e], sRed[0] * prob); \\\n"
      + "    dTopProbs[tkSlot] = sRed[0] * scaleE; \\\n"
      + "  } \\\n"
      + "  long hBase = tkSlot * expertFF; \\\n"
      + "  if (gDownExps != 0) { \\\n"
      + "    int totalW = dim * expertFF; \\\n"
      + "    for (int idx = threadIdx.x; idx < totalW; idx += 256) { \\\n"
      + "      int d = idx / expertFF; \\\n"
      + "      int h = idx % expertFF; \\\n"
      + "      double dDown = wFactor * (double)dMoeOut[dOutBase + d]; \\\n"
      + "      atomicAddDouble(&gDownExps[((long)e * dim + d) * expertFF + h], dDown * (double)hiddenBuf[hBase + h]); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "  int guDim = 2 * expertFF; \\\n"
      + "  long guBase = tkSlot * guDim; \\\n"
      + "  for (int h = threadIdx.x; h < expertFF; h += 256) { \\\n"
      + "    double dh = 0.0; \\\n"
      + "    for (int d = 0; d < dim; d++) { \\\n"
      + "      double dDown = wFactor * (double)dMoeOut[dOutBase + d]; \\\n"
      + "      dh += dDown * (double)downExps[((long)e * dim + d) * expertFF + h]; \\\n"
      + "    } \\\n"
      + "    double gVal = (double)gateUpBuf[guBase + h]; \\\n"
      + "    double uVal = (double)gateUpBuf[guBase + expertFF + h]; \\\n"
      + "    dGateUpBuf[guBase + h] = (T)(dh * uVal * geluDeriv(gVal)); \\\n"
      + "    dGateUpBuf[guBase + expertFF + h] = (T)(dh * geluVal(gVal)); \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void moe_gate_up_bwd##SFX( \\\n"
      + "    const T* dGateUpBuf, const T* moeInput, const T* gateUpExps, const int* topExperts, \\\n"
      + "    T* dMoeInput, double* gGateUpExps, \\\n"
      + "    int totalTokens, int dim, int topK, int expertFF) { \\\n"
      + "  int k = blockIdx.x; \\\n"
      + "  int t = blockIdx.y; \\\n"
      + "  if (k >= topK || t >= totalTokens) return; \\\n"
      + "  long tkSlot = (long)t * topK + k; \\\n"
      + "  int e = topExperts[tkSlot]; \\\n"
      + "  int guDim = 2 * expertFF; \\\n"
      + "  long guBase = tkSlot * guDim; \\\n"
      + "  long inBase = (long)t * dim; \\\n"
      + "  if (gGateUpExps != 0) { \\\n"
      + "    int totalW = guDim * dim; \\\n"
      + "    for (int idx = threadIdx.x; idx < totalW; idx += 256) { \\\n"
      + "      int o = idx / dim; \\\n"
      + "      int i = idx % dim; \\\n"
      + "      atomicAddDouble(&gGateUpExps[((long)e * guDim + o) * dim + i], \\\n"
      + "                      (double)dGateUpBuf[guBase + o] * (double)moeInput[inBase + i]); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "  for (int i = threadIdx.x; i < dim; i += 256) { \\\n"
      + "    double sum = 0.0; \\\n"
      + "    for (int o = 0; o < guDim; o++) { \\\n"
      + "      sum += (double)dGateUpBuf[guBase + o] * (double)gateUpExps[((long)e * guDim + o) * dim + i]; \\\n"
      + "    } \\\n"
      + "    atomicAddElem(&dMoeInput[inBase + i], sum); \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void moe_router_softmax_bwd##SFX( \\\n"
      + "    const double* probs, const int* topExperts, const double* topProbs, const double* dTopProbs, \\\n"
      + "    T* dRouterLogits, int totalTokens, int expertCount, int topK) { \\\n"
      + "  int t = blockIdx.x * blockDim.x + threadIdx.x; \\\n"
      + "  if (t >= totalTokens) return; \\\n"
      + "  long kBase = (long)t * topK; \\\n"
      + "  double pDotDp = 0.0; \\\n"
      + "  for (int k = 0; k < topK; k++) { \\\n"
      + "    pDotDp += topProbs[kBase + k] * dTopProbs[kBase + k]; \\\n"
      + "  } \\\n"
      + "  long lBase = (long)t * expertCount; \\\n"
      + "  for (int e = 0; e < expertCount; e++) { \\\n"
      + "    double dp = 0.0; \\\n"
      + "    for (int k = 0; k < topK; k++) { \\\n"
      + "      if (topExperts[kBase + k] == e) { \\\n"
      + "        dp = dTopProbs[kBase + k]; \\\n"
      + "        break; \\\n"
      + "      } \\\n"
      + "    } \\\n"
      + "    dRouterLogits[lBase + e] = (T)(probs[lBase + e] * (dp - pDotDp)); \\\n"
      + "  } \\\n"
      + "} \\\n"
      + "extern \"C\" __global__ void moe_input_norms_bwd##SFX( \\\n"
      + "    const T* x, const T* dMoeInput, const T* dRouterIn, \\\n"
      + "    const double* preNormW, const double* gateInpScale, \\\n"
      + "    T* dx, double* gPreNorm, double* gGateInpScale, \\\n"
      + "    int totalTokens, int dim, double eps) { \\\n"
      + "  int t = blockIdx.x; \\\n"
      + "  if (t >= totalTokens) return; \\\n"
      + "  __shared__ double sdata[768]; \\\n"
      + "  long base = (long)t * dim; \\\n"
      + "  double localSS = 0.0; \\\n"
      + "  double localDotPre = 0.0; \\\n"
      + "  double localDotRouter = 0.0; \\\n"
      + "  for (int i = threadIdx.x; i < dim; i += 256) { \\\n"
      + "    double xi = (double)x[base + i]; \\\n"
      + "    localSS += xi * xi; \\\n"
      + "    localDotPre += ((double)dMoeInput[base + i] * preNormW[i]) * xi; \\\n"
      + "    localDotRouter += ((double)dRouterIn[base + i] * gateInpScale[i]) * xi; \\\n"
      + "  } \\\n"
      + "  sdata[threadIdx.x] = localSS; \\\n"
      + "  sdata[256 + threadIdx.x] = localDotPre; \\\n"
      + "  sdata[512 + threadIdx.x] = localDotRouter; \\\n"
      + "  __syncthreads(); \\\n"
      + "  for (int s = 128; s > 0; s >>= 1) { \\\n"
      + "    if (threadIdx.x < s) { \\\n"
      + "      sdata[threadIdx.x] += sdata[threadIdx.x + s]; \\\n"
      + "      sdata[256 + threadIdx.x] += sdata[256 + threadIdx.x + s]; \\\n"
      + "      sdata[512 + threadIdx.x] += sdata[512 + threadIdx.x + s]; \\\n"
      + "    } \\\n"
      + "    __syncthreads(); \\\n"
      + "  } \\\n"
      + "  double rX = 1.0 / sqrt(sdata[0] / (double)dim + eps); \\\n"
      + "  double routerRms = rX / sqrt((double)dim); \\\n"
      + "  double coeffPre = ((rX * rX) / (double)dim) * sdata[256]; \\\n"
      + "  double coeffRouter = ((rX * rX) / (double)dim) * sdata[512]; \\\n"
      + "  if (gPreNorm != 0) { \\\n"
      + "    for (int i = threadIdx.x; i < dim; i += 256) { \\\n"
      + "      atomicAddDouble(&gPreNorm[i], (double)dMoeInput[base + i] * rX * (double)x[base + i]); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "  if (gGateInpScale != 0) { \\\n"
      + "    for (int i = threadIdx.x; i < dim; i += 256) { \\\n"
      + "      atomicAddDouble(&gGateInpScale[i], (double)dRouterIn[base + i] * routerRms * (double)x[base + i]); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "  if (dx != 0) { \\\n"
      + "    for (int i = threadIdx.x; i < dim; i += 256) { \\\n"
      + "      double xi = (double)x[base + i]; \\\n"
      + "      double dxPre = rX * ((double)dMoeInput[base + i] * preNormW[i] - coeffPre * xi); \\\n"
      + "      double dxRouter = routerRms * ((double)dRouterIn[base + i] * gateInpScale[i] - coeffRouter * xi); \\\n"
      + "      dx[base + i] = (T)(dxPre + dxRouter); \\\n"
      + "    } \\\n"
      + "  } \\\n"
      + "}\n"
      + "\n"
      + "DEFINE_KERNELS(_f32, float)\n"
      + "DEFINE_KERNELS(_f64, double)\n";
}
