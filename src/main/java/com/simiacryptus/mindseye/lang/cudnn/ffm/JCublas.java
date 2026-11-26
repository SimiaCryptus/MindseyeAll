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

package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Java 22 Foreign Function &amp; Memory (FFM) bindings for NVIDIA cuBLAS ({@code libcublas}).
 */
@RefIgnore
public final class JCublas {

  public static final int CUBLAS_STATUS_SUCCESS = 0;
  public static final int CUBLAS_OP_N = 0;
  public static final int CUBLAS_OP_T = 1;

  private static final ConcurrentHashMap<Long, MemorySegment> HANDLES = new ConcurrentHashMap<Long, MemorySegment>();

  private JCublas() {
  }

  private static final class Native {
    static final SymbolLookup LIB = NativeLibraryLoader.loadLibrary("NVIDIA cuBLAS (libcublas)", Arrays.asList(
        "libcublas.so.12",
        "/usr/local/cuda-12.6/lib64/libcublas.so.12",
        "/usr/local/cuda/lib64/libcublas.so.12",
        "libcublas.so",
        "/usr/local/cuda/lib64/libcublas.so",
        "cublas64_12.dll",
        "cublas"
    ));

    static final MethodHandle cublasCreate_v2 = NativeLibraryLoader.downcall(LIB, "cublasCreate_v2",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cublasDestroy_v2 = NativeLibraryLoader.downcall(LIB, "cublasDestroy_v2",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cublasSgemm_v2 = NativeLibraryLoader.downcall(LIB, "cublasSgemm_v2",
        FunctionDescriptor.of(JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_INT,
            JAVA_INT, JAVA_INT, JAVA_INT,
            ADDRESS,
            ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT,
            ADDRESS,
            ADDRESS, JAVA_INT));
    static final MethodHandle cublasDgemm_v2 = NativeLibraryLoader.downcall(LIB, "cublasDgemm_v2",
        FunctionDescriptor.of(JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_INT,
            JAVA_INT, JAVA_INT, JAVA_INT,
            ADDRESS,
            ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT,
            ADDRESS,
            ADDRESS, JAVA_INT));
  }

  private static MemorySegment getHandle(final int deviceId) {
    long key = (((long) CudaSystem.gpuGeneration.get()) << 32) | (deviceId & 0xFFFFFFFFL);
    MemorySegment existing = HANDLES.get(key);
    if (existing != null) {
      return existing;
    }
    synchronized (HANDLES) {
      existing = HANDLES.get(key);
      if (existing != null) {
        return existing;
      }
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment out = arena.allocate(ADDRESS);
        int status = (int) Native.cublasCreate_v2.invokeExact(out);
        if (status != CUBLAS_STATUS_SUCCESS) {
          throw new RuntimeException("cublasCreate_v2 failed with status " + status + " on device " + deviceId);
        }
        MemorySegment handle = out.get(ADDRESS, 0);
        HANDLES.put(key, handle);
        return handle;
      } catch (Throwable t) {
        throw propagate(t);
      }
    }
  }

  public static void sgemm(final int deviceId,
                           final int transa, final int transb,
                           final int m, final int n, final int k,
                           final float alpha,
                           final Pointer a, final int lda,
                           final Pointer b, final int ldb,
                           final float beta,
                           final Pointer c, final int ldc) {
    if (m <= 0 || n <= 0 || k <= 0) {
      return;
    }
    MemorySegment handle = getHandle(deviceId);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment alphaSeg = arena.allocateFrom(JAVA_FLOAT, alpha);
      MemorySegment betaSeg = arena.allocateFrom(JAVA_FLOAT, beta);
      MemorySegment aSeg = Pointer.toNativeOrNull(a, arena);
      MemorySegment bSeg = Pointer.toNativeOrNull(b, arena);
      MemorySegment cSeg = Pointer.toNativeOrNull(c, arena);
      int status = (int) Native.cublasSgemm_v2.invokeExact(
          handle, transa, transb,
          m, n, k,
          alphaSeg,
          aSeg, lda,
          bSeg, ldb,
          betaSeg,
          cSeg, ldc);
      if (status != CUBLAS_STATUS_SUCCESS) {
        throw new RuntimeException("cublasSgemm_v2 failed with status " + status);
      }
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static void dgemm(final int deviceId,
                           final int transa, final int transb,
                           final int m, final int n, final int k,
                           final double alpha,
                           final Pointer a, final int lda,
                           final Pointer b, final int ldb,
                           final double beta,
                           final Pointer c, final int ldc) {
    if (m <= 0 || n <= 0 || k <= 0) {
      return;
    }
    MemorySegment handle = getHandle(deviceId);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment alphaSeg = arena.allocateFrom(JAVA_DOUBLE, alpha);
      MemorySegment betaSeg = arena.allocateFrom(JAVA_DOUBLE, beta);
      MemorySegment aSeg = Pointer.toNativeOrNull(a, arena);
      MemorySegment bSeg = Pointer.toNativeOrNull(b, arena);
      MemorySegment cSeg = Pointer.toNativeOrNull(c, arena);
      int status = (int) Native.cublasDgemm_v2.invokeExact(
          handle, transa, transb,
          m, n, k,
          alphaSeg,
          aSeg, lda,
          bSeg, ldb,
          betaSeg,
          cSeg, ldc);
      if (status != CUBLAS_STATUS_SUCCESS) {
        throw new RuntimeException("cublasDgemm_v2 failed with status " + status);
      }
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  private static RuntimeException propagate(final Throwable t) {
    if (t instanceof RuntimeException) {
      return (RuntimeException) t;
    }
    if (t instanceof Error) {
      throw (Error) t;
    }
    return new RuntimeException(t);
  }
}
