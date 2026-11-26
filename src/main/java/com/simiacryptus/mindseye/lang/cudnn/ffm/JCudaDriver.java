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

import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Java 22 Foreign Function &amp; Memory (FFM) bindings for the NVIDIA CUDA Driver API ({@code libcuda})
 * supporting PTX module loading and kernel launches on the active CUDA context.
 */
@RefIgnore
public final class JCudaDriver {

  public static final int CUDA_SUCCESS = 0;
  private static volatile boolean initialized = false;

  private JCudaDriver() {
  }

  private static final class Native {
    static final SymbolLookup LIB = NativeLibraryLoader.loadLibrary("NVIDIA CUDA Driver (libcuda)", Arrays.asList(
        "libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "libcuda.so",
        "nvcuda.dll",
        "cuda"
    ));

    static final MethodHandle cuInit = NativeLibraryLoader.downcall(LIB, "cuInit",
        FunctionDescriptor.of(JAVA_INT, JAVA_INT));
    static final MethodHandle cuDeviceGet = NativeLibraryLoader.downcall(LIB, "cuDeviceGet",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
    static final MethodHandle cuCtxGetCurrent = NativeLibraryLoader.downcall(LIB, "cuCtxGetCurrent",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cuDevicePrimaryCtxRetain = NativeLibraryLoader.downcall(LIB, "cuDevicePrimaryCtxRetain",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
    static final MethodHandle cuCtxSetCurrent = NativeLibraryLoader.downcall(LIB, "cuCtxSetCurrent",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cuModuleLoadData = NativeLibraryLoader.downcall(LIB, "cuModuleLoadData",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle cuModuleGetFunction = NativeLibraryLoader.downcall(LIB, "cuModuleGetFunction",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle cuLaunchKernel = NativeLibraryLoader.downcall(LIB, "cuLaunchKernel",
        FunctionDescriptor.of(JAVA_INT,
            ADDRESS,
            JAVA_INT, JAVA_INT, JAVA_INT,
            JAVA_INT, JAVA_INT, JAVA_INT,
            JAVA_INT,
            ADDRESS,
            ADDRESS,
            ADDRESS));
  }

  public static void ensureContext(final int deviceId) {
    try (Arena arena = Arena.ofConfined()) {
      if (!initialized) {
        synchronized (JCudaDriver.class) {
          if (!initialized) {
            int initSt = (int) Native.cuInit.invokeExact(0);
            if (initSt != CUDA_SUCCESS) {
              throw new RuntimeException("cuInit(0) failed with status " + initSt);
            }
            initialized = true;
          }
        }
      }
      MemorySegment ctxOut = arena.allocate(ADDRESS);
      int getSt = (int) Native.cuCtxGetCurrent.invokeExact(ctxOut);
      if (getSt != CUDA_SUCCESS) {
        throw new RuntimeException("cuCtxGetCurrent failed with status " + getSt);
      }
      MemorySegment ctx = ctxOut.get(ADDRESS, 0);
      if (ctx.address() == 0L) {
        MemorySegment devOut = arena.allocate(JAVA_INT);
        int devSt = (int) Native.cuDeviceGet.invokeExact(devOut, deviceId);
        if (devSt != CUDA_SUCCESS) {
          throw new RuntimeException("cuDeviceGet failed with status " + devSt);
        }
        int cuDev = devOut.get(JAVA_INT, 0);
        int retSt = (int) Native.cuDevicePrimaryCtxRetain.invokeExact(ctxOut, cuDev);
        if (retSt != CUDA_SUCCESS) {
          throw new RuntimeException("cuDevicePrimaryCtxRetain failed with status " + retSt);
        }
        MemorySegment primaryCtx = ctxOut.get(ADDRESS, 0);
        int setSt = (int) Native.cuCtxSetCurrent.invokeExact(primaryCtx);
        if (setSt != CUDA_SUCCESS) {
          throw new RuntimeException("cuCtxSetCurrent failed with status " + setSt);
        }
      }
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static MemorySegment loadModulePtx(final int deviceId, final byte[] ptxNullTerminated) {
    ensureContext(deviceId);
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment modOut = arena.allocate(ADDRESS);
      MemorySegment ptxSeg = arena.allocateFrom(JAVA_BYTE, ptxNullTerminated);
      int st = (int) Native.cuModuleLoadData.invokeExact(modOut, ptxSeg);
      if (st != CUDA_SUCCESS) {
        throw new RuntimeException("cuModuleLoadData failed with status " + st);
      }
      return modOut.get(ADDRESS, 0);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static MemorySegment getFunction(final MemorySegment module, final String functionName) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment fnOut = arena.allocate(ADDRESS);
      MemorySegment nameSeg = arena.allocateFrom(functionName, StandardCharsets.UTF_8);
      int st = (int) Native.cuModuleGetFunction.invokeExact(fnOut, module, nameSeg);
      if (st != CUDA_SUCCESS) {
        throw new RuntimeException("cuModuleGetFunction(" + functionName + ") failed with status " + st);
      }
      return fnOut.get(ADDRESS, 0);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static void launchKernel(final MemorySegment function,
                                  final int gridX, final int gridY, final int gridZ,
                                  final int blockX, final int blockY, final int blockZ,
                                  final int sharedMemBytes,
                                  final Object... args) {
    if (gridX <= 0 || gridY <= 0 || gridZ <= 0) {
      return;
    }
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment params = arena.allocate(ADDRESS, args.length);
      for (int i = 0; i < args.length; i++) {
        Object arg = args[i];
        MemorySegment slot;
        if (arg == null) {
          slot = arena.allocate(ADDRESS);
          slot.set(ADDRESS, 0, MemorySegment.NULL);
        } else if (arg instanceof Pointer) {
          slot = arena.allocate(ADDRESS);
          slot.set(ADDRESS, 0, ((Pointer) arg).toNativeSegment(arena));
        } else if (arg instanceof MemorySegment) {
          slot = arena.allocate(ADDRESS);
          slot.set(ADDRESS, 0, (MemorySegment) arg);
        } else if (arg instanceof Integer) {
          slot = arena.allocate(JAVA_INT);
          slot.set(JAVA_INT, 0, ((Integer) arg).intValue());
        } else if (arg instanceof Long) {
          slot = arena.allocate(JAVA_LONG);
          slot.set(JAVA_LONG, 0, ((Long) arg).longValue());
        } else if (arg instanceof Float) {
          slot = arena.allocate(JAVA_FLOAT);
          slot.set(JAVA_FLOAT, 0, ((Float) arg).floatValue());
        } else if (arg instanceof Double) {
          slot = arena.allocate(JAVA_DOUBLE);
          slot.set(JAVA_DOUBLE, 0, ((Double) arg).doubleValue());
        } else {
          throw new IllegalArgumentException("Unsupported CUDA kernel arg type at index " + i + ": " + arg.getClass());
        }
        params.setAtIndex(ADDRESS, i, slot);
      }
      int st = (int) Native.cuLaunchKernel.invokeExact(
          function,
          gridX, gridY, gridZ,
          blockX, blockY, blockZ,
          sharedMemBytes,
          MemorySegment.NULL,
          params,
          MemorySegment.NULL);
      if (st != CUDA_SUCCESS) {
        throw new RuntimeException("cuLaunchKernel failed with status " + st);
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
