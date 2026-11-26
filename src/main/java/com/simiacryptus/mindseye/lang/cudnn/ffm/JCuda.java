package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.List;

import static java.lang.foreign.ValueLayout.*;

/**
 * Java 22 Foreign Function &amp; Memory (FFM) bindings for the NVIDIA CUDA Runtime API ({@code libcudart}).
 */
@RefIgnore
public final class JCuda {
  public static final int cudaSuccess = 0;
  public static final int cudaMemAttachGlobal = 0x01;
  public static final int cudaHostAllocDefault = 0x00;
  public static final int cudaHostAllocWriteCombined = 0x04;
  public static final int cudaDeviceScheduleBlockingSync = 0x04;

  // cudaDeviceAttr constants (stable across CUDA 11/12/13)
  private static final int cudaDevAttrMultiProcessorCount = 16;
  private static final int cudaDevAttrCanMapHostMemory = 19;
  private static final int cudaDevAttrComputeCapabilityMajor = 75;
  private static final int cudaDevAttrComputeCapabilityMinor = 76;

  private static volatile boolean exceptionsEnabled = false;

  private JCuda() {
  }

  public static void setExceptionsEnabled(boolean enabled) {
    exceptionsEnabled = enabled;
  }

  public static String getJCudaVersion() {
    return "FFM-Java22-CUDA12";
  }

  private static final class Native {
    static final SymbolLookup LIB = NativeLibraryLoader.loadLibrary("CUDA Runtime (libcudart)", List.of(
        "libcudart.so.12",
        "/usr/local/cuda-12.6/lib64/libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
        "cudart64_12.dll",
        "cudart"
    ));

    static final MethodHandle cudaGetDeviceCount = NativeLibraryLoader.downcall(LIB, "cudaGetDeviceCount",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudaSetDevice = NativeLibraryLoader.downcall(LIB, "cudaSetDevice",
        FunctionDescriptor.of(JAVA_INT, JAVA_INT));
    static final MethodHandle cudaSetDeviceFlags = NativeLibraryLoader.downcall(LIB, "cudaSetDeviceFlags",
        FunctionDescriptor.of(JAVA_INT, JAVA_INT));
    static final MethodHandle cudaDeviceReset = NativeLibraryLoader.downcall(LIB, "cudaDeviceReset",
        FunctionDescriptor.of(JAVA_INT));
    static final MethodHandle cudaDeviceSynchronize = NativeLibraryLoader.downcall(LIB, "cudaDeviceSynchronize",
        FunctionDescriptor.of(JAVA_INT));
    static final MethodHandle cudaDeviceGetLimit = NativeLibraryLoader.downcall(LIB, "cudaDeviceGetLimit",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
    static final MethodHandle cudaDeviceSetLimit = NativeLibraryLoader.downcall(LIB, "cudaDeviceSetLimit",
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_LONG));
    static final MethodHandle cudaRuntimeGetVersion = NativeLibraryLoader.downcall(LIB, "cudaRuntimeGetVersion",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudaDriverGetVersion = NativeLibraryLoader.downcall(LIB, "cudaDriverGetVersion",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudaMemGetInfo = NativeLibraryLoader.downcall(LIB, "cudaMemGetInfo",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle cudaGetDeviceProperties = NativeLibraryLoader.downcall(LIB, "cudaGetDeviceProperties",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
    static final MethodHandle cudaDeviceGetAttribute = NativeLibraryLoader.downcall(LIB, "cudaDeviceGetAttribute",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));
    static final MethodHandle cudaMalloc = NativeLibraryLoader.downcall(LIB, "cudaMalloc",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG));
    static final MethodHandle cudaMallocManaged = NativeLibraryLoader.downcall(LIB, "cudaMallocManaged",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT));
    static final MethodHandle cudaHostAlloc = NativeLibraryLoader.downcall(LIB, "cudaHostAlloc",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT));
    static final MethodHandle cudaFree = NativeLibraryLoader.downcall(LIB, "cudaFree",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudaFreeHost = NativeLibraryLoader.downcall(LIB, "cudaFreeHost",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudaMemcpy = NativeLibraryLoader.downcall(LIB, "cudaMemcpy",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT));
    static final MethodHandle cudaMemcpyAsync = NativeLibraryLoader.downcall(LIB, "cudaMemcpyAsync",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS));
    static final MethodHandle cudaMemset = NativeLibraryLoader.downcall(LIB, "cudaMemset",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG));
    static final MethodHandle cudaStreamCreate = NativeLibraryLoader.downcall(LIB, "cudaStreamCreate",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudaStreamDestroy = NativeLibraryLoader.downcall(LIB, "cudaStreamDestroy",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle cudaStreamSynchronize = NativeLibraryLoader.downcall(LIB, "cudaStreamSynchronize",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
  }

  private static int check(int status) {
    if (exceptionsEnabled && status != cudaSuccess) {
      throw new RuntimeException("CUDA error code: " + status);
    }
    return status;
  }

  public static int cudaGetDeviceCount(int[] count) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_INT);
      int status = (int) Native.cudaGetDeviceCount.invokeExact(out);
      count[0] = out.get(JAVA_INT, 0);
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaSetDevice(int device) {
    try {
      return check((int) Native.cudaSetDevice.invokeExact(device));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaSetDeviceFlags(int flags) {
    try {
      return check((int) Native.cudaSetDeviceFlags.invokeExact(flags));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaDeviceReset() {
    try {
      return check((int) Native.cudaDeviceReset.invokeExact());
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaDeviceSynchronize() {
    try {
      return check((int) Native.cudaDeviceSynchronize.invokeExact());
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaDeviceGetLimit(long[] pValue, int limit) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_LONG);
      int status = (int) Native.cudaDeviceGetLimit.invokeExact(out, limit);
      pValue[0] = out.get(JAVA_LONG, 0);
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaDeviceSetLimit(int limit, long value) {
    try {
      return check((int) Native.cudaDeviceSetLimit.invokeExact(limit, value));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaRuntimeGetVersion(int[] runtimeVersion) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_INT);
      int status = (int) Native.cudaRuntimeGetVersion.invokeExact(out);
      runtimeVersion[0] = out.get(JAVA_INT, 0);
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaDriverGetVersion(int[] driverVersion) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(JAVA_INT);
      int status = (int) Native.cudaDriverGetVersion.invokeExact(out);
      driverVersion[0] = out.get(JAVA_INT, 0);
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaMemGetInfo(long[] free, long[] total) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment freeOut = arena.allocate(JAVA_LONG);
      MemorySegment totalOut = arena.allocate(JAVA_LONG);
      int status = (int) Native.cudaMemGetInfo.invokeExact(freeOut, totalOut);
      free[0] = freeOut.get(JAVA_LONG, 0);
      total[0] = totalOut.get(JAVA_LONG, 0);
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaGetDeviceProperties(cudaDeviceProp prop, int device) {
    try (Arena arena = Arena.ofConfined()) {
      // Allocate generous buffer for struct cudaDeviceProp (char name[256] is always at offset 0).
      MemorySegment propBuf = arena.allocate(4096, 16);
      int status = (int) Native.cudaGetDeviceProperties.invokeExact(propBuf, device);
      if (status == cudaSuccess) {
        MemorySegment.copy(propBuf, JAVA_BYTE, 0, prop.name, 0, prop.name.length);
        MemorySegment attrVal = arena.allocate(JAVA_INT);
        if ((int) Native.cudaDeviceGetAttribute.invokeExact(attrVal, cudaDevAttrCanMapHostMemory, device) == cudaSuccess) {
          prop.canMapHostMemory = attrVal.get(JAVA_INT, 0);
        }
        if ((int) Native.cudaDeviceGetAttribute.invokeExact(attrVal, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess) {
          prop.major = attrVal.get(JAVA_INT, 0);
        }
        if ((int) Native.cudaDeviceGetAttribute.invokeExact(attrVal, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess) {
          prop.minor = attrVal.get(JAVA_INT, 0);
        }
        if ((int) Native.cudaDeviceGetAttribute.invokeExact(attrVal, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
          prop.multiProcessorCount = attrVal.get(JAVA_INT, 0);
        }
      }
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaMalloc(Pointer devPtr, long size) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outPtr = arena.allocate(ADDRESS);
      int status = (int) Native.cudaMalloc.invokeExact(outPtr, size);
      devPtr.setNativeSegment(outPtr.get(ADDRESS, 0));
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaMallocManaged(Pointer devPtr, long size, int flags) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outPtr = arena.allocate(ADDRESS);
      int status = (int) Native.cudaMallocManaged.invokeExact(outPtr, size, flags);
      devPtr.setNativeSegment(outPtr.get(ADDRESS, 0));
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaHostAlloc(Pointer pHost, long size, int flags) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment outPtr = arena.allocate(ADDRESS);
      int status = (int) Native.cudaHostAlloc.invokeExact(outPtr, size, flags);
      pHost.setNativeSegment(outPtr.get(ADDRESS, 0));
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaFree(Pointer devPtr) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = Pointer.toNativeOrNull(devPtr, arena);
      return check((int) Native.cudaFree.invokeExact(seg));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaFreeHost(Pointer ptr) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = Pointer.toNativeOrNull(ptr, arena);
      return check((int) Native.cudaFreeHost.invokeExact(seg));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaMemcpy(Pointer dst, Pointer src, long count, int kind) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment dstSeg = dst.toWritableNativeSegment(arena, count);
      MemorySegment srcSeg = src.toNativeSegment(arena);
      int status = (int) Native.cudaMemcpy.invokeExact(dstSeg, srcSeg, count, kind);
      if (status == cudaSuccess && dst.isHostArray()) {
        dst.copyBackFrom(dstSeg, count);
      }
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaMemcpyAsync(Pointer dst, Pointer src, long count, int kind, cudaStream_t stream) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment dstSeg = dst.toWritableNativeSegment(arena, count);
      MemorySegment srcSeg = src.toNativeSegment(arena);
      MemorySegment streamSeg = NativeHandle.handleOrNull(stream);
      int status = (int) Native.cudaMemcpyAsync.invokeExact(dstSeg, srcSeg, count, kind, streamSeg);
      if (status == cudaSuccess && dst.isHostArray()) {
        int syncStatus = (int) Native.cudaStreamSynchronize.invokeExact(streamSeg);
        if (syncStatus == cudaSuccess) {
          dst.copyBackFrom(dstSeg, count);
        } else {
          status = syncStatus;
        }
      }
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaMemset(Pointer devPtr, int value, long count) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = Pointer.toNativeOrNull(devPtr, arena);
      return check((int) Native.cudaMemset.invokeExact(seg, value, count));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaStreamCreate(cudaStream_t pStream) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment out = arena.allocate(ADDRESS);
      int status = (int) Native.cudaStreamCreate.invokeExact(out);
      pStream.setHandle(out.get(ADDRESS, 0));
      return check(status);
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaStreamDestroy(cudaStream_t stream) {
    try {
      return check((int) Native.cudaStreamDestroy.invokeExact(NativeHandle.handleOrNull(stream)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  public static int cudaStreamSynchronize(cudaStream_t stream) {
    try {
      return check((int) Native.cudaStreamSynchronize.invokeExact(NativeHandle.handleOrNull(stream)));
    } catch (Throwable t) {
      throw propagate(t);
    }
  }

  private static RuntimeException propagate(Throwable t) {
    if (t instanceof RuntimeException re) return re;
    if (t instanceof Error err) throw err;
    return new RuntimeException(t);
  }
}
