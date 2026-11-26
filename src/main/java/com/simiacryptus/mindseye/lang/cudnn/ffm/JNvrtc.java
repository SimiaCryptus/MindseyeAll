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
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Java 22 Foreign Function &amp; Memory (FFM) bindings for NVIDIA NVRTC ({@code libnvrtc})
 * to compile CUDA C++ kernels to PTX at runtime without external build toolchains.
 */
@RefIgnore
public final class JNvrtc {

  public static final int NVRTC_SUCCESS = 0;

  private JNvrtc() {
  }

  private static final class Native {
    static final SymbolLookup LIB = NativeLibraryLoader.loadLibrary("NVIDIA NVRTC (libnvrtc)", Arrays.asList(
        "libnvrtc.so.12",
        "/usr/local/cuda-12.6/lib64/libnvrtc.so.12",
        "/usr/local/cuda/lib64/libnvrtc.so.12",
        "libnvrtc.so",
        "/usr/local/cuda/lib64/libnvrtc.so",
        "nvrtc64_120_0.dll",
        "nvrtc"
    ));

    static final MethodHandle nvrtcCreateProgram = NativeLibraryLoader.downcall(LIB, "nvrtcCreateProgram",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle nvrtcDestroyProgram = NativeLibraryLoader.downcall(LIB, "nvrtcDestroyProgram",
        FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle nvrtcCompileProgram = NativeLibraryLoader.downcall(LIB, "nvrtcCompileProgram",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
    static final MethodHandle nvrtcGetProgramLogSize = NativeLibraryLoader.downcall(LIB, "nvrtcGetProgramLogSize",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle nvrtcGetProgramLog = NativeLibraryLoader.downcall(LIB, "nvrtcGetProgramLog",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle nvrtcGetPTXSize = NativeLibraryLoader.downcall(LIB, "nvrtcGetPTXSize",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle nvrtcGetPTX = NativeLibraryLoader.downcall(LIB, "nvrtcGetPTX",
        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  }

  /**
   * Compiles CUDA C++ source code into null-terminated PTX bytes for the specified virtual architecture
   * (e.g. {@code "compute_75"} or {@code "compute_70"}).
   */
  public static byte[] compileToPtx(final String cudaSource, final String programName, final String computeArch) {
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment progOut = arena.allocate(ADDRESS);
      MemorySegment srcSeg = arena.allocateFrom(cudaSource, StandardCharsets.UTF_8);
      MemorySegment nameSeg = arena.allocateFrom(programName, StandardCharsets.UTF_8);

      int createStatus = (int) Native.nvrtcCreateProgram.invokeExact(
          progOut, srcSeg, nameSeg, 0, MemorySegment.NULL, MemorySegment.NULL);
      if (createStatus != NVRTC_SUCCESS) {
        throw new RuntimeException("nvrtcCreateProgram failed with status " + createStatus);
      }

      MemorySegment prog = progOut.get(ADDRESS, 0);
      try {
        int compileStatus = tryCompile(arena, prog, "--gpu-architecture=" + computeArch);
        if (compileStatus != NVRTC_SUCCESS && !"compute_70".equals(computeArch)) {
          compileStatus = tryCompile(arena, prog, "--gpu-architecture=compute_70");
        }
        if (compileStatus != NVRTC_SUCCESS) {
          String log = getProgramLog(arena, prog);
          throw new RuntimeException("nvrtcCompileProgram failed (status=" + compileStatus + "):\n" + log);
        }

        MemorySegment ptxSizeOut = arena.allocate(JAVA_LONG);
        int sizeStatus = (int) Native.nvrtcGetPTXSize.invokeExact(prog, ptxSizeOut);
        if (sizeStatus != NVRTC_SUCCESS) {
          throw new RuntimeException("nvrtcGetPTXSize failed with status " + sizeStatus);
        }
        long ptxSize = ptxSizeOut.get(JAVA_LONG, 0);
        MemorySegment ptxBuf = arena.allocate(ptxSize, 1);
        int getStatus = (int) Native.nvrtcGetPTX.invokeExact(prog, ptxBuf);
        if (getStatus != NVRTC_SUCCESS) {
          throw new RuntimeException("nvrtcGetPTX failed with status " + getStatus);
        }
        byte[] ptx = new byte[(int) ptxSize];
        MemorySegment.copy(ptxBuf, JAVA_BYTE, 0, ptx, 0, (int) ptxSize);
        return ptx;
      } finally {
        MemorySegment destroySlot = arena.allocate(ADDRESS);
        destroySlot.set(ADDRESS, 0, prog);
        int ignored = (int) Native.nvrtcDestroyProgram.invokeExact(destroySlot);
      }
    } catch (Throwable t) {
      if (t instanceof RuntimeException) {
        throw (RuntimeException) t;
      }
      if (t instanceof Error) {
        throw (Error) t;
      }
      throw new RuntimeException(t);
    }
  }

  private static int tryCompile(final Arena arena, final MemorySegment prog, final String archOption) throws Throwable {
    MemorySegment opts = arena.allocate(ADDRESS, 1);
    opts.setAtIndex(ADDRESS, 0, arena.allocateFrom(archOption, StandardCharsets.UTF_8));
    return (int) Native.nvrtcCompileProgram.invokeExact(prog, 1, opts);
  }

  private static String getProgramLog(final Arena arena, final MemorySegment prog) throws Throwable {
    MemorySegment logSizeOut = arena.allocate(JAVA_LONG);
    int st = (int) Native.nvrtcGetProgramLogSize.invokeExact(prog, logSizeOut);
    if (st != NVRTC_SUCCESS) {
      return "<failed to retrieve NVRTC log: " + st + ">";
    }
    long logSize = logSizeOut.get(JAVA_LONG, 0);
    if (logSize <= 1) {
      return "";
    }
    MemorySegment logBuf = arena.allocate(logSize, 1);
    int st2 = (int) Native.nvrtcGetProgramLog.invokeExact(prog, logBuf);
    if (st2 != NVRTC_SUCCESS) {
      return "<failed to read NVRTC log: " + st2 + ">";
    }
    byte[] bytes = new byte[(int) (logSize - 1)];
    MemorySegment.copy(logBuf, JAVA_BYTE, 0, bytes, 0, bytes.length);
    return new String(bytes, StandardCharsets.UTF_8);
  }
}
