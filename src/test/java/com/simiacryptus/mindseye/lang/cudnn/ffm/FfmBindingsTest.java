package com.simiacryptus.mindseye.lang.cudnn.ffm;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.*;

public class FfmBindingsTest {

  @Test
  @Tag("UnitTest")
  public void testPointerHostArrayRoundTrip() {
    float[] src = new float[]{1.5f, -2.25f, 3.75f, 4.0f};
    float[] dst = new float[4];
    Pointer srcPtr = Pointer.to(src);
    Pointer dstPtr = Pointer.to(dst);

    try (Arena arena = Arena.ofConfined()) {
      MemorySegment srcSeg = srcPtr.toNativeSegment(arena);
      MemorySegment dstSeg = dstPtr.toWritableNativeSegment(arena, (long) src.length * Float.BYTES);
      MemorySegment.copy(srcSeg, 0, dstSeg, 0, (long) src.length * Float.BYTES);
      dstPtr.copyBackFrom(dstSeg, (long) dst.length * Float.BYTES);
    }

    assertArrayEquals(src, dst, 0.0f);
  }

  @Test
  @Tag("UnitTest")
  public void testPointerByteOffset() {
    Pointer base = new Pointer();
    base.setNativeAddress(0x1000L);
    Pointer offset = base.withByteOffset(64L);
    assertEquals(64L, offset.getByteOffset());
    try (Arena arena = Arena.ofConfined()) {
      assertEquals(0x1040L, offset.toNativeSegment(arena).address());
    }
  }

  @Test
  @Tag("UnitTest")
  public void testAlgoPerfStructLayout() {
    assertEquals(48L, cudnnConvolutionFwdAlgoPerf.LAYOUT.byteSize());
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment seg = arena.allocate(cudnnConvolutionFwdAlgoPerf.LAYOUT);
      seg.set(ValueLayout.JAVA_INT, 0, cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT);
      seg.set(ValueLayout.JAVA_INT, 4, cudnnStatus.CUDNN_STATUS_SUCCESS);
      seg.set(ValueLayout.JAVA_FLOAT, 8, 1.25f);
      seg.set(ValueLayout.JAVA_LONG, 16, 4096L);
      seg.set(ValueLayout.JAVA_INT, 24, 1);
      seg.set(ValueLayout.JAVA_INT, 28, 2);

      cudnnConvolutionFwdAlgoPerf perf = new cudnnConvolutionFwdAlgoPerf();
      perf.readFrom(seg, 0);
      assertEquals(cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT, perf.algo);
      assertEquals(cudnnStatus.CUDNN_STATUS_SUCCESS, perf.status);
      assertEquals(1.25f, perf.time, 0.0f);
      assertEquals(4096L, perf.memory);
      assertEquals(1, perf.determinism);
      assertEquals(2, perf.mathType);
    }
  }
}
