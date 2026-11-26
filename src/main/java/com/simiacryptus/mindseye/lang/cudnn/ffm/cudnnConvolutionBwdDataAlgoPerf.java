package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

@RefIgnore
public class cudnnConvolutionBwdDataAlgoPerf {
  public static final StructLayout LAYOUT = cudnnConvolutionFwdAlgoPerf.LAYOUT;

  public int algo;
  public int status;
  public float time;
  public long memory;
  public int determinism;
  public int mathType;
  public int[] reserved = new int[3];

  public void readFrom(MemorySegment seg, long offset) {
    this.algo = seg.get(ValueLayout.JAVA_INT, offset);
    this.status = seg.get(ValueLayout.JAVA_INT, offset + 4);
    this.time = seg.get(ValueLayout.JAVA_FLOAT, offset + 8);
    this.memory = seg.get(ValueLayout.JAVA_LONG, offset + 16);
    this.determinism = seg.get(ValueLayout.JAVA_INT, offset + 24);
    this.mathType = seg.get(ValueLayout.JAVA_INT, offset + 28);
    for (int i = 0; i < 3; i++) {
      this.reserved[i] = seg.get(ValueLayout.JAVA_INT, offset + 32 + (long) i * 4);
    }
  }

  @Override
  public String toString() {
    return String.format("cudnnConvolutionBwdDataAlgoPerf[algo=%d, status=%d, time=%f, memory=%d]", algo, status, time, memory);
  }
}
