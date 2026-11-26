package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudaMemcpyKind {
  public static final int cudaMemcpyHostToHost = 0;
  public static final int cudaMemcpyHostToDevice = 1;
  public static final int cudaMemcpyDeviceToHost = 2;
  public static final int cudaMemcpyDeviceToDevice = 3;
  public static final int cudaMemcpyDefault = 4;

  private cudaMemcpyKind() {
  }

  public static String stringFor(int kind) {
    return switch (kind) {
      case cudaMemcpyHostToHost -> "cudaMemcpyHostToHost";
      case cudaMemcpyHostToDevice -> "cudaMemcpyHostToDevice";
      case cudaMemcpyDeviceToHost -> "cudaMemcpyDeviceToHost";
      case cudaMemcpyDeviceToDevice -> "cudaMemcpyDeviceToDevice";
      case cudaMemcpyDefault -> "cudaMemcpyDefault";
      default -> "INVALID cudaMemcpyKind: " + kind;
    };
  }
}
