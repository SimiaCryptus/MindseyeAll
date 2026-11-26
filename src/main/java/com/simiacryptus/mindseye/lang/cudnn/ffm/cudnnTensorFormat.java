package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnTensorFormat {
  public static final int CUDNN_TENSOR_NCHW = 0;
  public static final int CUDNN_TENSOR_NHWC = 1;
  public static final int CUDNN_TENSOR_NCHW_VECT_C = 2;
  private cudnnTensorFormat() {}
}
