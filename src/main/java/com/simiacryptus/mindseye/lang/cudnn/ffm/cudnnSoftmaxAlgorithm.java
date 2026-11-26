package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnSoftmaxAlgorithm {
  public static final int CUDNN_SOFTMAX_FAST = 0;
  public static final int CUDNN_SOFTMAX_ACCURATE = 1;
  public static final int CUDNN_SOFTMAX_LOG = 2;
  private cudnnSoftmaxAlgorithm() {}
}
