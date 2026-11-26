package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnActivationMode {
  public static final int CUDNN_ACTIVATION_SIGMOID = 0;
  public static final int CUDNN_ACTIVATION_RELU = 1;
  public static final int CUDNN_ACTIVATION_TANH = 2;
  public static final int CUDNN_ACTIVATION_CLIPPED_RELU = 3;
  public static final int CUDNN_ACTIVATION_ELU = 4;
  public static final int CUDNN_ACTIVATION_IDENTITY = 5;
  public static final int CUDNN_ACTIVATION_SWISH = 6;
  private cudnnActivationMode() {}
}
