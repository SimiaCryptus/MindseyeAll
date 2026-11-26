package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnPoolingMode {
  public static final int CUDNN_POOLING_MAX = 0;
  public static final int CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1;
  public static final int CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2;
  public static final int CUDNN_POOLING_MAX_DETERMINISTIC = 3;
  private cudnnPoolingMode() {}
}
