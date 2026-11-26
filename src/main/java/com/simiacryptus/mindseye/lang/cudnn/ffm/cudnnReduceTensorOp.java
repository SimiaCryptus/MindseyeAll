package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnReduceTensorOp {
  public static final int CUDNN_REDUCE_TENSOR_ADD = 0;
  public static final int CUDNN_REDUCE_TENSOR_MUL = 1;
  public static final int CUDNN_REDUCE_TENSOR_MIN = 2;
  public static final int CUDNN_REDUCE_TENSOR_MAX = 3;
  public static final int CUDNN_REDUCE_TENSOR_AMAX = 4;
  public static final int CUDNN_REDUCE_TENSOR_AVG = 5;
  public static final int CUDNN_REDUCE_TENSOR_NORM1 = 6;
  public static final int CUDNN_REDUCE_TENSOR_NORM2 = 7;
  public static final int CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8;
  private cudnnReduceTensorOp() {}
}
