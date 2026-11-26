package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnOpTensorOp {
  public static final int CUDNN_OP_TENSOR_ADD = 0;
  public static final int CUDNN_OP_TENSOR_MUL = 1;
  public static final int CUDNN_OP_TENSOR_MIN = 2;
  public static final int CUDNN_OP_TENSOR_MAX = 3;
  public static final int CUDNN_OP_TENSOR_SQRT = 4;
  public static final int CUDNN_OP_TENSOR_NOT = 5;
  private cudnnOpTensorOp() {}
}
