package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnDataType {
  public static final int CUDNN_DATA_FLOAT = 0;
  public static final int CUDNN_DATA_DOUBLE = 1;
  public static final int CUDNN_DATA_HALF = 2;
  public static final int CUDNN_DATA_INT8 = 3;
  public static final int CUDNN_DATA_INT32 = 4;
  public static final int CUDNN_DATA_INT8x4 = 5;
  public static final int CUDNN_DATA_UINT8 = 6;
  public static final int CUDNN_DATA_UINT8x4 = 7;
  public static final int CUDNN_DATA_INT8x32 = 8;
  public static final int CUDNN_DATA_BFLOAT16 = 9;
  public static final int CUDNN_DATA_INT64 = 10;
  public static final int CUDNN_DATA_BOOLEAN = 11;
  private cudnnDataType() {}
}
