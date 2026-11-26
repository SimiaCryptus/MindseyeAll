package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnConvolutionBwdDataAlgo {
  public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0;
  public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1;
  public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2;
  public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3;
  public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4;
  public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5;
  public static final int CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6;
  private cudnnConvolutionBwdDataAlgo() {}
}
