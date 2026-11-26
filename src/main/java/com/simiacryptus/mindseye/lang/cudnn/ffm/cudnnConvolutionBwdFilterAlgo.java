package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnConvolutionBwdFilterAlgo {
  public static final int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0;
  public static final int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1;
  public static final int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2;
  public static final int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3;
  public static final int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4;
  public static final int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5;
  public static final int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6;
  public static final int CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7;
  private cudnnConvolutionBwdFilterAlgo() {}
}
