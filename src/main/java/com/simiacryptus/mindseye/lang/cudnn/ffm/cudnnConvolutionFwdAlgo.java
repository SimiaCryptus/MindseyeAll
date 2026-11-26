package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnConvolutionFwdAlgo {
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0;
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1;
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2;
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3;
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4;
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5;
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6;
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7;
  public static final int CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8;
  private cudnnConvolutionFwdAlgo() {}
}
