package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudnnStatus {
  public static final int CUDNN_STATUS_SUCCESS = 0;
  public static final int CUDNN_STATUS_NOT_INITIALIZED = 1001;
  public static final int CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH = 1002;
  public static final int CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH = 1003;
  public static final int CUDNN_STATUS_DEPRECATED = 1004;
  public static final int CUDNN_STATUS_LICENSE_ERROR = 1005;
  public static final int CUDNN_STATUS_RUNTIME_IN_PROGRESS = 1006;
  public static final int CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 1007;
  public static final int CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED = 1008;
  public static final int CUDNN_STATUS_BAD_PARAM = 2000;
  public static final int CUDNN_STATUS_INVALID_VALUE = 2001;
  public static final int CUDNN_STATUS_BAD_PARAM_NULL_POINTER = 2002;
  public static final int CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER = 2003;
  public static final int CUDNN_STATUS_NOT_SUPPORTED = 3000;
  public static final int CUDNN_STATUS_ARCH_MISMATCH = 3007;
  public static final int CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 3008;
  public static final int CUDNN_STATUS_INTERNAL_ERROR = 4000;
  public static final int CUDNN_STATUS_ALLOC_FAILED = 4003;
  public static final int CUDNN_STATUS_MAPPING_ERROR = 4006;
  public static final int CUDNN_STATUS_EXECUTION_FAILED = 5000;

  private cudnnStatus() {
  }

  public static String stringFor(int status) {
    return switch (status) {
      case CUDNN_STATUS_SUCCESS -> "CUDNN_STATUS_SUCCESS";
      case CUDNN_STATUS_NOT_INITIALIZED -> "CUDNN_STATUS_NOT_INITIALIZED";
      case CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH -> "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH";
      case CUDNN_STATUS_BAD_PARAM -> "CUDNN_STATUS_BAD_PARAM";
      case CUDNN_STATUS_INVALID_VALUE -> "CUDNN_STATUS_INVALID_VALUE";
      case CUDNN_STATUS_NOT_SUPPORTED -> "CUDNN_STATUS_NOT_SUPPORTED";
      case CUDNN_STATUS_ARCH_MISMATCH -> "CUDNN_STATUS_ARCH_MISMATCH";
      case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING -> "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
      case CUDNN_STATUS_INTERNAL_ERROR -> "CUDNN_STATUS_INTERNAL_ERROR";
      case CUDNN_STATUS_ALLOC_FAILED -> "CUDNN_STATUS_ALLOC_FAILED";
      case CUDNN_STATUS_MAPPING_ERROR -> "CUDNN_STATUS_MAPPING_ERROR";
      case CUDNN_STATUS_EXECUTION_FAILED -> "CUDNN_STATUS_EXECUTION_FAILED";
      default -> "CUDNN_STATUS_" + status;
    };
  }
}
