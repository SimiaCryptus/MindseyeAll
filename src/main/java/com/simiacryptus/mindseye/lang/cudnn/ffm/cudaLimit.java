package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

@RefIgnore
public final class cudaLimit {
  public static final int cudaLimitStackSize = 0;
  public static final int cudaLimitPrintfFifoSize = 1;
  public static final int cudaLimitMallocHeapSize = 2;
  public static final int cudaLimitDevRuntimeSyncDepth = 3;
  public static final int cudaLimitDevRuntimePendingLaunchCount = 4;
  public static final int cudaLimitMaxL2FetchGranularity = 5;
  public static final int cudaLimitPersistingL2CacheSize = 6;

  private cudaLimit() {
  }
}
