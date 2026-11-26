package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

import java.nio.charset.StandardCharsets;

@RefIgnore
public class cudaDeviceProp {
  public byte[] name = new byte[256];
  public int major;
  public int minor;
  public int multiProcessorCount;
  public int canMapHostMemory;

  public cudaDeviceProp() {
  }

  public String getName() {
    int len = 0;
    while (len < name.length && name[len] != 0) len++;
    return new String(name, 0, len, StandardCharsets.US_ASCII).trim();
  }

  @Override
  public String toString() {
    return String.format(
        "cudaDeviceProp{name='%s', computeCapability=%d.%d, multiProcessorCount=%d, canMapHostMemory=%d}",
        getName(), major, minor, multiProcessorCount, canMapHostMemory);
  }
}
