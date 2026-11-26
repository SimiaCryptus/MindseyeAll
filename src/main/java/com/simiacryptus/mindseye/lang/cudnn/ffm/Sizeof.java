package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.ValueLayout;

@RefIgnore
public final class Sizeof {
  public static final int BYTE = Byte.BYTES;
  public static final int SHORT = Short.BYTES;
  public static final int INT = Integer.BYTES;
  public static final int LONG = Long.BYTES;
  public static final int FLOAT = Float.BYTES;
  public static final int DOUBLE = Double.BYTES;
  public static final int POINTER = (int) ValueLayout.ADDRESS.byteSize();

  private Sizeof() {
  }
}
