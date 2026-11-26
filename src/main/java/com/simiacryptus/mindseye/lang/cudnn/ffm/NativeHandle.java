package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.MemorySegment;

/**
 * Base class for opaque native CUDA / cuDNN handles backed by a Java 22 FFM MemorySegment.
 */
@RefIgnore
public abstract class NativeHandle {
  private MemorySegment handle = MemorySegment.NULL;

  public MemorySegment handle() {
    return handle;
  }

  public void setHandle(MemorySegment handle) {
    this.handle = handle == null ? MemorySegment.NULL : handle;
  }

  public static MemorySegment handleOrNull(NativeHandle h) {
    return h == null ? MemorySegment.NULL : h.handle();
  }

  @Override
  public String toString() {
    return String.format("%s[0x%x]", getClass().getSimpleName(), handle.address());
  }
}
