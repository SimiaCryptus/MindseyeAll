package com.simiacryptus.mindseye.lang.cudnn.ffm;

import com.simiacryptus.ref.lang.RefIgnore;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Java 22 FFM-backed pointer representing either a native/device memory address or a Java host array.
 */
@RefIgnore
public class Pointer {
  private MemorySegment nativeSegment;
  private final Object hostArray;
  private final long byteOffset;

  public Pointer() {
    this.nativeSegment = MemorySegment.NULL;
    this.hostArray = null;
    this.byteOffset = 0L;
  }

  public Pointer(MemorySegment nativeSegment, long byteOffset) {
    this.nativeSegment = nativeSegment == null ? MemorySegment.NULL : nativeSegment;
    this.hostArray = null;
    this.byteOffset = byteOffset;
  }

  public Pointer(Pointer other) {
    this(other, 0L);
  }

  public Pointer(Pointer other, long byteOffset) {
    this.nativeSegment = other.nativeSegment;
    this.hostArray = other.hostArray;
    this.byteOffset = other.byteOffset + byteOffset;
  }

  private Pointer(Object hostArray, long byteOffset) {
    this.nativeSegment = MemorySegment.NULL;
    this.hostArray = hostArray;
    this.byteOffset = byteOffset;
  }

  public static Pointer to(float[] values) {
    return new Pointer(values, 0L);
  }

  public static Pointer to(double[] values) {
    return new Pointer(values, 0L);
  }

  public static Pointer to(int[] values) {
    return new Pointer(values, 0L);
  }

  public static Pointer to(long[] values) {
    return new Pointer(values, 0L);
  }

  public static Pointer to(byte[] values) {
    return new Pointer(values, 0L);
  }

  public long getByteOffset() {
    return byteOffset;
  }

  public Pointer withByteOffset(long byteOffset) {
    return new Pointer(this, byteOffset);
  }

  public boolean isHostArray() {
    return hostArray != null;
  }

  public void setNativeAddress(long address) {
    this.nativeSegment = MemorySegment.ofAddress(address);
  }

  public void setNativeSegment(MemorySegment segment) {
    this.nativeSegment = segment == null ? MemorySegment.NULL : segment;
  }

  /**
   * Returns the effective native MemorySegment (with byteOffset applied) for native/device pointers,
   * or allocates and populates a temporary off-heap segment in the given Arena for host arrays.
   */
  public MemorySegment toNativeSegment(Arena arena) {
    if (hostArray == null) {
      long base = nativeSegment.address();
      if (base == 0L && byteOffset == 0L) {
        return MemorySegment.NULL;
      }
      return MemorySegment.ofAddress(base + byteOffset);
    }
    if (hostArray instanceof float[] arr) {
      MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_FLOAT, arr);
      return byteOffset == 0L ? seg : seg.asSlice(byteOffset);
    } else if (hostArray instanceof double[] arr) {
      MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_DOUBLE, arr);
      return byteOffset == 0L ? seg : seg.asSlice(byteOffset);
    } else if (hostArray instanceof int[] arr) {
      MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_INT, arr);
      return byteOffset == 0L ? seg : seg.asSlice(byteOffset);
    } else if (hostArray instanceof long[] arr) {
      MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_LONG, arr);
      return byteOffset == 0L ? seg : seg.asSlice(byteOffset);
    } else if (hostArray instanceof byte[] arr) {
      MemorySegment seg = arena.allocateFrom(ValueLayout.JAVA_BYTE, arr);
      return byteOffset == 0L ? seg : seg.asSlice(byteOffset);
    }
    throw new IllegalStateException("Unsupported host array type: " + hostArray.getClass());
  }

  /**
   * Allocates a writable off-heap buffer for receiving native data into a host array (or returns the native address).
   */
  public MemorySegment toWritableNativeSegment(Arena arena, long byteCount) {
    if (hostArray == null) {
      long base = nativeSegment.address();
      if (base == 0L && byteOffset == 0L) {
        return MemorySegment.NULL;
      }
      return MemorySegment.ofAddress(base + byteOffset);
    }
    return arena.allocate(byteCount, 8);
  }

  /**
   * Copies bytes written by a native call back into the backing Java host array.
   */
  public void copyBackFrom(MemorySegment segment, long byteCount) {
    if (hostArray == null) {
      return;
    }
    if (hostArray instanceof float[] arr) {
      int count = (int) (byteCount / Float.BYTES);
      int offsetElems = (int) (byteOffset / Float.BYTES);
      MemorySegment.copy(segment, ValueLayout.JAVA_FLOAT, 0, arr, offsetElems, count);
    } else if (hostArray instanceof double[] arr) {
      int count = (int) (byteCount / Double.BYTES);
      int offsetElems = (int) (byteOffset / Double.BYTES);
      MemorySegment.copy(segment, ValueLayout.JAVA_DOUBLE, 0, arr, offsetElems, count);
    } else if (hostArray instanceof int[] arr) {
      int count = (int) (byteCount / Integer.BYTES);
      int offsetElems = (int) (byteOffset / Integer.BYTES);
      MemorySegment.copy(segment, ValueLayout.JAVA_INT, 0, arr, offsetElems, count);
    } else if (hostArray instanceof long[] arr) {
      int count = (int) (byteCount / Long.BYTES);
      int offsetElems = (int) (byteOffset / Long.BYTES);
      MemorySegment.copy(segment, ValueLayout.JAVA_LONG, 0, arr, offsetElems, count);
    } else if (hostArray instanceof byte[] arr) {
      int count = (int) byteCount;
      int offsetElems = (int) byteOffset;
      MemorySegment.copy(segment, ValueLayout.JAVA_BYTE, 0, arr, offsetElems, count);
    } else {
      throw new IllegalStateException("Unsupported host array type: " + hostArray.getClass());
    }
  }

  public static MemorySegment toNativeOrNull(Pointer ptr, Arena arena) {
    return ptr == null ? MemorySegment.NULL : ptr.toNativeSegment(arena);
  }

  @Override
  public String toString() {
    if (hostArray != null) {
      return "Pointer[host=" + hostArray.getClass().getComponentType().getSimpleName() + "[], offset=" + byteOffset + "]";
    }
    return String.format("Pointer[0x%x+%d]", nativeSegment.address(), byteOffset);
  }
}
