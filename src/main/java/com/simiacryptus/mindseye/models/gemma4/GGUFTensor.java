/*
 * Copyright (c) 2026 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.models.gemma4;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RefIgnore;

import javax.annotation.Nonnull;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * MemorySegment-backed and array-backed tensor representations supporting GGUF quantizations
 * (F32, F16, BF16, Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, Q6_K, Q8_0, MXFP4) and conversion to MindsEye {@link Tensor}.
 */
@RefIgnore
public abstract class GGUFTensor {

  private static final ValueLayout.OfShort SHORT_LE = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  @FunctionalInterface
  public interface AggregateFunction {
    float apply(float acc, float value);
  }

  @FunctionalInterface
  public interface MapFunction {
    float apply(float value);
  }

  @FunctionalInterface
  public interface MapWithIndexFunction {
    float apply(float value, int index);
  }

  public static short readShort(@Nonnull final MemorySegment segment, final long offset) {
    return segment.get(SHORT_LE, offset);
  }

  public static void writeShort(@Nonnull final MemorySegment segment, final long offset, final short value) {
    segment.set(SHORT_LE, offset, value);
  }

  public static float readFloat16(@Nonnull final MemorySegment segment, final long offset) {
    return Float.float16ToFloat(readShort(segment, offset));
  }

  public static byte readByte(@Nonnull final MemorySegment segment, final long offset) {
    return segment.get(ValueLayout.JAVA_BYTE, offset);
  }

  public static int readInt32LE(@Nonnull final MemorySegment segment, final long offset) {
    return segment.get(INT_LE, offset);
  }

  public static float readFloat32LE(@Nonnull final MemorySegment segment, final long offset) {
    return segment.get(FLOAT_LE, offset);
  }

  public abstract long size();

  public abstract float getFloat(long index);

  public void setFloat(final int index, final float value) {
    throw new UnsupportedOperationException("Read-only GGUFTensor: " + type());
  }

  @Nonnull
  public abstract GGMLType type();

  public static int numberOfElements(@Nonnull final int... dimensions) {
    int result = 1;
    for (int d : dimensions) {
      if (d <= 0) {
        throw new IllegalArgumentException("Invalid dimension: " + Arrays.toString(dimensions));
      }
      result = Math.multiplyExact(result, d);
    }
    return result;
  }

  public static long numberOfElementsLong(@Nonnull final int... dimensions) {
    long result = 1L;
    for (int d : dimensions) {
      if (d <= 0) {
        throw new IllegalArgumentException("Invalid dimension: " + Arrays.toString(dimensions));
      }
      result = Math.multiplyExact(result, d);
    }
    return result;
  }

  public static float scalarDot(@Nonnull final GGUFTensor left, final int leftOffset,
                                @Nonnull final GGUFTensor right, final int rightOffset, final int size) {
    float result = 0f;
    for (int j = 0; j < size; j++) {
      result += left.getFloat((long) leftOffset + j) * right.getFloat((long) rightOffset + j);
    }
    return result;
  }

  public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
    return scalarDot(this, thisOffset, that, thatOffset, size);
  }

  public void matmul(@Nonnull final GGUFTensor that, @Nonnull final GGUFTensor out, final int dim0, final int dim1) {
    IntStream.range(0, dim0).parallel().forEach(i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
  }

  public void matmul(@Nonnull final GGUFTensor that, @Nonnull final GGUFTensor out,
                     final int dim0, final int dim1, final int thisOffset) {
    IntStream.range(0, dim0).parallel().forEach(i -> out.setFloat(i, dot(thisOffset + i * dim1, that, 0, dim1)));
  }

  public float reduce(final int thisOffset, final int size, final float seed, @Nonnull final AggregateFunction reduce) {
    float result = seed;
    for (int i = 0; i < size; ++i) {
      result = reduce.apply(result, getFloat((long) thisOffset + i));
    }
    return result;
  }

  public float sum(final int thisOffset, final int size) {
    return reduce(thisOffset, size, 0f, Float::sum);
  }

  public float max(final int thisOffset, final int size) {
    return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
  }

  public void copyTo(final long thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
    for (int i = 0; i < size; i++) {
      that.setFloat(thatOffset + i, getFloat(thisOffset + i));
    }
  }

  public int argmax(final int thisOffset, final int size) {
    if (size <= 0) {
      throw new IllegalArgumentException("size must be positive");
    }
    int maxIndex = thisOffset;
    float maxValue = getFloat(maxIndex);
    int endIndex = thisOffset + size;
    for (int i = thisOffset + 1; i < endIndex; ++i) {
      float f = getFloat(i);
      if (f > maxValue) {
        maxValue = f;
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  public int argmax() {
    return argmax(0, Math.toIntExact(size()));
  }

  @Nonnull
  public GGUFTensor mapInPlace(final int thisOffset, final int size, @Nonnull final MapFunction mapFunction) {
    int endIndex = thisOffset + size;
    for (int i = thisOffset; i < endIndex; ++i) {
      setFloat(i, mapFunction.apply(getFloat(i)));
    }
    return this;
  }

  @Nonnull
  public GGUFTensor mapInPlace(@Nonnull final MapFunction mapFunction) {
    return mapInPlace(0, Math.toIntExact(size()), mapFunction);
  }

  @Nonnull
  public GGUFTensor mapWithIndexInPlace(final int thisOffset, final int size,
                                        @Nonnull final MapWithIndexFunction mapWithIndexFunction) {
    int endOffset = thisOffset + size;
    for (int i = thisOffset; i < endOffset; ++i) {
      setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
    }
    return this;
  }

  @Nonnull
  public GGUFTensor addInPlace(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
    for (int i = 0; i < size; i++) {
      setFloat(thisOffset + i, getFloat((long) thisOffset + i) + that.getFloat((long) thatOffset + i));
    }
    return this;
  }

  @Nonnull
  public GGUFTensor addInPlace(@Nonnull final GGUFTensor that) {
    return addInPlace(0, that, 0, Math.toIntExact(size()));
  }

  @Nonnull
  public GGUFTensor multiplyInPlace(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
    for (int i = 0; i < size; i++) {
      setFloat(thisOffset + i, getFloat((long) thisOffset + i) * that.getFloat((long) thatOffset + i));
    }
    return this;
  }

  @Nonnull
  public GGUFTensor divideInPlace(final int thisOffset, final int size, final float value) {
    return mapInPlace(thisOffset, size, f -> f / value);
  }

  @Nonnull
  public GGUFTensor fillInPlace(final int thisOffset, final int size, final float value) {
    return mapInPlace(thisOffset, size, unused -> value);
  }

  @Nonnull
  public GGUFTensor softmaxInPlace(final int thisOffset, final int size) {
    float maxVal = max(thisOffset, size);
    mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
    float sumVal = sum(thisOffset, size);
    return divideInPlace(thisOffset, size, sumVal);
  }

  @Nonnull
  public GGUFTensor saxpyInPlace(final int thisOffset, @Nonnull final GGUFTensor that,
                                 final int thatOffset, final int size, final float a) {
    for (int i = 0; i < size; ++i) {
      setFloat(thisOffset + i, a * that.getFloat((long) thatOffset + i) + getFloat((long) thisOffset + i));
    }
    return this;
  }

  /**
   * Dequantizes all elements into a new float array.
   */
  @Nonnull
  public float[] toFloatArray() {
    int n = Math.toIntExact(size());
    float[] out = new float[n];
    if (n >= 4096) {
      IntStream.range(0, n).parallel().forEach(i -> out[i] = getFloat(i));
    } else {
      for (int i = 0; i < n; i++) {
        out[i] = getFloat(i);
      }
    }
    return out;
  }

  /**
   * Dequantizes a slice [offset, offset + length) into a new double array.
   */
  @Nonnull
  public double[] toDoubleArray(final long offset, final int length) {
    double[] out = new double[length];
    if (length >= 4096) {
      IntStream.range(0, length).parallel().forEach(i -> out[i] = getFloat(offset + i));
    } else {
      for (int i = 0; i < length; i++) {
        out[i] = getFloat(offset + i);
      }
    }
    return out;
  }

  /**
   * Dequantizes all elements into a new double array.
   */
  @Nonnull
  public double[] toDoubleArray() {
    return toDoubleArray(0L, Math.toIntExact(size()));
  }

  /**
   * Converts this tensor into a reference-counted MindsEye {@link Tensor} with the given dimensions.
   */
  @Nonnull
  public Tensor toMindsEyeTensor(@Nonnull final int... dims) {
    int expected = Tensor.length(dims);
    if ((long) expected != size()) {
      throw new IllegalArgumentException("Dimension length " + expected + " != GGUFTensor size " + size());
    }
    return new Tensor(toDoubleArray(), dims);
  }

  /**
   * Wraps a double array into an {@link ArrayFloatTensor}.
   */
  @Nonnull
  public static ArrayFloatTensor fromDoubleArray(@Nonnull final double[] values) {
    float[] floats = new float[values.length];
    for (int i = 0; i < values.length; i++) {
      floats[i] = (float) values[i];
    }
    return new ArrayFloatTensor(floats);
  }

  /**
   * Loads a {@link GGUFTensor} from a {@link GGUF.GGMLTensorEntry}.
   */
  @Nonnull
  public static GGUFTensor fromEntry(@Nonnull final GGUF.GGMLTensorEntry entry) {
    GGMLType ggmlType = entry.ggmlType();
    long numElements = numberOfElementsLong(entry.shape());
    MemorySegment seg = entry.memorySegment();
    switch (ggmlType) {
      case F32:
        return new F32Tensor(numElements, seg);
      case F16:
        return new F16Tensor(numElements, seg);
      case BF16:
        return new BF16Tensor(numElements, seg);
      case Q4_0:
        return new Q4_0Tensor(numElements, seg);
      case Q4_1:
        return new Q4_1Tensor(numElements, seg);
      case Q5_1:
        return new Q5_1Tensor(numElements, seg);
      case Q4_K:
        return new Q4_KTensor(numElements, seg);
      case Q5_K:
        return new Q5_KTensor(numElements, seg);
      case Q6_K:
        return new Q6_KTensor(numElements, seg);
      case Q8_0:
        return new Q8_0Tensor(numElements, seg);
      case MXFP4:
        return new MXFP4Tensor(numElements, seg);
      default:
        throw new UnsupportedOperationException("Unsupported quantization format: " + ggmlType);
    }
  }

  /**
   * Mutable float[]-backed tensor used for activations and unquantized weights.
   */
  @RefIgnore
  public static final class ArrayFloatTensor extends GGUFTensor {
    private final float[] values;

    public ArrayFloatTensor(@Nonnull final float[] values) {
      this.values = values;
    }

    public ArrayFloatTensor(@Nonnull final FloatBuffer buf) {
      this.values = new float[buf.remaining()];
      int pos = buf.position();
      buf.get(this.values);
      buf.position(pos);
    }

    @Nonnull
    public static ArrayFloatTensor allocate(@Nonnull final int... dims) {
      return new ArrayFloatTensor(new float[numberOfElements(dims)]);
    }

    @Nonnull
    public float[] values() {
      return values;
    }

    @Override
    public long size() {
      return values.length;
    }

    @Override
    public float getFloat(final long index) {
      return values[Math.toIntExact(index)];
    }

    @Override
    public void setFloat(final int index, final float value) {
      values[index] = value;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.F32;
    }

    @Nonnull
    @Override
    public GGUFTensor fillInPlace(final int thisOffset, final int size, final float value) {
      Arrays.fill(values, thisOffset, thisOffset + size, value);
      return this;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (that instanceof ArrayFloatTensor) {
        float[] other = ((ArrayFloatTensor) that).values;
        float sum = 0f;
        for (int i = 0; i < size; i++) {
          sum += this.values[thisOffset + i] * other[thatOffset + i];
        }
        return sum;
      }
      return that.dot(thatOffset, this, thisOffset, size);
    }
  }

  /**
   * Read-only F32 MemorySegment tensor.
   */
  @RefIgnore
  public static final class F32Tensor extends GGUFTensor {
    private final long size;
    private final MemorySegment segment;

    public F32Tensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Override
    public float getFloat(final long index) {
      return readFloat32LE(segment, index * Float.BYTES);
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.F32;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (that instanceof ArrayFloatTensor) {
        float[] other = ((ArrayFloatTensor) that).values;
        float sum = 0f;
        long bytePos = (long) thisOffset * Float.BYTES;
        for (int i = 0; i < size; i++, bytePos += Float.BYTES) {
          sum += readFloat32LE(segment, bytePos) * other[thatOffset + i];
        }
        return sum;
      }
      return scalarDot(this, thisOffset, that, thatOffset, size);
    }
  }

  /**
   * F16 MemorySegment tensor (also supports allocation for compact KV caches).
   */
  @RefIgnore
  public static final class F16Tensor extends GGUFTensor {
    private final long size;
    private final MemorySegment segment;

    public F16Tensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Nonnull
    public static F16Tensor allocate(@Nonnull final int... dims) {
      int n = numberOfElements(dims);
      return new F16Tensor(n, Arena.ofAuto().allocate((long) n * GGMLType.FLOAT16_BYTES));
    }

    @Override
    public long size() {
      return size;
    }

    @Override
    public float getFloat(final long index) {
      return readFloat16(segment, index * 2L);
    }

    @Override
    public void setFloat(final int index, final float value) {
      writeShort(segment, (long) index * 2L, Float.floatToFloat16(value));
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.F16;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (that instanceof ArrayFloatTensor) {
        float[] other = ((ArrayFloatTensor) that).values;
        float sum = 0f;
        long bytePos = (long) thisOffset * 2L;
        for (int i = 0; i < size; i++, bytePos += 2L) {
          sum += readFloat16(segment, bytePos) * other[thatOffset + i];
        }
        return sum;
      }
      return scalarDot(this, thisOffset, that, thatOffset, size);
    }
  }

  /**
   * BF16 MemorySegment tensor.
   */
  @RefIgnore
  public static final class BF16Tensor extends GGUFTensor {
    private final long size;
    private final MemorySegment segment;

    public BF16Tensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Override
    public float getFloat(final long index) {
      short bits = readShort(segment, index * 2L);
      return Float.intBitsToFloat(((int) bits) << 16);
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.BF16;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (that instanceof ArrayFloatTensor) {
        float[] other = ((ArrayFloatTensor) that).values;
        float sum = 0f;
        long bytePos = (long) thisOffset * 2L;
        for (int i = 0; i < size; i++, bytePos += 2L) {
          float w = Float.intBitsToFloat(((int) readShort(segment, bytePos)) << 16);
          sum += w * other[thatOffset + i];
        }
        return sum;
      }
      return scalarDot(this, thisOffset, that, thatOffset, size);
    }
  }

  /**
   * Q4_0 quantized tensor: 32 elements per 18-byte block (fp16 scale + 16 packed nibble bytes, signed offset -8).
   */
  @RefIgnore
  public static final class Q4_0Tensor extends GGUFTensor {
    private final long size;
    private final MemorySegment segment;

    public Q4_0Tensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.Q4_0;
    }

    @Override
    public float getFloat(final long index) {
      long blockIndex = index / 32L;
      long blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
      float scale = readFloat16(segment, blockOffset);
      int modIndex = (int) (index & 31L);
      int quant;
      if (modIndex < 16) {
        quant = (readByte(segment, blockOffset + 2L + modIndex) & 0x0F) - 8;
      } else {
        quant = ((Byte.toUnsignedInt(readByte(segment, blockOffset + 2L + modIndex - 16)) >>> 4) & 0x0F) - 8;
      }
      return quant * scale;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (!(that instanceof ArrayFloatTensor)) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
      }
      float[] x = ((ArrayFloatTensor) that).values;
      float result = 0f;
      int j = 0;
      int alignmentBound = Math.min(size, -thisOffset & 31);
      if (alignmentBound > 0) {
        result += scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
        j += alignmentBound;
      }
      long blockOffset = ((long) (thisOffset + j) / 32L) * GGMLType.Q4_0.getTypeSize();
      int upperBound = j + ((size - j) & ~31);
      for (; j < upperBound; j += 32, blockOffset += GGMLType.Q4_0.getTypeSize()) {
        float scale = readFloat16(segment, blockOffset);
        float blockSum = 0f;
        int base = thatOffset + j;
        for (int p = 0; p < 16; p++) {
          int packed = Byte.toUnsignedInt(readByte(segment, blockOffset + 2L + p));
          int lo = (packed & 0x0F) - 8;
          int hi = ((packed >>> 4) & 0x0F) - 8;
          blockSum += lo * x[base + p] + hi * x[base + 16 + p];
        }
        result += blockSum * scale;
      }
      if (j < size) {
        result += scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
      }
      return result;
    }
  }

  /**
   * Q4_1 quantized tensor: 32 elements per 20-byte block (fp16 delta + fp16 min + 16 packed nibble bytes).
   */
  @RefIgnore
  public static final class Q4_1Tensor extends GGUFTensor {
    private final long size;
    private final MemorySegment segment;

    public Q4_1Tensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.Q4_1;
    }

    @Override
    public float getFloat(final long index) {
      long blockIndex = index / 32L;
      long blockOffset = blockIndex * GGMLType.Q4_1.getTypeSize();
      float delta = readFloat16(segment, blockOffset);
      float min = readFloat16(segment, blockOffset + 2L);
      int modIndex = (int) (index & 31L);
      int quant;
      if (modIndex < 16) {
        quant = Byte.toUnsignedInt(readByte(segment, blockOffset + 4L + modIndex)) & 0x0F;
      } else {
        quant = (Byte.toUnsignedInt(readByte(segment, blockOffset + 4L + modIndex - 16)) >>> 4) & 0x0F;
      }
      return delta * quant + min;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (!(that instanceof ArrayFloatTensor)) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
      }
      float[] x = ((ArrayFloatTensor) that).values;
      float result = 0f;
      int j = 0;
      int alignmentBound = Math.min(size, -thisOffset & 31);
      if (alignmentBound > 0) {
        result += scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
        j += alignmentBound;
      }
      long blockOffset = ((long) (thisOffset + j) / 32L) * GGMLType.Q4_1.getTypeSize();
      int upperBound = j + ((size - j) & ~31);
      for (; j < upperBound; j += 32, blockOffset += GGMLType.Q4_1.getTypeSize()) {
        float delta = readFloat16(segment, blockOffset);
        float min = readFloat16(segment, blockOffset + 2L);
        float qSum = 0f;
        float xSum = 0f;
        int base = thatOffset + j;
        for (int p = 0; p < 16; p++) {
          int packed = Byte.toUnsignedInt(readByte(segment, blockOffset + 4L + p));
          int lo = packed & 0x0F;
          int hi = (packed >>> 4) & 0x0F;
          float xLo = x[base + p];
          float xHi = x[base + 16 + p];
          qSum += lo * xLo + hi * xHi;
          xSum += xLo + xHi;
        }
        result += delta * qSum + min * xSum;
      }
      if (j < size) {
        result += scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
      }
      return result;
    }
  }

  /**
   * Q5_1 quantized tensor: 32 elements per 24-byte block (fp16 d + fp16 m + uint32 qh + 16 packed nibble bytes).
   */
  @RefIgnore
  public static final class Q5_1Tensor extends GGUFTensor {
    private final long size;
    private final MemorySegment segment;

    public Q5_1Tensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.Q5_1;
    }

    @Override
    public float getFloat(final long index) {
      long blockIndex = index / 32L;
      int inBlockIndex = (int) (index & 31L);
      long blockOffset = blockIndex * GGMLType.Q5_1.getTypeSize();
      float d = readFloat16(segment, blockOffset);
      float m = readFloat16(segment, blockOffset + 2L);
      int qh = readInt32LE(segment, blockOffset + 4L);

      int j;
      int nibble;
      int xh;
      if (inBlockIndex < 16) {
        j = inBlockIndex;
        nibble = Byte.toUnsignedInt(readByte(segment, blockOffset + 8L + j)) & 0x0F;
        xh = ((qh >> j) << 4) & 0x10;
      } else {
        j = inBlockIndex - 16;
        nibble = (Byte.toUnsignedInt(readByte(segment, blockOffset + 8L + j)) >>> 4) & 0x0F;
        xh = (qh >> (j + 12)) & 0x10;
      }
      return (nibble | xh) * d + m;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (!(that instanceof ArrayFloatTensor)) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
      }
      float[] x = ((ArrayFloatTensor) that).values;
      float result = 0f;
      int j = 0;
      int alignmentBound = Math.min(size, -thisOffset & 31);
      if (alignmentBound > 0) {
        result += scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
        j += alignmentBound;
      }
      long blockOffset = ((long) (thisOffset + j) / 32L) * GGMLType.Q5_1.getTypeSize();
      int upperBound = j + ((size - j) & ~31);
      for (; j < upperBound; j += 32, blockOffset += GGMLType.Q5_1.getTypeSize()) {
        float d = readFloat16(segment, blockOffset);
        float m = readFloat16(segment, blockOffset + 2L);
        int qh = readInt32LE(segment, blockOffset + 4L);
        long qsBase = blockOffset + 8L;
        int base = thatOffset + j;
        float qSum = 0f;
        float xSum = 0f;
        for (int p = 0; p < 16; p++) {
          int packed = Byte.toUnsignedInt(readByte(segment, qsBase + p));
          int x0 = (packed & 0x0F) | (((qh >> p) << 4) & 0x10);
          int x1 = ((packed >>> 4) & 0x0F) | ((qh >> (p + 12)) & 0x10);
          float v0 = x[base + p];
          float v1 = x[base + 16 + p];
          qSum += x0 * v0 + x1 * v1;
          xSum += v0 + v1;
        }
        result += d * qSum + m * xSum;
      }
      if (j < size) {
        result += scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
      }
      return result;
    }
  }

  static int getScaleMinK4(final int j, @Nonnull final MemorySegment mem, final long scalesOffset, final boolean isMin) {
    if (j < 4) {
      int idx = isMin ? j + 4 : j;
      return Byte.toUnsignedInt(readByte(mem, scalesOffset + idx)) & 63;
    } else {
      int lowIdx = j + 4;
      int highIdx = isMin ? j : j - 4;
      int low = isMin
          ? (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) >> 4)
          : (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) & 0xF);
      int high = (Byte.toUnsignedInt(readByte(mem, scalesOffset + highIdx)) >> 6) & 0x3;
      return low | (high << 4);
    }
  }

  /**
   * Q4_K k-quant tensor: 256 elements per 144-byte super-block.
   */
  @RefIgnore
  public static final class Q4_KTensor extends GGUFTensor {
    private static final int BLOCK_SIZE = GGMLType.QK_K;
    private static final int TYPE_SIZE = GGMLType.Q4_K.getTypeSize();
    private final long size;
    private final MemorySegment segment;

    public Q4_KTensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.Q4_K;
    }

    @Override
    public float getFloat(final long index) {
      long blockIndex = index / BLOCK_SIZE;
      int withinBlock = (int) (index % BLOCK_SIZE);
      long blockOffset = blockIndex * TYPE_SIZE;
      float d = readFloat16(segment, blockOffset);
      float dmin = readFloat16(segment, blockOffset + 2L);
      long scalesOffset = blockOffset + 4L;
      long qsOffset = blockOffset + 16L;

      int group = withinBlock / 64;
      int inGroup = withinBlock % 64;
      int subBlock;
      int nibbleIndex;
      boolean isHigh;
      if (inGroup < 32) {
        subBlock = group * 2;
        nibbleIndex = inGroup;
        isHigh = false;
      } else {
        subBlock = group * 2 + 1;
        nibbleIndex = inGroup - 32;
        isHigh = true;
      }

      int sc = getScaleMinK4(subBlock, segment, scalesOffset, false);
      int m = getScaleMinK4(subBlock, segment, scalesOffset, true);
      byte qsByte = readByte(segment, qsOffset + group * 32L + nibbleIndex);
      int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);
      return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (!(that instanceof ArrayFloatTensor)) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
      }
      float[] x = ((ArrayFloatTensor) that).values;
      float result = 0f;
      int j = 0;
      int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
      if (alignmentBound > 0) {
        result += scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
        j += alignmentBound;
      }
      long blockOffset = ((long) (thisOffset + j) / BLOCK_SIZE) * TYPE_SIZE;
      int upperBound = j + ((size - j) / BLOCK_SIZE) * BLOCK_SIZE;
      for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
        float d = readFloat16(segment, blockOffset);
        float dmin = readFloat16(segment, blockOffset + 2L);
        long scalesOff = blockOffset + 4L;
        long qsOff = blockOffset + 16L;
        for (int g = 0; g < 4; g++) {
          float d1 = d * getScaleMinK4(g * 2, segment, scalesOff, false);
          float m1 = dmin * getScaleMinK4(g * 2, segment, scalesOff, true);
          float d2 = d * getScaleMinK4(g * 2 + 1, segment, scalesOff, false);
          float m2 = dmin * getScaleMinK4(g * 2 + 1, segment, scalesOff, true);
          int loBase = thatOffset + j + g * 64;
          int hiBase = loBase + 32;
          long groupQs = qsOff + g * 32L;
          for (int l = 0; l < 32; l++) {
            int packed = Byte.toUnsignedInt(readByte(segment, groupQs + l));
            int lo = packed & 0xF;
            int hi = (packed >>> 4) & 0xF;
            result += (d1 * lo - m1) * x[loBase + l] + (d2 * hi - m2) * x[hiBase + l];
          }
        }
      }
      if (j < size) {
        result += scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
      }
      return result;
    }
  }

  /**
   * Q5_K k-quant tensor: 256 elements per 176-byte super-block.
   */
  @RefIgnore
  public static final class Q5_KTensor extends GGUFTensor {
    private static final int BLOCK_SIZE = GGMLType.QK_K;
    private static final int TYPE_SIZE = GGMLType.Q5_K.getTypeSize();
    private final long size;
    private final MemorySegment segment;

    public Q5_KTensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.Q5_K;
    }

    @Override
    public float getFloat(final long index) {
      long blockIndex = index / BLOCK_SIZE;
      int withinBlock = (int) (index % BLOCK_SIZE);
      long blockOffset = blockIndex * TYPE_SIZE;
      float d = readFloat16(segment, blockOffset);
      float dmin = readFloat16(segment, blockOffset + 2L);
      long scalesOffset = blockOffset + 4L;
      long qhOffset = blockOffset + 16L;
      long qsOffset = blockOffset + 48L;

      int group = withinBlock / 64;
      int inGroup = withinBlock % 64;
      boolean isHigh = inGroup >= 32;
      int l = isHigh ? inGroup - 32 : inGroup;
      int subBlock = isHigh ? group * 2 + 1 : group * 2;

      int sc = getScaleMinK4(subBlock, segment, scalesOffset, false);
      int m = getScaleMinK4(subBlock, segment, scalesOffset, true);

      byte qsByte = readByte(segment, qsOffset + group * 32L + l);
      int nibble = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);
      int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
      int qhBit = (Byte.toUnsignedInt(readByte(segment, qhOffset + l)) >> qhBitPos) & 1;
      int quant = nibble | (qhBit << 4);
      return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (!(that instanceof ArrayFloatTensor)) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
      }
      float[] x = ((ArrayFloatTensor) that).values;
      float result = 0f;
      int j = 0;
      int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
      if (alignmentBound > 0) {
        result += scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
        j += alignmentBound;
      }
      long blockOffset = ((long) (thisOffset + j) / BLOCK_SIZE) * TYPE_SIZE;
      int upperBound = j + ((size - j) / BLOCK_SIZE) * BLOCK_SIZE;
      for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
        float d = readFloat16(segment, blockOffset);
        float dmin = readFloat16(segment, blockOffset + 2L);
        long scalesOff = blockOffset + 4L;
        long qhOff = blockOffset + 16L;
        long qsOff = blockOffset + 48L;
        for (int g = 0; g < 4; g++) {
          float d1 = d * getScaleMinK4(g * 2, segment, scalesOff, false);
          float m1 = dmin * getScaleMinK4(g * 2, segment, scalesOff, true);
          float d2 = d * getScaleMinK4(g * 2 + 1, segment, scalesOff, false);
          float m2 = dmin * getScaleMinK4(g * 2 + 1, segment, scalesOff, true);
          int qhBitPosLo = 2 * g;
          int qhBitPosHi = qhBitPosLo + 1;
          int loBase = thatOffset + j + g * 64;
          int hiBase = loBase + 32;
          long groupQs = qsOff + g * 32L;
          for (int l = 0; l < 32; l++) {
            int packed = Byte.toUnsignedInt(readByte(segment, groupQs + l));
            int qhByte = Byte.toUnsignedInt(readByte(segment, qhOff + l));
            int lo = (packed & 0xF) | (((qhByte >> qhBitPosLo) & 1) << 4);
            int hi = ((packed >>> 4) & 0xF) | (((qhByte >> qhBitPosHi) & 1) << 4);
            result += (d1 * lo - m1) * x[loBase + l] + (d2 * hi - m2) * x[hiBase + l];
          }
        }
      }
      if (j < size) {
        result += scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
      }
      return result;
    }
  }

  /**
   * Q6_K k-quant tensor: 256 elements per 210-byte super-block.
   */
  @RefIgnore
  public static final class Q6_KTensor extends GGUFTensor {
    private static final int BLOCK_SIZE = GGMLType.QK_K;
    private static final int TYPE_SIZE = GGMLType.Q6_K.getTypeSize();
    private final long size;
    private final MemorySegment segment;

    public Q6_KTensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.Q6_K;
    }

    @Override
    public float getFloat(final long index) {
      long blockIndex = index / BLOCK_SIZE;
      int withinBlock = (int) (index % BLOCK_SIZE);
      long blockOffset = blockIndex * TYPE_SIZE;
      long qlOff = blockOffset;
      long qhOff = blockOffset + 128L;
      long scOff = blockOffset + 192L;
      float d = readFloat16(segment, blockOffset + 208L);

      int half = withinBlock / 128;
      int rem128 = withinBlock % 128;
      int sub32 = rem128 / 32;
      int l = rem128 % 32;

      long qlBase = qlOff + half * 64L;
      long qhBase = qhOff + half * 32L;

      int qlNibble;
      int qhShift;
      switch (sub32) {
        case 0:
          qlNibble = Byte.toUnsignedInt(readByte(segment, qlBase + l)) & 0xF;
          qhShift = 0;
          break;
        case 1:
          qlNibble = Byte.toUnsignedInt(readByte(segment, qlBase + 32L + l)) & 0xF;
          qhShift = 2;
          break;
        case 2:
          qlNibble = (Byte.toUnsignedInt(readByte(segment, qlBase + l)) >> 4) & 0xF;
          qhShift = 4;
          break;
        case 3:
          qlNibble = (Byte.toUnsignedInt(readByte(segment, qlBase + 32L + l)) >> 4) & 0xF;
          qhShift = 6;
          break;
        default:
          throw new IllegalStateException();
      }
      int qhBits = (Byte.toUnsignedInt(readByte(segment, qhBase + l)) >> qhShift) & 3;
      int q6 = (qlNibble | (qhBits << 4)) - 32;
      int sc = readByte(segment, scOff + half * 8L + sub32 * 2L + (l / 16));
      return d * sc * q6;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (!(that instanceof ArrayFloatTensor)) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
      }
      float[] x = ((ArrayFloatTensor) that).values;
      float result = 0f;
      int j = 0;
      int alignmentBound = Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
      if (alignmentBound > 0) {
        result += scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
        j += alignmentBound;
      }
      long blockOffset = ((long) (thisOffset + j) / BLOCK_SIZE) * TYPE_SIZE;
      int upperBound = j + ((size - j) / BLOCK_SIZE) * BLOCK_SIZE;
      for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
        long qlOff = blockOffset;
        long qhOff = blockOffset + 128L;
        long scOff = blockOffset + 192L;
        float d = readFloat16(segment, blockOffset + 208L);
        for (int h = 0; h < 2; h++) {
          long qlBase = qlOff + h * 64L;
          long qhBase = qhOff + h * 32L;
          int base = thatOffset + j + h * 128;
          for (int c = 0; c < 2; c++) {
            float ds0 = d * readByte(segment, scOff + h * 8L + c);
            float ds1 = d * readByte(segment, scOff + h * 8L + 2L + c);
            float ds2 = d * readByte(segment, scOff + h * 8L + 4L + c);
            float ds3 = d * readByte(segment, scOff + h * 8L + 6L + c);
            int off16 = c * 16;
            for (int p = 0; p < 16; p++) {
              int idx = off16 + p;
              int qlA = Byte.toUnsignedInt(readByte(segment, qlBase + idx));
              int qlB = Byte.toUnsignedInt(readByte(segment, qlBase + 32L + idx));
              int qhV = Byte.toUnsignedInt(readByte(segment, qhBase + idx));
              int q0 = ((qlA & 0xF) | ((qhV & 3) << 4)) - 32;
              int q1 = ((qlB & 0xF) | (((qhV >>> 2) & 3) << 4)) - 32;
              int q2 = (((qlA >>> 4) & 0xF) | (((qhV >>> 4) & 3) << 4)) - 32;
              int q3 = (((qlB >>> 4) & 0xF) | (((qhV >>> 6) & 3) << 4)) - 32;
              result += ds0 * q0 * x[base + idx]
                  + ds1 * q1 * x[base + 32 + idx]
                  + ds2 * q2 * x[base + 64 + idx]
                  + ds3 * q3 * x[base + 96 + idx];
            }
          }
        }
      }
      if (j < size) {
        result += scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
      }
      return result;
    }
  }

  /**
   * Q8_0 quantized tensor: 32 elements per 34-byte block (fp16 scale + 32 signed int8 quants).
   */
  @RefIgnore
  public static final class Q8_0Tensor extends GGUFTensor {
    private final long size;
    private final MemorySegment segment;

    public Q8_0Tensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(final long index) {
      long blockIndex = index / 32L;
      long withinBlockIndex = index & 31L;
      long blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
      byte quant = readByte(segment, blockOffset + 2L + withinBlockIndex);
      float scale = readFloat16(segment, blockOffset);
      return quant * scale;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (!(that instanceof ArrayFloatTensor)) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
      }
      float[] x = ((ArrayFloatTensor) that).values;
      float result = 0f;
      int j = 0;
      int alignmentBound = Math.min(size, -thisOffset & 31);
      if (alignmentBound > 0) {
        result += scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
        j += alignmentBound;
      }
      long blockOffset = ((long) (thisOffset + j) / 32L) * GGMLType.Q8_0.getTypeSize();
      int upperBound = j + ((size - j) & ~31);
      for (; j < upperBound; j += 32, blockOffset += GGMLType.Q8_0.getTypeSize()) {
        float scale = readFloat16(segment, blockOffset);
        float blockSum = 0f;
        int base = thatOffset + j;
        for (int p = 0; p < 32; p++) {
          blockSum += readByte(segment, blockOffset + 2L + p) * x[base + p];
        }
        result += blockSum * scale;
      }
      if (j < size) {
        result += scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
      }
      return result;
    }
  }

  /**
   * MXFP4 quantized tensor: 32 elements per 17-byte block (E8M0 shared exponent + 16 packed E2M1 FP4 bytes).
   */
  @RefIgnore
  public static final class MXFP4Tensor extends GGUFTensor {
    private static final int[] MXFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    private final long size;
    private final MemorySegment segment;

    public MXFP4Tensor(final long size, @Nonnull final MemorySegment segment) {
      this.size = size;
      this.segment = segment;
    }

    @Override
    public long size() {
      return size;
    }

    @Nonnull
    @Override
    public GGMLType type() {
      return GGMLType.MXFP4;
    }

    public static float e8m0ToFp32Half(final int x) {
      int bits;
      if (x < 2) {
        bits = 0x00200000 << x;
      } else {
        bits = (x - 1) << 23;
      }
      return Float.intBitsToFloat(bits);
    }

    @Override
    public float getFloat(final long index) {
      long blockIndex = index / 32L;
      int inBlockIndex = (int) (index & 31L);
      long blockOffset = blockIndex * GGMLType.MXFP4.getTypeSize();
      int e8m0 = Byte.toUnsignedInt(readByte(segment, blockOffset));
      float d = e8m0ToFp32Half(e8m0);
      int packed = Byte.toUnsignedInt(readByte(segment, blockOffset + 1L + (inBlockIndex & 0x0F)));
      int nibble = inBlockIndex < 16 ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
      return MXFP4_VALUES[nibble] * d;
    }

    @Override
    public float dot(final int thisOffset, @Nonnull final GGUFTensor that, final int thatOffset, final int size) {
      if (!(that instanceof ArrayFloatTensor)) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
      }
      float[] x = ((ArrayFloatTensor) that).values;
      float result = 0f;
      int j = 0;
      int alignmentBound = Math.min(size, -thisOffset & 31);
      if (alignmentBound > 0) {
        result += scalarDot(this, thisOffset, that, thatOffset, alignmentBound);
        j += alignmentBound;
      }
      long blockOffset = ((long) (thisOffset + j) / 32L) * GGMLType.MXFP4.getTypeSize();
      int upperBound = j + ((size - j) & ~31);
      for (; j < upperBound; j += 32, blockOffset += GGMLType.MXFP4.getTypeSize()) {
        float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(segment, blockOffset)));
        float blockSum = 0f;
        int base = thatOffset + j;
        for (int p = 0; p < 16; p++) {
          int packed = Byte.toUnsignedInt(readByte(segment, blockOffset + 1L + p));
          blockSum += MXFP4_VALUES[packed & 0x0F] * x[base + p]
              + MXFP4_VALUES[(packed >>> 4) & 0x0F] * x[base + 16 + p];
        }
        result += blockSum * d;
      }
      if (j < size) {
        result += scalarDot(this, thisOffset + j, that, thatOffset + j, size - j);
      }
      return result;
    }
  }
}
