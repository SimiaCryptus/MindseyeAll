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

import com.simiacryptus.ref.lang.RefIgnore;

import javax.annotation.Nonnull;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parser and writer for GGUF v2/v3 model files and memory buffers.
 */
@RefIgnore
public final class GGUF {

  public static final int GGUF_MAGIC = 0x46554747;
  public static final int DEFAULT_ALIGNMENT = 32;
  private static final int PARSE_BUFFER_SIZE = 1 << 20;
  private static final List<Integer> SUPPORTED_GGUF_VERSIONS = Arrays.asList(2, 3);

  private int magic;
  private int version;
  private int tensorCount;
  private int alignment;
  private int metadataKvCount;
  private Map<String, Object> metadata;
  private Map<String, GGUFTensorInfo> tensorInfos;
  private long tensorDataOffset;

  public enum MetadataValueType {
    UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64;

    private static final MetadataValueType[] VALUES = values();

    public static MetadataValueType fromIndex(final int index) {
      if (index < 0 || index >= VALUES.length) {
        throw new IllegalArgumentException("Unsupported GGUF metadata type index: " + index);
      }
      return VALUES[index];
    }
  }

  @RefIgnore
  public static final class GGUFTensorInfo {
    private final String name;
    private final int[] dimensions;
    private final GGMLType ggmlType;
    private final long offset;

    public GGUFTensorInfo(@Nonnull final String name, @Nonnull final int[] dimensions,
                          @Nonnull final GGMLType ggmlType, final long offset) {
      this.name = name;
      this.dimensions = dimensions.clone();
      this.ggmlType = ggmlType;
      this.offset = offset;
    }

    @Nonnull
    public String name() {
      return name;
    }

    @Nonnull
    public int[] dimensions() {
      return dimensions.clone();
    }

    @Nonnull
    public GGMLType ggmlType() {
      return ggmlType;
    }

    public long offset() {
      return offset;
    }
  }

  @RefIgnore
  public static final class GGMLTensorEntry {
    private final MemorySegment mappedFile;
    private final String name;
    private final GGMLType ggmlType;
    private final int[] shape;
    private final MemorySegment memorySegment;

    public GGMLTensorEntry(@Nonnull final MemorySegment mappedFile, @Nonnull final String name,
                           @Nonnull final GGMLType ggmlType, @Nonnull final int[] shape,
                           @Nonnull final MemorySegment memorySegment) {
      this.mappedFile = mappedFile;
      this.name = name;
      this.ggmlType = ggmlType;
      this.shape = shape.clone();
      this.memorySegment = memorySegment;
    }

    @Nonnull
    public MemorySegment mappedFile() {
      return mappedFile;
    }

    @Nonnull
    public String name() {
      return name;
    }

    @Nonnull
    public GGMLType ggmlType() {
      return ggmlType;
    }

    @Nonnull
    public int[] shape() {
      return shape.clone();
    }

    @Nonnull
    public MemorySegment memorySegment() {
      return memorySegment;
    }

    @Nonnull
    public FloatBuffer toFloatBuffer() {
      if (ggmlType == GGMLType.F32) {
        return memorySegment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
      }
      return FloatBuffer.wrap(GGUFTensor.fromEntry(this).toFloatArray());
    }
  }

  @RefIgnore
  private static final class ChannelReader {
    private final ReadableByteChannel channel;
    private final ByteBuffer buffer;
    private long position;

    private ChannelReader(@Nonnull final ReadableByteChannel channel, final int bufferSize) {
      this.channel = channel;
      this.buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.LITTLE_ENDIAN);
      this.buffer.limit(0);
      this.position = 0L;
    }

    long position() {
      return position;
    }

    private void ensure(final int required) throws IOException {
      if (required > buffer.capacity()) {
        throw new IllegalArgumentException("Requested read " + required + " exceeds capacity " + buffer.capacity());
      }
      if (buffer.remaining() >= required) {
        return;
      }
      buffer.compact();
      while (buffer.position() < required) {
        int read = channel.read(buffer);
        if (read < 0) {
          throw new IOException("Unexpected EOF while reading GGUF metadata");
        }
      }
      buffer.flip();
    }

    byte readByte() throws IOException {
      ensure(Byte.BYTES);
      position += Byte.BYTES;
      return buffer.get();
    }

    short readShort() throws IOException {
      ensure(Short.BYTES);
      position += Short.BYTES;
      return buffer.getShort();
    }

    int readInt() throws IOException {
      ensure(Integer.BYTES);
      position += Integer.BYTES;
      return buffer.getInt();
    }

    long readLong() throws IOException {
      ensure(Long.BYTES);
      position += Long.BYTES;
      return buffer.getLong();
    }

    float readFloat() throws IOException {
      return Float.intBitsToFloat(readInt());
    }

    double readDouble() throws IOException {
      return Double.longBitsToDouble(readLong());
    }

    byte[] readBytes(final int length) throws IOException {
      byte[] bytes = new byte[length];
      int copied = 0;
      while (copied < length) {
        if (!buffer.hasRemaining()) {
          ensure(1);
        }
        int chunk = Math.min(length - copied, buffer.remaining());
        buffer.get(bytes, copied, chunk);
        copied += chunk;
        position += chunk;
      }
      return bytes;
    }

    void skipBytes(final int length) throws IOException {
      int remaining = length;
      while (remaining > 0) {
        if (!buffer.hasRemaining()) {
          ensure(1);
        }
        int chunk = Math.min(remaining, buffer.remaining());
        buffer.position(buffer.position() + chunk);
        remaining -= chunk;
        position += chunk;
      }
    }
  }

  public int getVersion() {
    return version;
  }

  public int getTensorCount() {
    return tensorCount;
  }

  @Nonnull
  public Map<String, Object> getMetadata() {
    return metadata;
  }

  @Nonnull
  public Map<String, GGUFTensorInfo> getTensorInfos() {
    return tensorInfos;
  }

  public long getTensorDataOffset() {
    return tensorDataOffset;
  }

  public int getAlignment() {
    if (alignment != 0) {
      return alignment;
    }
    Object raw = metadata.get("general.alignment");
    alignment = raw instanceof Number ? ((Number) raw).intValue() : DEFAULT_ALIGNMENT;
    if (Integer.bitCount(alignment) != 1) {
      throw new IllegalArgumentException("GGUF alignment must be a power of 2, got " + alignment);
    }
    return alignment;
  }

  @Nonnull
  public static GGUF loadModel(@Nonnull final Path modelPath) throws IOException {
    try (FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
      return loadModel(fileChannel);
    }
  }

  @Nonnull
  public static GGUF loadModel(@Nonnull final FileChannel fileChannel) throws IOException {
    fileChannel.position(0L);
    return loadModel((ReadableByteChannel) fileChannel);
  }

  @Nonnull
  public static GGUF loadModel(@Nonnull final ReadableByteChannel channel) throws IOException {
    GGUF gguf = new GGUF();
    ChannelReader reader = new ChannelReader(channel, PARSE_BUFFER_SIZE);
    gguf.loadModelImpl(reader);
    return gguf;
  }

  @Nonnull
  public static Map<String, GGMLTensorEntry> loadTensors(@Nonnull final FileChannel fileChannel,
                                                         final long tensorDataOffset,
                                                         @Nonnull final Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
    Arena arena = Arena.ofAuto();
    MemorySegment tensorData = fileChannel.map(
        FileChannel.MapMode.READ_ONLY,
        tensorDataOffset,
        fileChannel.size() - tensorDataOffset,
        arena
    );
    return sliceTensors(tensorData, tensorInfos);
  }

  @Nonnull
  public static Map<String, GGMLTensorEntry> sliceTensors(@Nonnull final MemorySegment tensorData,
                                                          @Nonnull final Map<String, GGUFTensorInfo> tensorInfos) {
    Map<String, GGMLTensorEntry> entries = new LinkedHashMap<>(tensorInfos.size());
    for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
      GGUFTensorInfo ti = entry.getValue();
      long numberOfElements = GGUFTensor.numberOfElementsLong(ti.dimensions());
      long sizeInBytes = ti.ggmlType().byteSizeFor(numberOfElements);
      MemorySegment seg = tensorData.asSlice(ti.offset(), sizeInBytes);
      entries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), seg));
    }
    return entries;
  }

  private void loadModelImpl(@Nonnull final ChannelReader reader) throws IOException {
    readHeader(reader);
    this.tensorInfos = new LinkedHashMap<>(tensorCount);
    for (int i = 0; i < tensorCount; ++i) {
      GGUFTensorInfo ti = readTensorInfo(reader);
      if (tensorInfos.containsKey(ti.name())) {
        throw new IllegalArgumentException("Duplicate tensor in GGUF: " + ti.name());
      }
      tensorInfos.put(ti.name(), ti);
    }
    long position = reader.position();
    int align = getAlignment();
    int padding = (int) ((align - (position % align)) % align);
    reader.skipBytes(padding);
    this.tensorDataOffset = reader.position();
  }

  private void readHeader(@Nonnull final ChannelReader reader) throws IOException {
    this.magic = reader.readInt();
    if (magic != GGUF_MAGIC) {
      throw new IllegalArgumentException("Unsupported GGUF header magic: 0x" + Integer.toHexString(magic));
    }
    this.version = reader.readInt();
    if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
      throw new IllegalArgumentException("Unsupported GGUF version: " + version);
    }
    this.tensorCount = Math.toIntExact(reader.readLong());
    this.metadataKvCount = Math.toIntExact(reader.readLong());
    this.metadata = new LinkedHashMap<>(metadataKvCount);
    for (int i = 0; i < metadataKvCount; ++i) {
      String key = readString(reader);
      Object value = readMetadataValue(reader);
      metadata.put(key, value);
    }
  }

  private GGUFTensorInfo readTensorInfo(@Nonnull final ChannelReader reader) throws IOException {
    String name = readString(reader);
    int nDimensions = reader.readInt();
    if (nDimensions < 1 || nDimensions > 4) {
      throw new IllegalArgumentException("Invalid GGUF tensor rank " + nDimensions + " for " + name);
    }
    int[] dimensions = new int[nDimensions];
    for (int i = 0; i < nDimensions; ++i) {
      dimensions[i] = Math.toIntExact(reader.readLong());
    }
    GGMLType ggmlType = GGMLType.fromId(reader.readInt());
    long offset = reader.readLong();
    return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
  }

  private String readString(@Nonnull final ChannelReader reader) throws IOException {
    int len = Math.toIntExact(reader.readLong());
    return new String(reader.readBytes(len), StandardCharsets.UTF_8);
  }

  private Object readMetadataValue(@Nonnull final ChannelReader reader) throws IOException {
    MetadataValueType valueType = MetadataValueType.fromIndex(reader.readInt());
    return readMetadataValueOfType(valueType, reader);
  }

  private Object readMetadataValueOfType(@Nonnull final MetadataValueType valueType,
                                         @Nonnull final ChannelReader reader) throws IOException {
    switch (valueType) {
      case UINT8:
      case INT8:
        return reader.readByte();
      case UINT16:
      case INT16:
        return reader.readShort();
      case UINT32:
      case INT32:
        return reader.readInt();
      case FLOAT32:
        return reader.readFloat();
      case UINT64:
      case INT64:
        return reader.readLong();
      case FLOAT64:
        return reader.readDouble();
      case BOOL:
        return reader.readByte() != 0;
      case STRING:
        return readString(reader);
      case ARRAY:
        return readArray(reader);
      default:
        throw new UnsupportedOperationException("Unsupported metadata type: " + valueType);
    }
  }

  private Object readArray(@Nonnull final ChannelReader reader) throws IOException {
    MetadataValueType valueType = MetadataValueType.fromIndex(reader.readInt());
    int len = Math.toIntExact(reader.readLong());
    switch (valueType) {
      case UINT8:
      case INT8:
        return reader.readBytes(len);
      case UINT16:
      case INT16: {
        short[] shorts = new short[len];
        for (int i = 0; i < len; ++i) {
          shorts[i] = reader.readShort();
        }
        return shorts;
      }
      case UINT32:
      case INT32: {
        int[] ints = new int[len];
        for (int i = 0; i < len; ++i) {
          ints[i] = reader.readInt();
        }
        return ints;
      }
      case UINT64:
      case INT64: {
        long[] longs = new long[len];
        for (int i = 0; i < len; ++i) {
          longs[i] = reader.readLong();
        }
        return longs;
      }
      case FLOAT32: {
        float[] floats = new float[len];
        for (int i = 0; i < len; ++i) {
          floats[i] = reader.readFloat();
        }
        return floats;
      }
      case FLOAT64: {
        double[] doubles = new double[len];
        for (int i = 0; i < len; ++i) {
          doubles[i] = reader.readDouble();
        }
        return doubles;
      }
      case BOOL: {
        boolean[] booleans = new boolean[len];
        for (int i = 0; i < len; ++i) {
          booleans[i] = reader.readByte() != 0;
        }
        return booleans;
      }
      case STRING: {
        String[] strings = new String[len];
        for (int i = 0; i < len; ++i) {
          strings[i] = readString(reader);
        }
        return strings;
      }
      case ARRAY: {
        Object[] arrays = new Object[len];
        for (int i = 0; i < len; ++i) {
          arrays[i] = readArray(reader);
        }
        return arrays;
      }
      default:
        throw new UnsupportedOperationException("Unsupported array metadata type: " + valueType);
    }
  }

  /**
   * Builder for serializing valid GGUF v3 binary files (for model export and unit testing).
   */
  @RefIgnore
  public static final class Builder {
    @RefIgnore
    private static final class TensorPayload {
      final String name;
      final int[] dims;
      final GGMLType type;
      final byte[] rawBytes;

      TensorPayload(final String name, final int[] dims, final GGMLType type, final byte[] rawBytes) {
        this.name = name;
        this.dims = dims.clone();
        this.type = type;
        this.rawBytes = rawBytes;
      }
    }

    private final Map<String, Object> metadata = new LinkedHashMap<>();
    private final List<TensorPayload> tensors = new ArrayList<>();
    private int alignment = DEFAULT_ALIGNMENT;

    @Nonnull
    public Builder putInt(@Nonnull final String key, final int value) {
      metadata.put(key, value);
      return this;
    }

    @Nonnull
    public Builder putFloat(@Nonnull final String key, final float value) {
      metadata.put(key, value);
      return this;
    }

    @Nonnull
    public Builder putBoolean(@Nonnull final String key, final boolean value) {
      metadata.put(key, value);
      return this;
    }

    @Nonnull
    public Builder putString(@Nonnull final String key, @Nonnull final String value) {
      metadata.put(key, value);
      return this;
    }

    @Nonnull
    public Builder putIntArray(@Nonnull final String key, @Nonnull final int[] values) {
      metadata.put(key, values.clone());
      return this;
    }

    @Nonnull
    public Builder putFloatArray(@Nonnull final String key, @Nonnull final float[] values) {
      metadata.put(key, values.clone());
      return this;
    }

    @Nonnull
    public Builder putBooleanArray(@Nonnull final String key, @Nonnull final boolean[] values) {
      metadata.put(key, values.clone());
      return this;
    }

    @Nonnull
    public Builder putStringArray(@Nonnull final String key, @Nonnull final String[] values) {
      metadata.put(key, values.clone());
      return this;
    }

    @Nonnull
    public Builder addFloatTensor(@Nonnull final String name, @Nonnull final int[] dims, @Nonnull final float[] values) {
      int expected = GGUFTensor.numberOfElements(dims);
      if (values.length != expected) {
        throw new IllegalArgumentException("Expected " + expected + " floats for " + name + ", got " + values.length);
      }
      ByteBuffer buf = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (float v : values) {
        buf.putFloat(v);
      }
      return addRawTensor(name, dims, GGMLType.F32, buf.array());
    }

    @Nonnull
    public Builder addRawTensor(@Nonnull final String name, @Nonnull final int[] dims,
                                @Nonnull final GGMLType type, @Nonnull final byte[] rawBytes) {
      long expectedBytes = type.byteSizeFor(GGUFTensor.numberOfElementsLong(dims));
      if (rawBytes.length != expectedBytes) {
        throw new IllegalArgumentException("Expected " + expectedBytes + " bytes for " + name + " (" + type + "), got " + rawBytes.length);
      }
      tensors.add(new TensorPayload(name, dims, type, rawBytes.clone()));
      return this;
    }

    public void writeTo(@Nonnull final Path path) throws IOException {
      try (OutputStream out = Files.newOutputStream(path)) {
        out.write(toByteArray());
      }
    }

    @Nonnull
    public byte[] toByteArray() throws IOException {
      ByteArrayOutputStream headerOut = new ByteArrayOutputStream();
      writeIntLE(headerOut, GGUF_MAGIC);
      writeIntLE(headerOut, 3);
      writeLongLE(headerOut, tensors.size());
      writeLongLE(headerOut, metadata.size());
      for (Map.Entry<String, Object> entry : metadata.entrySet()) {
        writeString(headerOut, entry.getKey());
        writeMetadataValue(headerOut, entry.getValue());
      }
      long currentDataOffset = 0L;
      for (TensorPayload tp : tensors) {
        writeString(headerOut, tp.name);
        writeIntLE(headerOut, tp.dims.length);
        for (int d : tp.dims) {
          writeLongLE(headerOut, d);
        }
        writeIntLE(headerOut, tp.type.ordinal());
        writeLongLE(headerOut, currentDataOffset);
        long padded = ((tp.rawBytes.length + alignment - 1L) / alignment) * alignment;
        currentDataOffset += padded;
      }
      int headerPadding = (int) ((alignment - (headerOut.size() % alignment)) % alignment);
      for (int i = 0; i < headerPadding; i++) {
        headerOut.write(0);
      }
      ByteArrayOutputStream fullOut = new ByteArrayOutputStream(headerOut.size() + (int) currentDataOffset);
      headerOut.writeTo(fullOut);
      for (TensorPayload tp : tensors) {
        fullOut.write(tp.rawBytes);
        int pad = (int) ((alignment - (tp.rawBytes.length % alignment)) % alignment);
        for (int i = 0; i < pad; i++) {
          fullOut.write(0);
        }
      }
      return fullOut.toByteArray();
    }

    private static void writeIntLE(final OutputStream out, final int v) throws IOException {
      ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v);
      out.write(b.array());
    }

    private static void writeLongLE(final OutputStream out, final long v) throws IOException {
      ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v);
      out.write(b.array());
    }

    private static void writeFloatLE(final OutputStream out, final float v) throws IOException {
      writeIntLE(out, Float.floatToRawIntBits(v));
    }

    private static void writeString(final OutputStream out, final String s) throws IOException {
      byte[] utf8 = s.getBytes(StandardCharsets.UTF_8);
      writeLongLE(out, utf8.length);
      out.write(utf8);
    }

    private static void writeMetadataValue(final OutputStream out, final Object val) throws IOException {
      if (val instanceof Integer) {
        writeIntLE(out, MetadataValueType.INT32.ordinal());
        writeIntLE(out, (Integer) val);
      } else if (val instanceof Float) {
        writeIntLE(out, MetadataValueType.FLOAT32.ordinal());
        writeFloatLE(out, (Float) val);
      } else if (val instanceof Boolean) {
        writeIntLE(out, MetadataValueType.BOOL.ordinal());
        out.write(((Boolean) val) ? 1 : 0);
      } else if (val instanceof String) {
        writeIntLE(out, MetadataValueType.STRING.ordinal());
        writeString(out, (String) val);
      } else if (val instanceof int[]) {
        int[] arr = (int[]) val;
        writeIntLE(out, MetadataValueType.ARRAY.ordinal());
        writeIntLE(out, MetadataValueType.INT32.ordinal());
        writeLongLE(out, arr.length);
        for (int v : arr) {
          writeIntLE(out, v);
        }
      } else if (val instanceof float[]) {
        float[] arr = (float[]) val;
        writeIntLE(out, MetadataValueType.ARRAY.ordinal());
        writeIntLE(out, MetadataValueType.FLOAT32.ordinal());
        writeLongLE(out, arr.length);
        for (float v : arr) {
          writeFloatLE(out, v);
        }
      } else if (val instanceof boolean[]) {
        boolean[] arr = (boolean[]) val;
        writeIntLE(out, MetadataValueType.ARRAY.ordinal());
        writeIntLE(out, MetadataValueType.BOOL.ordinal());
        writeLongLE(out, arr.length);
        for (boolean v : arr) {
          out.write(v ? 1 : 0);
        }
      } else if (val instanceof String[]) {
        String[] arr = (String[]) val;
        writeIntLE(out, MetadataValueType.ARRAY.ordinal());
        writeIntLE(out, MetadataValueType.STRING.ordinal());
        writeLongLE(out, arr.length);
        for (String v : arr) {
          writeString(out, v);
        }
      } else {
        throw new UnsupportedOperationException("Unsupported builder metadata value: " + val);
      }
    }
  }

  @Nonnull
  public static GGUF loadFromBytes(@Nonnull final byte[] bytes) throws IOException {
    try (ReadableByteChannel ch = Channels.newChannel(new java.io.ByteArrayInputStream(bytes))) {
      return loadModel(ch);
    }
  }
}
