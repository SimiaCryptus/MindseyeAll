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
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

/**
 * Unit tests for GGUF v3 binary parsing/building and all supported {@link GGUFTensor} quantization formats.
 */
@RefIgnore
public class GGUFAndQuantizationTest {

  private static float[] randomFloats(final int n, final long seed) {
    Random rng = new Random(seed);
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = (rng.nextFloat() * 2f - 1f) * 1.5f;
    }
    return out;
  }

  private static void assertDotMatchesScalar(final GGUFTensor tensor, final long seed) {
    int n = Math.toIntExact(tensor.size());
    GGUFTensor.ArrayFloatTensor vec = new GGUFTensor.ArrayFloatTensor(randomFloats(n, seed));

    // Aligned full dot
    float fastFull = tensor.dot(0, vec, 0, n);
    float refFull = GGUFTensor.scalarDot(tensor, 0, vec, 0, n);
    Assertions.assertEquals(refFull, fastFull, Math.max(1e-3f, Math.abs(refFull) * 1e-4f),
        "Full dot mismatch for " + tensor.type());

    // Unaligned sub-slice dot (exercises head alignment + block loop + tail)
    if (n >= 300) {
      int off = 7;
      int len = n - 19;
      float fastSub = tensor.dot(off, vec, 3, len);
      float refSub = GGUFTensor.scalarDot(tensor, off, vec, 3, len);
      Assertions.assertEquals(refSub, fastSub, Math.max(1e-3f, Math.abs(refSub) * 1e-4f),
          "Unaligned dot mismatch for " + tensor.type());
    }
  }

  @Test
  @Tag("UnitTest")
  public void testF32F16AndBF16Tensors() {
    int n = 320;
    float[] src = randomFloats(n, 101L);

    // F32
    ByteBuffer f32Buf = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : src) {
      f32Buf.putFloat(v);
    }
    MemorySegment f32Seg = Arena.ofAuto().allocate(n * 4L);
    f32Seg.copyFrom(MemorySegment.ofArray(f32Buf.array()));
    GGUFTensor.F32Tensor f32 = new GGUFTensor.F32Tensor(n, f32Seg);
    Assertions.assertEquals(src[17], f32.getFloat(17), 1e-7f);
    assertDotMatchesScalar(f32, 201L);

    // F16
    GGUFTensor.F16Tensor f16 = GGUFTensor.F16Tensor.allocate(n);
    for (int i = 0; i < n; i++) {
      f16.setFloat(i, src[i]);
      Assertions.assertEquals(Float.float16ToFloat(Float.floatToFloat16(src[i])), f16.getFloat(i), 1e-6f);
    }
    assertDotMatchesScalar(f16, 202L);

    // BF16
    ByteBuffer bf16Buf = ByteBuffer.allocate(n * 2).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : src) {
      short top16 = (short) (Float.floatToRawIntBits(v) >>> 16);
      bf16Buf.putShort(top16);
    }
    MemorySegment bf16Seg = Arena.ofAuto().allocate(n * 2L);
    bf16Seg.copyFrom(MemorySegment.ofArray(bf16Buf.array()));
    GGUFTensor.BF16Tensor bf16 = new GGUFTensor.BF16Tensor(n, bf16Seg);
    for (int i = 0; i < n; i++) {
      Assertions.assertEquals(src[i], bf16.getFloat(i), 2e-2f);
    }
    assertDotMatchesScalar(bf16, 203L);

    Tensor meTensor = f32.toMindsEyeTensor(16, 20);
    Assertions.assertArrayEquals(new int[]{16, 20}, meTensor.getDimensions());
    Assertions.assertEquals(src[42], meTensor.getData()[42], 1e-6);
    meTensor.freeRef();
  }

  @Test
  @Tag("UnitTest")
  public void testBlock32Quantizations_Q4_0_Q4_1_Q5_1_Q8_0_MXFP4() {
    int blocks = 12;
    int n = blocks * 32;
    Random rng = new Random(303L);

    // Q4_0: 18 bytes/block
    byte[] q4_0Bytes = new byte[blocks * 18];
    rng.nextBytes(q4_0Bytes);
    ByteBuffer b40 = ByteBuffer.wrap(q4_0Bytes).order(ByteOrder.LITTLE_ENDIAN);
    for (int b = 0; b < blocks; b++) {
      b40.putShort(b * 18, Float.floatToFloat16(0.125f * (b + 1)));
    }
    MemorySegment s40 = Arena.ofAuto().allocate(q4_0Bytes.length);
    s40.copyFrom(MemorySegment.ofArray(q4_0Bytes));
    assertDotMatchesScalar(new GGUFTensor.Q4_0Tensor(n, s40), 401L);

    // Q4_1: 20 bytes/block
    byte[] q4_1Bytes = new byte[blocks * 20];
    rng.nextBytes(q4_1Bytes);
    ByteBuffer b41 = ByteBuffer.wrap(q4_1Bytes).order(ByteOrder.LITTLE_ENDIAN);
    for (int b = 0; b < blocks; b++) {
      b41.putShort(b * 20, Float.floatToFloat16(0.1f));
      b41.putShort(b * 20 + 2, Float.floatToFloat16(-0.5f));
    }
    MemorySegment s41 = Arena.ofAuto().allocate(q4_1Bytes.length);
    s41.copyFrom(MemorySegment.ofArray(q4_1Bytes));
    assertDotMatchesScalar(new GGUFTensor.Q4_1Tensor(n, s41), 402L);

    // Q5_1: 24 bytes/block
    byte[] q5_1Bytes = new byte[blocks * 24];
    rng.nextBytes(q5_1Bytes);
    ByteBuffer b51 = ByteBuffer.wrap(q5_1Bytes).order(ByteOrder.LITTLE_ENDIAN);
    for (int b = 0; b < blocks; b++) {
      b51.putShort(b * 24, Float.floatToFloat16(0.05f));
      b51.putShort(b * 24 + 2, Float.floatToFloat16(-0.25f));
    }
    MemorySegment s51 = Arena.ofAuto().allocate(q5_1Bytes.length);
    s51.copyFrom(MemorySegment.ofArray(q5_1Bytes));
    assertDotMatchesScalar(new GGUFTensor.Q5_1Tensor(n, s51), 403L);

    // Q8_0: 34 bytes/block
    byte[] q8_0Bytes = new byte[blocks * 34];
    rng.nextBytes(q8_0Bytes);
    ByteBuffer b80 = ByteBuffer.wrap(q8_0Bytes).order(ByteOrder.LITTLE_ENDIAN);
    for (int b = 0; b < blocks; b++) {
      b80.putShort(b * 34, Float.floatToFloat16(0.02f * (b + 1)));
    }
    MemorySegment s80 = Arena.ofAuto().allocate(q8_0Bytes.length);
    s80.copyFrom(MemorySegment.ofArray(q8_0Bytes));
    assertDotMatchesScalar(new GGUFTensor.Q8_0Tensor(n, s80), 404L);

    // MXFP4: 17 bytes/block (first byte is E8M0 exponent in sensible range 120..130)
    byte[] mxfpBytes = new byte[blocks * 17];
    rng.nextBytes(mxfpBytes);
    for (int b = 0; b < blocks; b++) {
      mxfpBytes[b * 17] = (byte) (124 + (b % 5));
    }
    MemorySegment sMxfp = Arena.ofAuto().allocate(mxfpBytes.length);
    sMxfp.copyFrom(MemorySegment.ofArray(mxfpBytes));
    assertDotMatchesScalar(new GGUFTensor.MXFP4Tensor(n, sMxfp), 405L);
  }

  @Test
  @Tag("UnitTest")
  public void testKQuantizations_Q4_K_Q5_K_Q6_K() {
    int blocks = 3;
    int n = blocks * 256;
    Random rng = new Random(505L);

    // Q4_K: 144 bytes/block
    byte[] q4k = new byte[blocks * 144];
    rng.nextBytes(q4k);
    ByteBuffer b4k = ByteBuffer.wrap(q4k).order(ByteOrder.LITTLE_ENDIAN);
    for (int b = 0; b < blocks; b++) {
      b4k.putShort(b * 144, Float.floatToFloat16(0.03f));
      b4k.putShort(b * 144 + 2, Float.floatToFloat16(0.01f));
    }
    MemorySegment s4k = Arena.ofAuto().allocate(q4k.length);
    s4k.copyFrom(MemorySegment.ofArray(q4k));
    assertDotMatchesScalar(new GGUFTensor.Q4_KTensor(n, s4k), 601L);

    // Q5_K: 176 bytes/block
    byte[] q5k = new byte[blocks * 176];
    rng.nextBytes(q5k);
    ByteBuffer b5k = ByteBuffer.wrap(q5k).order(ByteOrder.LITTLE_ENDIAN);
    for (int b = 0; b < blocks; b++) {
      b5k.putShort(b * 176, Float.floatToFloat16(0.025f));
      b5k.putShort(b * 176 + 2, Float.floatToFloat16(0.015f));
    }
    MemorySegment s5k = Arena.ofAuto().allocate(q5k.length);
    s5k.copyFrom(MemorySegment.ofArray(q5k));
    assertDotMatchesScalar(new GGUFTensor.Q5_KTensor(n, s5k), 602L);

    // Q6_K: 210 bytes/block (fp16 d at offset 208)
    byte[] q6k = new byte[blocks * 210];
    rng.nextBytes(q6k);
    ByteBuffer b6k = ByteBuffer.wrap(q6k).order(ByteOrder.LITTLE_ENDIAN);
    for (int b = 0; b < blocks; b++) {
      b6k.putShort(b * 210 + 208, Float.floatToFloat16(0.02f));
    }
    MemorySegment s6k = Arena.ofAuto().allocate(q6k.length);
    s6k.copyFrom(MemorySegment.ofArray(q6k));
    assertDotMatchesScalar(new GGUFTensor.Q6_KTensor(n, s6k), 603L);
  }

  @Test
  @Tag("UnitTest")
  public void testGGUFBuilderAndParserRoundTrip(@TempDir final Path tempDir) throws IOException {
    float[] t1 = randomFloats(64, 11L);
    byte[] q8Raw = new byte[34 * 2];
    ByteBuffer.wrap(q8Raw).order(ByteOrder.LITTLE_ENDIAN).putShort(0, Float.floatToFloat16(0.5f));
    ByteBuffer.wrap(q8Raw).order(ByteOrder.LITTLE_ENDIAN).putShort(34, Float.floatToFloat16(0.25f));
    q8Raw[2] = 4;
    q8Raw[36] = -8;

    Path ggufFile = tempDir.resolve("test_roundtrip.gguf");
    new GGUF.Builder()
        .putString("general.architecture", "gemma4")
        .putInt("gemma4.embedding_length", 16)
        .putFloat("gemma4.rope.freq_base", 1000000f)
        .putBoolean("custom.flag", true)
        .putIntArray("gemma4.feed_forward_length", new int[]{32, 64})
        .putBooleanArray("gemma4.attention.sliding_window_pattern", new boolean[]{true, false})
        .putStringArray("tokenizer.ggml.tokens", new String[]{"<pad>", "<eos>", "<bos>", "hello"})
        .putFloatArray("tokenizer.ggml.scores", new float[]{0f, 0f, 0f, 1.5f})
        .addFloatTensor("weight_f32", new int[]{16, 4}, t1)
        .addRawTensor("weight_q8", new int[]{32, 2}, GGMLType.Q8_0, q8Raw)
        .writeTo(ggufFile);

    try (java.nio.channels.FileChannel ch = java.nio.channels.FileChannel.open(ggufFile)) {
      GGUF parsed = GGUF.loadModel(ch);
      Assertions.assertEquals(3, parsed.getVersion());
      Assertions.assertEquals(2, parsed.getTensorCount());
      Assertions.assertEquals("gemma4", parsed.getMetadata().get("general.architecture"));
      Assertions.assertEquals(16, parsed.getMetadata().get("gemma4.embedding_length"));
      Assertions.assertEquals(Boolean.TRUE, parsed.getMetadata().get("custom.flag"));
      Assertions.assertArrayEquals(new int[]{32, 64}, (int[]) parsed.getMetadata().get("gemma4.feed_forward_length"));

      Map<String, GGUF.GGMLTensorEntry> entries = GGUF.loadTensors(ch, parsed.getTensorDataOffset(), parsed.getTensorInfos());
      GGUFTensor loadedF32 = GGUFTensor.fromEntry(entries.get("weight_f32"));
      GGUFTensor loadedQ8 = GGUFTensor.fromEntry(entries.get("weight_q8"));
      Assertions.assertEquals(t1[19], loadedF32.getFloat(19), 1e-6f);
      Assertions.assertEquals(2.0f, loadedQ8.getFloat(0), 1e-5f);
      Assertions.assertEquals(-2.0f, loadedQ8.getFloat(32), 1e-5f);
    }
  }
}
