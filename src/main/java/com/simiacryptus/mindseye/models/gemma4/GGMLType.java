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

/**
 * GGML tensor storage and quantization types supported in GGUF model files.
 */
@RefIgnore
public enum GGMLType {
  F32(Float.BYTES),
  F16(2),
  Q4_0(2 + 16 * Byte.BYTES, 32),
  Q4_1(2 * 2 + 16 * Byte.BYTES, 32),
  UNSUPPORTED_Q4_2(Integer.MAX_VALUE),
  UNSUPPORTED_Q4_3(Integer.MAX_VALUE),
  Q5_0(Integer.MAX_VALUE),
  Q5_1(2 * 2 + Integer.BYTES + 16 * Byte.BYTES, 32),
  Q8_0(2 + 32 * Byte.BYTES, 32),
  Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
  Q2_K(Integer.MAX_VALUE),
  Q3_K(Integer.MAX_VALUE),
  Q4_K(2 * 2 + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
  Q5_K(2 * 2 + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K),
  Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + 2, GGMLType.QK_K),
  Q8_K(Integer.MAX_VALUE),
  IQ2_XXS(Integer.MAX_VALUE),
  IQ2_XS(Integer.MAX_VALUE),
  IQ3_XXS(Integer.MAX_VALUE),
  IQ1_S(Integer.MAX_VALUE),
  IQ4_NL(Integer.MAX_VALUE),
  IQ3_S(Integer.MAX_VALUE),
  IQ2_S(Integer.MAX_VALUE),
  IQ4_XS(Integer.MAX_VALUE),
  I8(Byte.BYTES),
  I16(Short.BYTES),
  I32(Integer.BYTES),
  I64(Long.BYTES),
  F64(Double.BYTES),
  IQ1_M(Integer.MAX_VALUE),
  BF16(2),
  UNSUPPORTED_Q4_0_4_4(Integer.MAX_VALUE),
  UNSUPPORTED_Q4_0_4_8(Integer.MAX_VALUE),
  UNSUPPORTED_Q4_0_8_8(Integer.MAX_VALUE),
  TQ1_0(Integer.MAX_VALUE),
  TQ2_0(Integer.MAX_VALUE),
  UNSUPPORTED_IQ4_NL_4_4(Integer.MAX_VALUE),
  UNSUPPORTED_IQ4_NL_4_8(Integer.MAX_VALUE),
  UNSUPPORTED_IQ4_NL_8_8(Integer.MAX_VALUE),
  MXFP4(Byte.BYTES + GGMLType.QK_MXFP4 / 2, GGMLType.QK_MXFP4),
  NVFP4(Integer.MAX_VALUE);

  public static final int QK_K = 256;
  public static final int QK_MXFP4 = 32;
  public static final int FLOAT16_BYTES = 2;

  private static final GGMLType[] VALUES = values();

  private final int typeSize;
  private final int blockSize;

  GGMLType(final int typeSize) {
    this(typeSize, 1);
  }

  GGMLType(final int typeSize, final int blockSize) {
    if (blockSize <= 0 || typeSize <= 0 || (blockSize & (blockSize - 1)) != 0) {
      throw new IllegalArgumentException("Invalid GGMLType block configuration");
    }
    this.typeSize = typeSize;
    this.blockSize = blockSize;
  }

  public int getTypeSize() {
    return typeSize;
  }

  public int getBlockSize() {
    return blockSize;
  }

  public long byteSizeFor(final long numberOfElements) {
    if (typeSize == Integer.MAX_VALUE) {
      throw new UnsupportedOperationException("Unsupported GGML type: " + name());
    }
    if (numberOfElements % blockSize != 0) {
      throw new IllegalArgumentException("Element count " + numberOfElements + " is not divisible by block size " + blockSize + " for " + name());
    }
    return (numberOfElements / blockSize) * (long) typeSize;
  }

  public static GGMLType fromId(final int id) {
    if (0 <= id && id < VALUES.length) {
      return VALUES[id];
    }
    throw new UnsupportedOperationException("Unsupported GGML tensor type id: " + id);
  }
}
