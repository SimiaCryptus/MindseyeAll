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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * Extracts a contiguous sub-vector {@code [chunkIndex * chunkSize, (chunkIndex + 1) * chunkSize)}
 * from the leading dimension {@code totalChunks * chunkSize} at each sequence position.
 * Used in Gemma 4 to slice per-layer embeddings for transformer block {@code l}.
 */
@SuppressWarnings("serial")
public class ChunkSelectLayer extends LayerBase {

  private final int chunkSize;
  private final int totalChunks;
  private final int chunkIndex;

  public ChunkSelectLayer(final int chunkSize, final int totalChunks, final int chunkIndex) {
    super();
    if (chunkSize <= 0 || totalChunks <= 0 || chunkIndex < 0 || chunkIndex >= totalChunks) {
      throw new IllegalArgumentException(
          "Invalid ChunkSelectLayer params: chunkSize=" + chunkSize + ", totalChunks=" + totalChunks + ", chunkIndex=" + chunkIndex);
    }
    this.chunkSize = chunkSize;
    this.totalChunks = totalChunks;
    this.chunkIndex = chunkIndex;
    this.frozen = true;
  }

  protected ChunkSelectLayer(@Nonnull final JsonObject json) {
    super(json);
    this.chunkSize = json.get("chunkSize").getAsInt();
    this.totalChunks = json.get("totalChunks").getAsInt();
    this.chunkIndex = json.get("chunkIndex").getAsInt();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ChunkSelectLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new ChunkSelectLayer(json);
  }

  @RefIgnore
  @Nonnull
  static int[] computeOutDims(@Nonnull final int[] inDims, final int chunkSize, final int totalChunks) {
    int expectedDim0 = chunkSize * totalChunks;
    if (inDims.length == 0 || inDims[0] != expectedDim0) {
      throw new IllegalArgumentException("Expected leading dim " + expectedDim0 + ", got " + Arrays.toString(inDims));
    }
    int[] outDims = inDims.clone();
    outDims[0] = chunkSize;
    return outDims;
  }

  @RefIgnore
  static void sliceForward(@Nonnull final double[] src, @Nonnull final double[] dst,
                           final int chunkSize, final int totalChunks, final int chunkIndex) {
    int stride = chunkSize * totalChunks;
    int positions = src.length / stride;
    int offset = chunkIndex * chunkSize;
    for (int p = 0; p < positions; p++) {
      System.arraycopy(src, p * stride + offset, dst, p * chunkSize, chunkSize);
    }
  }

  @RefIgnore
  static void sliceBackward(@Nonnull final double[] dy, @Nonnull final double[] dx,
                            final int chunkSize, final int totalChunks, final int chunkIndex) {
    int stride = chunkSize * totalChunks;
    int positions = dy.length / chunkSize;
    int offset = chunkIndex * chunkSize;
    for (int p = 0; p < positions; p++) {
      System.arraycopy(dy, p * chunkSize, dx, p * stride + offset, chunkSize);
    }
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (inObj.length != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("ChunkSelectLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] inDims = inputData.getDimensions();
    int[] outDims = computeOutDims(inDims, chunkSize, totalChunks);
    int batchSize = inputData.length();
    TensorArray outputData = new TensorArray(RefIntStream.range(0, batchSize)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
          Tensor inTensor = inputData.get(b);
          Tensor outTensor = new Tensor(outDims);
          sliceForward(inTensor.getData(), outTensor.getData(), chunkSize, totalChunks, chunkIndex);
          inTensor.freeRef();
          return outTensor;
        }, inputData))
        .toArray(Tensor[]::new));
    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(inDims, chunkSize, totalChunks, chunkIndex, upstream, alive);
    return new Result(outputData, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("chunkSize", chunkSize);
    json.addProperty("totalChunks", totalChunks);
    json.addProperty("chunkIndex", chunkIndex);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  @Override
  public void _free() {
    super._free();
  }

  @Nonnull
  @Override
  public ChunkSelectLayer addRef() {
    return (ChunkSelectLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final int[] inDims;
    private final int chunkSize;
    private final int totalChunks;
    private final int chunkIndex;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final int[] inDims, final int chunkSize, final int totalChunks,
                final int chunkIndex, final Result.Accumulator upstream, final boolean alive) {
      this.inDims = inDims;
      this.chunkSize = chunkSize;
      this.totalChunks = totalChunks;
      this.chunkIndex = chunkIndex;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      if (alive) {
        int batchSize = delta.length();
        TensorArray passback = new TensorArray(RefIntStream.range(0, batchSize)
            .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
              Tensor dTensor = delta.get(b);
              Tensor dxTensor = new Tensor(inDims);
              sliceBackward(dTensor.getData(), dxTensor.getData(), chunkSize, totalChunks, chunkIndex);
              dTensor.freeRef();
              return dxTensor;
            }, delta))
            .toArray(Tensor[]::new));
        upstream.accept(buffer, passback);
      } else {
        delta.freeRef();
        if (buffer != null) {
          buffer.freeRef();
        }
      }
    }

    @Override
    public void _free() {
      super._free();
      upstream.freeRef();
    }
  }
}
