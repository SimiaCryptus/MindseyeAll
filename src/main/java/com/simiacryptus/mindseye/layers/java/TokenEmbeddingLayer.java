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
import com.simiacryptus.mindseye.lang.Delta;
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
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * Token embedding lookup layer mapping integer token IDs {@code Tensor(seqLen)} to {@code [embeddingDim, seqLen]}
 * with an optional constant multiplier {@code scale} (e.g. {@code sqrt(embeddingDim)} in Gemma 4).
 */
@SuppressWarnings("serial")
public class TokenEmbeddingLayer extends LayerBase {

  private final int vocabSize;
  private final int embeddingDim;
  private final double scale;
  @Nonnull
  private final Tensor weights;

  public TokenEmbeddingLayer(final int vocabSize, final int embeddingDim) {
    this(vocabSize, embeddingDim, 1.0);
  }

  public TokenEmbeddingLayer(final int vocabSize, final int embeddingDim, final double scale) {
    super();
    if (vocabSize <= 0 || embeddingDim <= 0) {
      throw new IllegalArgumentException("Invalid embedding dimensions: " + vocabSize + " x " + embeddingDim);
    }
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.scale = scale;
    this.weights = new Tensor(embeddingDim, vocabSize);
    double std = 1.0 / Math.sqrt(embeddingDim);
    double[] w = this.weights.getData();
    for (int i = 0; i < w.length; i++) {
      w[i] = (1.0 - 2.0 * Util.R.get().nextDouble()) * std;
    }
  }

  public TokenEmbeddingLayer(final int vocabSize, final int embeddingDim, final double scale,
                             @Nonnull final Tensor weights) {
    super();
    if (weights.length() != vocabSize * embeddingDim) {
      weights.freeRef();
      throw new IllegalArgumentException("Embedding table size mismatch");
    }
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.scale = scale;
    this.weights = weights;
  }

  protected TokenEmbeddingLayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    this.vocabSize = json.get("vocabSize").getAsInt();
    this.embeddingDim = json.get("embeddingDim").getAsInt();
    this.scale = json.get("scale").getAsDouble();
    Tensor loaded = Tensor.fromJson(json.get("weights"), resources);
    if (loaded == null) {
      throw new IllegalArgumentException("Missing weights in TokenEmbeddingLayer JSON");
    }
    this.weights = loaded;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static TokenEmbeddingLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new TokenEmbeddingLayer(json, rs);
  }

  public int getVocabSize() {
    return vocabSize;
  }

  public int getEmbeddingDim() {
    return embeddingDim;
  }

  public double getScale() {
    return scale;
  }

  @Nonnull
  public Tensor getWeights() {
    return weights.addRef();
  }

  public void setWeights(@Nonnull final double[] values) {
    if (values.length != vocabSize * embeddingDim) {
      throw new IllegalArgumentException("Expected " + (vocabSize * embeddingDim) + " values, got " + values.length);
    }
    System.arraycopy(values, 0, weights.getData(), 0, values.length);
  }

  @RefIgnore
  static void lookupForward(@Nonnull final double[] tokenIds, @Nonnull final double[] table,
                            @Nonnull final double[] out, final int vocabSize,
                            final int embeddingDim, final double scale) {
    for (int t = 0; t < tokenIds.length; t++) {
      int tok = (int) Math.round(tokenIds[t]);
      if (tok < 0 || tok >= vocabSize) {
        throw new IndexOutOfBoundsException("Token index " + tok + " out of range [0, " + vocabSize + ")");
      }
      int srcOff = tok * embeddingDim;
      int dstOff = t * embeddingDim;
      for (int d = 0; d < embeddingDim; d++) {
        out[dstOff + d] = table[srcOff + d] * scale;
      }
    }
  }

  @RefIgnore
  static void lookupBackward(@Nonnull final double[] tokenIds, @Nonnull final double[] dy,
                             @Nonnull final double[] dTable, final int vocabSize,
                             final int embeddingDim, final double scale) {
    for (int t = 0; t < tokenIds.length; t++) {
      int tok = (int) Math.round(tokenIds[t]);
      if (tok >= 0 && tok < vocabSize) {
        int dstOff = tok * embeddingDim;
        int srcOff = t * embeddingDim;
        for (int d = 0; d < embeddingDim; d++) {
          dTable[dstOff + d] += dy[srcOff + d] * scale;
        }
      }
    }
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (inObj.length != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("TokenEmbeddingLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int seqLen = Tensor.length(inputData.getDimensions());
    int[] outDims = new int[]{embeddingDim, seqLen};
    int batchSize = inputData.length();
    double[] tableData = weights.getData();
    TensorArray outputData = new TensorArray(RefIntStream.range(0, batchSize)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
          Tensor inTensor = inputData.get(b);
          Tensor outTensor = new Tensor(outDims);
          lookupForward(inTensor.getData(), tableData, outTensor.getData(), vocabSize, embeddingDim, scale);
          inTensor.freeRef();
          return outTensor;
        }, inputData.addRef()))
        .toArray(Tensor[]::new));
    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(inputData, weights.addRef(), vocabSize, embeddingDim, scale, isFrozen(), upstream, alive);
    return new Result(outputData, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("vocabSize", vocabSize);
    json.addProperty("embeddingDim", embeddingDim);
    json.addProperty("scale", scale);
    json.add("weights", weights.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(weights.getData());
  }

  @Override
  public void _free() {
    weights.freeRef();
    super._free();
  }

  @Nonnull
  @Override
  public TokenEmbeddingLayer addRef() {
    return (TokenEmbeddingLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList inputData;
    private final Tensor weights;
    private final int vocabSize;
    private final int embeddingDim;
    private final double scale;
    private final boolean frozen;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final Tensor weights, final int vocabSize, final int embeddingDim,
                final double scale, final boolean frozen, final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.weights = weights;
      this.vocabSize = vocabSize;
      this.embeddingDim = embeddingDim;
      this.scale = scale;
      this.frozen = frozen;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      int batchSize = inputData.length();
      if (!frozen && buffer != null) {
        double[] dTable = new double[vocabSize * embeddingDim];
        for (int b = 0; b < batchSize; b++) {
          Tensor inTensor = inputData.get(b);
          Tensor dTensor = delta.get(b);
          lookupBackward(inTensor.getData(), dTensor.getData(), dTable, vocabSize, embeddingDim, scale);
          dTensor.freeRef();
          inTensor.freeRef();
        }
        Delta<UUID> deltaBuf = buffer.get(weights.getId(), weights.getData());
        if (deltaBuf != null) {
          deltaBuf.addInPlace(dTable);
          deltaBuf.freeRef();
        }
      }
      if (alive) {
        int[] inDims = inputData.getDimensions();
        TensorArray zeroPassback = new TensorArray(RefIntStream.range(0, batchSize)
            .mapToObj(b -> new Tensor(inDims))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> bufferCopy = buffer == null ? null : buffer.addRef();
        upstream.accept(bufferCopy, zeroPassback);
      }
      delta.freeRef();
      if (buffer != null) {
        buffer.freeRef();
      }
    }

    @Override
    public void _free() {
      super._free();
      inputData.freeRef();
      weights.freeRef();
      upstream.freeRef();
    }
  }
}
