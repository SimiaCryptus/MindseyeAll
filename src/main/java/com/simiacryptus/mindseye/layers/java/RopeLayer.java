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

import com.google.gson.JsonArray;
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
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * NeoX-style Rotary Positional Embedding (RoPE) layer operating on {@code [numHeads * headSize, seqLen]}
 * (or 1D {@code [numHeads * headSize]}). Supports both standard {@code theta} frequencies and Gemma 4
 * per-frequency scaling divisors ({@code rope_freqs.weight}).
 */
@SuppressWarnings("serial")
public class RopeLayer extends LayerBase {

  private final int numHeads;
  private final int headSize;
  private final double ropeTheta;
  @Nullable
  private final double[] freqFactors;
  private int startPosition;

  public RopeLayer(final int numHeads, final int headSize, final double ropeTheta) {
    this(numHeads, headSize, ropeTheta, null, 0);
  }

  public RopeLayer(final int numHeads, final int headSize, final double ropeTheta,
                   @Nullable final double[] freqFactors, final int startPosition) {
    super();
    if (numHeads <= 0 || headSize <= 0 || (headSize & 1) != 0) {
      throw new IllegalArgumentException("Invalid RoPE head configuration: heads=" + numHeads + ", headSize=" + headSize);
    }
    if (freqFactors != null && freqFactors.length != headSize / 2) {
      throw new IllegalArgumentException("freqFactors length " + freqFactors.length + " != headSize/2 (" + (headSize / 2) + ")");
    }
    this.numHeads = numHeads;
    this.headSize = headSize;
    this.ropeTheta = ropeTheta;
    this.freqFactors = freqFactors == null ? null : freqFactors.clone();
    this.startPosition = startPosition;
    this.frozen = true;
  }

  protected RopeLayer(@Nonnull final JsonObject json) {
    super(json);
    this.numHeads = json.get("numHeads").getAsInt();
    this.headSize = json.get("headSize").getAsInt();
    this.ropeTheta = json.get("ropeTheta").getAsDouble();
    this.startPosition = json.has("startPosition") ? json.get("startPosition").getAsInt() : 0;
    if (json.has("freqFactors")) {
      JsonArray arr = json.getAsJsonArray("freqFactors");
      this.freqFactors = new double[arr.size()];
      for (int i = 0; i < arr.size(); i++) {
        this.freqFactors[i] = arr.get(i).getAsDouble();
      }
    } else {
      this.freqFactors = null;
    }
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static RopeLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new RopeLayer(json);
  }

  public int getStartPosition() {
    return startPosition;
  }

  public void setStartPosition(final int startPosition) {
    this.startPosition = startPosition;
  }

  @RefIgnore
  static void applyRope(@Nonnull final double[] src, @Nonnull final double[] dst,
                        final int numHeads, final int headSize, final double ropeTheta,
                        @Nullable final double[] freqFactors, final int startPos, final boolean inverse) {
    int totalHeadDim = numHeads * headSize;
    if (src.length % totalHeadDim != 0) {
      throw new IllegalArgumentException("Input length " + src.length + " is not divisible by " + totalHeadDim);
    }
    int seqLen = src.length / totalHeadDim;
    int halfHead = headSize / 2;
    double[] invFreq = new double[halfHead];
    for (int ic = 0; ic < halfHead; ic++) {
      double base = 1.0 / Math.pow(ropeTheta, (2.0 * ic) / headSize);
      invFreq[ic] = freqFactors != null ? (base / freqFactors[ic]) : base;
    }

    for (int t = 0; t < seqLen; t++) {
      int pos = startPos + t;
      int seqOffset = t * totalHeadDim;
      for (int ic = 0; ic < halfHead; ic++) {
        double angle = pos * invFreq[ic];
        double fcr = Math.cos(angle);
        double fci = inverse ? -Math.sin(angle) : Math.sin(angle);
        for (int h = 0; h < numHeads; h++) {
          int hOff = seqOffset + h * headSize;
          double v0 = src[hOff + ic];
          double v1 = src[hOff + ic + halfHead];
          dst[hOff + ic] = v0 * fcr - v1 * fci;
          dst[hOff + ic + halfHead] = v0 * fci + v1 * fcr;
        }
      }
    }
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (inObj.length != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("RopeLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int batchSize = inputData.length();
    final int capturedStart = this.startPosition;
    final double[] capturedFactors = this.freqFactors;
    TensorArray outputData = new TensorArray(RefIntStream.range(0, batchSize)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
          Tensor inTensor = inputData.get(b);
          Tensor outTensor = new Tensor(inTensor.getDimensions());
          applyRope(inTensor.getData(), outTensor.getData(), numHeads, headSize, ropeTheta, capturedFactors, capturedStart, false);
          inTensor.freeRef();
          return outTensor;
        }, inputData))
        .toArray(Tensor[]::new));
    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(numHeads, headSize, ropeTheta, capturedFactors, capturedStart, upstream, alive);
    return new Result(outputData, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("numHeads", numHeads);
    json.addProperty("headSize", headSize);
    json.addProperty("ropeTheta", ropeTheta);
    json.addProperty("startPosition", startPosition);
    if (freqFactors != null) {
      JsonArray arr = new JsonArray();
      for (double f : freqFactors) {
        arr.add(f);
      }
      json.add("freqFactors", arr);
    }
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
  public RopeLayer addRef() {
    return (RopeLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final int numHeads;
    private final int headSize;
    private final double ropeTheta;
    @Nullable
    private final double[] freqFactors;
    private final int startPos;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final int numHeads, final int headSize, final double ropeTheta,
                @Nullable final double[] freqFactors, final int startPos,
                final Result.Accumulator upstream, final boolean alive) {
      this.numHeads = numHeads;
      this.headSize = headSize;
      this.ropeTheta = ropeTheta;
      this.freqFactors = freqFactors;
      this.startPos = startPos;
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
              Tensor dxTensor = new Tensor(dTensor.getDimensions());
              applyRope(dTensor.getData(), dxTensor.getData(), numHeads, headSize, ropeTheta, freqFactors, startPos, true);
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
