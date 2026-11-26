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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.nio.FloatBuffer;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * Root Mean Square Layer Normalization (RMSNorm) operating over contiguous chunks of {@code normSize}.
 * Supports both learned per-feature scale weights and bare unit-RMS normalization ({@code hasWeights == false}).
 */
@SuppressWarnings("serial")
public class RmsNormLayer extends LayerBase {

  private final int normSize;
  private final double eps;
  private final boolean hasWeights;
  @Nullable
  private final Tensor weights;

  public RmsNormLayer(final int normSize) {
    this(normSize, 1e-6, true);
  }

  public RmsNormLayer(final int normSize, final double eps) {
    this(normSize, eps, true);
  }

  public RmsNormLayer(final int normSize, final double eps, final boolean hasWeights) {
    super();
    if (normSize <= 0) {
      throw new IllegalArgumentException("normSize must be positive: " + normSize);
    }
    this.normSize = normSize;
    this.eps = eps;
    this.hasWeights = hasWeights;
    if (hasWeights) {
      this.weights = new Tensor(normSize);
      this.weights.setAll(1.0);
    } else {
      this.weights = null;
      this.frozen = true;
    }
  }

  public RmsNormLayer(final int normSize, final double eps, final boolean hasWeights, @Nullable final Tensor weights) {
    super();
    if (normSize <= 0) {
      if (weights != null) {
        weights.freeRef();
      }
      throw new IllegalArgumentException("normSize must be positive: " + normSize);
    }
    this.normSize = normSize;
    this.eps = eps;
    this.hasWeights = hasWeights;
    if (hasWeights) {
      if (weights == null || weights.length() != normSize) {
        if (weights != null) {
          weights.freeRef();
        }
        throw new IllegalArgumentException("Expected weights of length " + normSize);
      }
      this.weights = weights;
    } else {
      if (weights != null) {
        weights.freeRef();
      }
      this.weights = null;
      this.frozen = true;
    }
  }

  protected RmsNormLayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    this.normSize = json.get("normSize").getAsInt();
    this.eps = json.get("eps").getAsDouble();
    this.hasWeights = json.get("hasWeights").getAsBoolean();
    this.weights = hasWeights ? Tensor.fromJson(json.get("weights"), resources) : null;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static RmsNormLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new RmsNormLayer(json, rs);
  }

  public int getNormSize() {
    return normSize;
  }

  public double getEps() {
    return eps;
  }

  public boolean hasWeights() {
    return hasWeights;
  }

  @Nullable
  public Tensor getWeights() {
    return weights == null ? null : weights.addRef();
  }

  public void setWeights(@Nonnull final double[] values) {
    if (!hasWeights || weights == null) {
      throw new IllegalStateException("RmsNormLayer was configured without learned weights");
    }
    if (values.length != normSize) {
      throw new IllegalArgumentException("Expected " + normSize + " weights, got " + values.length);
    }
    double[] target = weights.getData();
    System.arraycopy(values, 0, target, 0, normSize);
  }

  public void setWeights(@Nonnull final float[] values) {
    if (!hasWeights || weights == null) {
      throw new IllegalStateException("RmsNormLayer was configured without learned weights");
    }
    if (values.length != normSize) {
      throw new IllegalArgumentException("Expected " + normSize + " weights, got " + values.length);
    }
    double[] target = weights.getData();
    for (int i = 0; i < normSize; i++) {
      target[i] = values[i];
    }
  }

  public void setWeights(@Nonnull final FloatBuffer buffer) {
    float[] arr = new float[normSize];
    int pos = buffer.position();
    buffer.get(arr);
    buffer.position(pos);
    setWeights(arr);
  }

  @RefIgnore
  static void forwardArray(@Nonnull final double[] in, @Nonnull final double[] out,
                           @Nullable final double[] w, final int normSize, final double eps) {
    if (in.length % normSize != 0) {
      throw new IllegalArgumentException("Input length " + in.length + " is not divisible by normSize " + normSize);
    }
    int chunks = in.length / normSize;
    for (int c = 0; c < chunks; c++) {
      int base = c * normSize;
      double ss = 0.0;
      for (int i = 0; i < normSize; i++) {
        double xi = in[base + i];
        ss += xi * xi;
      }
      double r = 1.0 / Math.sqrt(ss / normSize + eps);
      if (w != null) {
        for (int i = 0; i < normSize; i++) {
          out[base + i] = w[i] * r * in[base + i];
        }
      } else {
        for (int i = 0; i < normSize; i++) {
          out[base + i] = r * in[base + i];
        }
      }
    }
  }

  @RefIgnore
  static void backwardWeightArray(@Nonnull final double[] in, @Nonnull final double[] delta,
                                  @Nonnull final double[] wGrad, final int normSize, final double eps) {
    int chunks = in.length / normSize;
    for (int c = 0; c < chunks; c++) {
      int base = c * normSize;
      double ss = 0.0;
      for (int i = 0; i < normSize; i++) {
        double xi = in[base + i];
        ss += xi * xi;
      }
      double r = 1.0 / Math.sqrt(ss / normSize + eps);
      for (int i = 0; i < normSize; i++) {
        wGrad[i] += delta[base + i] * r * in[base + i];
      }
    }
  }

  @RefIgnore
  static void backwardInputArray(@Nonnull final double[] in, @Nonnull final double[] delta,
                                 @Nullable final double[] w, @Nonnull final double[] dx,
                                 final int normSize, final double eps) {
    int chunks = in.length / normSize;
    for (int c = 0; c < chunks; c++) {
      int base = c * normSize;
      double ss = 0.0;
      for (int i = 0; i < normSize; i++) {
        double xi = in[base + i];
        ss += xi * xi;
      }
      double r = 1.0 / Math.sqrt(ss / normSize + eps);
      double r2OverD = (r * r) / normSize;
      double dot = 0.0;
      for (int i = 0; i < normSize; i++) {
        double ui = w != null ? delta[base + i] * w[i] : delta[base + i];
        dot += ui * in[base + i];
      }
      double coeff = r2OverD * dot;
      for (int i = 0; i < normSize; i++) {
        double ui = w != null ? delta[base + i] * w[i] : delta[base + i];
        dx[base + i] = r * (ui - coeff * in[base + i]);
      }
    }
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (inObj.length != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("RmsNormLayer requires exactly 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int batchSize = inputData.length();
    double[] weightArr = (hasWeights && weights != null) ? weights.getData() : null;
    TensorArray outputData = new TensorArray(RefIntStream.range(0, batchSize)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
          Tensor inTensor = inputData.get(b);
          Tensor outTensor = new Tensor(inTensor.getDimensions());
          forwardArray(inTensor.getData(), outTensor.getData(), weightArr, normSize, eps);
          inTensor.freeRef();
          return outTensor;
        }, inputData.addRef()))
        .toArray(Tensor[]::new));
    boolean alive = in0.isAlive();
    boolean trainable = hasWeights && !isFrozen();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Tensor weightsRef = weights == null ? null : weights.addRef();
    Accumulator accumulator = new Accumulator(inputData, weightsRef, normSize, eps, hasWeights, isFrozen(), getId(), upstream, alive);
    return new Result(outputData, accumulator, alive || trainable);
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("normSize", normSize);
    json.addProperty("eps", eps);
    json.addProperty("hasWeights", hasWeights);
    if (hasWeights && weights != null) {
      json.add("weights", weights.getJson(resources, dataSerializer));
    }
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    if (hasWeights && weights != null) {
      return RefArrays.asList(weights.getData());
    }
    return RefArrays.asList();
  }

  @Override
  public void _free() {
    if (weights != null) {
      weights.freeRef();
    }
    super._free();
  }

  @Nonnull
  @Override
  public RmsNormLayer addRef() {
    return (RmsNormLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList inputData;
    @Nullable
    private final Tensor weights;
    private final int normSize;
    private final double eps;
    private final boolean hasWeights;
    private final boolean frozen;
    private final UUID layerId;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, @Nullable final Tensor weights, final int normSize,
                final double eps, final boolean hasWeights, final boolean frozen, final UUID layerId,
                final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.weights = weights;
      this.normSize = normSize;
      this.eps = eps;
      this.hasWeights = hasWeights;
      this.frozen = frozen;
      this.layerId = layerId;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      int batchSize = inputData.length();
      if (hasWeights && !frozen && weights != null && buffer != null) {
        double[] wGrad = new double[normSize];
        for (int b = 0; b < batchSize; b++) {
          Tensor inTensor = inputData.get(b);
          Tensor dTensor = delta.get(b);
          backwardWeightArray(inTensor.getData(), dTensor.getData(), wGrad, normSize, eps);
          dTensor.freeRef();
          inTensor.freeRef();
        }
        Delta<UUID> deltaBuf = buffer.get(layerId, weights.getData());
        if (deltaBuf != null) {
          deltaBuf.addInPlace(wGrad);
          deltaBuf.freeRef();
        }
      }
      if (alive) {
        double[] weightArr = (hasWeights && weights != null) ? weights.getData() : null;
        TensorArray passback = new TensorArray(RefIntStream.range(0, batchSize)
            .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
              Tensor inTensor = inputData.get(b);
              Tensor dTensor = delta.get(b);
              Tensor dxTensor = new Tensor(inTensor.getDimensions());
              backwardInputArray(inTensor.getData(), dTensor.getData(), weightArr, dxTensor.getData(), normSize, eps);
              dTensor.freeRef();
              inTensor.freeRef();
              return dxTensor;
            }, inputData.addRef(), delta.addRef()))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> bufferCopy = buffer == null ? null : buffer.addRef();
        upstream.accept(bufferCopy, passback);
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
      if (weights != null) {
        weights.freeRef();
      }
      upstream.freeRef();
    }
  }
}
