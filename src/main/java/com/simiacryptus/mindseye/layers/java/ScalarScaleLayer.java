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
 * Pure scalar multiplication layer ({@code y = scale * x}) with a single scalar parameter (no bias).
 * Used for Gemma 4 per-layer output scaling and fixed normalization constants.
 */
@SuppressWarnings("serial")
public class ScalarScaleLayer extends LayerBase {

  @Nonnull
  private final Tensor weight;

  public ScalarScaleLayer(final double scale) {
    this(scale, false);
  }

  public ScalarScaleLayer(final double scale, final boolean frozen) {
    super();
    this.weight = new Tensor(1);
    this.weight.set(0, scale);
    this.frozen = frozen;
  }

  public ScalarScaleLayer(@Nonnull final Tensor weight, final boolean frozen) {
    super();
    if (weight.length() != 1) {
      weight.freeRef();
      throw new IllegalArgumentException("ScalarScaleLayer expects weight of length 1");
    }
    this.weight = weight;
    this.frozen = frozen;
  }

  protected ScalarScaleLayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    Tensor loaded = Tensor.fromJson(json.get("weight"), resources);
    if (loaded == null) {
      throw new IllegalArgumentException("Missing weight in ScalarScaleLayer JSON");
    }
    this.weight = loaded;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ScalarScaleLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new ScalarScaleLayer(json, rs);
  }

  public double getScale() {
    return weight.get(0);
  }

  public void setScale(final double scale) {
    weight.set(0, scale);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (inObj.length != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("ScalarScaleLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int batchSize = inputData.length();
    double scale = weight.get(0);
    TensorArray outputData = new TensorArray(RefIntStream.range(0, batchSize)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
          Tensor inTensor = inputData.get(b);
          Tensor outTensor = new Tensor(inTensor.getDimensions());
          double[] src = inTensor.getData();
          double[] dst = outTensor.getData();
          for (int i = 0; i < src.length; i++) {
            dst[i] = src[i] * scale;
          }
          inTensor.freeRef();
          return outTensor;
        }, inputData.addRef()))
        .toArray(Tensor[]::new));
    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(inputData, weight.addRef(), isFrozen(), getId(), upstream, alive);
    return new Result(outputData, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.add("weight", weight.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(weight.getData());
  }

  @Override
  public void _free() {
    weight.freeRef();
    super._free();
  }

  @Nonnull
  @Override
  public ScalarScaleLayer addRef() {
    return (ScalarScaleLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList inputData;
    private final Tensor weight;
    private final boolean frozen;
    private final UUID layerId;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final Tensor weight, final boolean frozen,
                final UUID layerId, final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.weight = weight;
      this.frozen = frozen;
      this.layerId = layerId;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      int batchSize = inputData.length();
      if (!frozen && buffer != null) {
        double grad = 0.0;
        for (int b = 0; b < batchSize; b++) {
          Tensor inTensor = inputData.get(b);
          Tensor dTensor = delta.get(b);
          double[] x = inTensor.getData();
          double[] dy = dTensor.getData();
          for (int i = 0; i < x.length; i++) {
            grad += dy[i] * x[i];
          }
          dTensor.freeRef();
          inTensor.freeRef();
        }
        Delta<UUID> deltaBuf = buffer.get(layerId, weight.getData());
        if (deltaBuf != null) {
          deltaBuf.addInPlace(new double[]{grad});
          deltaBuf.freeRef();
        }
      }
      if (alive) {
        double scale = weight.get(0);
        TensorArray passback = new TensorArray(RefIntStream.range(0, batchSize)
            .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
              Tensor dTensor = delta.get(b);
              Tensor dxTensor = new Tensor(dTensor.getDimensions());
              double[] dy = dTensor.getData();
              double[] dx = dxTensor.getData();
              for (int i = 0; i < dy.length; i++) {
                dx[i] = dy[i] * scale;
              }
              dTensor.freeRef();
              return dxTensor;
            }, delta.addRef()))
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
      weight.freeRef();
      upstream.freeRef();
    }
  }
}
