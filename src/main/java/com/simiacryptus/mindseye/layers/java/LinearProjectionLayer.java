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
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**
 * Bias-free sequence-aware linear projection layer ({@code y = W x}) mapping {@code [inputDim, ...]} to {@code [outputDim, ...]}.
 * Weight layout {@code [inputDim, outputDim]} stores row {@code outIdx} contiguously at {@code outIdx * inputDim + inIdx},
 * matching both GGUF weight matrices and {@link FullyConnectedLayer}.
 */
@SuppressWarnings("serial")
public class LinearProjectionLayer extends LayerBase {

  private final int inputDim;
  private final int outputDim;
  @Nonnull
  private final Tensor weights;

  public LinearProjectionLayer(final int inputDim, final int outputDim) {
    super();
    if (inputDim <= 0 || outputDim <= 0) {
      throw new IllegalArgumentException("Dimensions must be positive: " + inputDim + " -> " + outputDim);
    }
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.weights = new Tensor(inputDim, outputDim);
    double scale = Math.sqrt(6.0 / (inputDim + outputDim + 1.0));
    double[] w = this.weights.getData();
    for (int i = 0; i < w.length; i++) {
      w[i] = (1.0 - 2.0 * Util.R.get().nextDouble()) * scale;
    }
  }

  public LinearProjectionLayer(final int inputDim, final int outputDim, @Nonnull final Tensor weights) {
    super();
    int expected = inputDim * outputDim;
    if (weights.length() != expected) {
      weights.freeRef();
      throw new IllegalArgumentException("Weight length mismatch: expected " + expected);
    }
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.weights = weights;
  }

  protected LinearProjectionLayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    this.inputDim = json.get("inputDim").getAsInt();
    this.outputDim = json.get("outputDim").getAsInt();
    Tensor loaded = Tensor.fromJson(json.get("weights"), resources);
    if (loaded == null) {
      throw new IllegalArgumentException("Missing weights in LinearProjectionLayer JSON");
    }
    this.weights = loaded;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static LinearProjectionLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new LinearProjectionLayer(json, rs);
  }

  public int getInputDim() {
    return inputDim;
  }

  public int getOutputDim() {
    return outputDim;
  }

  @Nonnull
  public Tensor getWeights() {
    return weights.addRef();
  }

  public void setWeights(@Nonnull final double[] data) {
    if (data.length != inputDim * outputDim) {
      throw new IllegalArgumentException("Expected " + (inputDim * outputDim) + " weights, got " + data.length);
    }
    System.arraycopy(data, 0, weights.getData(), 0, data.length);
  }

  @RefIgnore
  @Nonnull
  static int[] computeOutputDims(@Nonnull final int[] inDims, final int inputDim, final int outputDim) {
    if (inDims.length == 0 || inDims[0] != inputDim) {
      throw new IllegalArgumentException("Expected leading dimension " + inputDim + ", got " + Arrays.toString(inDims));
    }
    int[] outDims = inDims.clone();
    outDims[0] = outputDim;
    return outDims;
  }

  @RefIgnore
  static void forwardPositions(@Nonnull final double[] in, @Nonnull final double[] w,
                               @Nonnull final double[] out, final int inputDim, final int outputDim) {
    int positions = in.length / inputDim;
    if (positions * outputDim >= 512) {
      IntStream.range(0, positions).parallel().forEach(p -> {
        int inBase = p * inputDim;
        int outBase = p * outputDim;
        for (int o = 0; o < outputDim; o++) {
          int wBase = o * inputDim;
          double sum = 0.0;
          for (int i = 0; i < inputDim; i++) {
            sum += w[wBase + i] * in[inBase + i];
          }
          out[outBase + o] = sum;
        }
      });
    } else {
      for (int p = 0; p < positions; p++) {
        int inBase = p * inputDim;
        int outBase = p * outputDim;
        for (int o = 0; o < outputDim; o++) {
          int wBase = o * inputDim;
          double sum = 0.0;
          for (int i = 0; i < inputDim; i++) {
            sum += w[wBase + i] * in[inBase + i];
          }
          out[outBase + o] = sum;
        }
      }
    }
  }

  @RefIgnore
  static void backwardWeights(@Nonnull final double[] in, @Nonnull final double[] dy,
                              @Nonnull final double[] dw, final int inputDim, final int outputDim) {
    int positions = in.length / inputDim;
    for (int p = 0; p < positions; p++) {
      int inBase = p * inputDim;
      int outBase = p * outputDim;
      for (int o = 0; o < outputDim; o++) {
        double dyo = dy[outBase + o];
        if (dyo != 0.0) {
          int wBase = o * inputDim;
          for (int i = 0; i < inputDim; i++) {
            dw[wBase + i] += dyo * in[inBase + i];
          }
        }
      }
    }
  }

  @RefIgnore
  static void backwardInput(@Nonnull final double[] dy, @Nonnull final double[] w,
                            @Nonnull final double[] dx, final int inputDim, final int outputDim) {
    int positions = dy.length / outputDim;
    for (int p = 0; p < positions; p++) {
      int inBase = p * inputDim;
      int outBase = p * outputDim;
      for (int o = 0; o < outputDim; o++) {
        double dyo = dy[outBase + o];
        if (dyo != 0.0) {
          int wBase = o * inputDim;
          for (int i = 0; i < inputDim; i++) {
            dx[inBase + i] += dyo * w[wBase + i];
          }
        }
      }
    }
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (inObj.length != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("LinearProjectionLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] outDims = computeOutputDims(inputData.getDimensions(), inputDim, outputDim);
    int batchSize = inputData.length();
    double[] wData = weights.getData();
    TensorArray outputData = new TensorArray(RefIntStream.range(0, batchSize)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
          Tensor inTensor = inputData.get(b);
          Tensor outTensor = new Tensor(outDims);
          forwardPositions(inTensor.getData(), wData, outTensor.getData(), inputDim, outputDim);
          inTensor.freeRef();
          return outTensor;
        }, inputData.addRef()))
        .toArray(Tensor[]::new));
    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(inputData, weights.addRef(), inputDim, outputDim, isFrozen(), upstream, alive);
    return new Result(outputData, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("inputDim", inputDim);
    json.addProperty("outputDim", outputDim);
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
  public LinearProjectionLayer addRef() {
    return (LinearProjectionLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList inputData;
    private final Tensor weights;
    private final int inputDim;
    private final int outputDim;
    private final boolean frozen;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final Tensor weights, final int inputDim, final int outputDim,
                final boolean frozen, final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.weights = weights;
      this.inputDim = inputDim;
      this.outputDim = outputDim;
      this.frozen = frozen;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      int batchSize = inputData.length();
      if (!frozen && buffer != null) {
        double[] dw = new double[inputDim * outputDim];
        for (int b = 0; b < batchSize; b++) {
          Tensor inTensor = inputData.get(b);
          Tensor dTensor = delta.get(b);
          backwardWeights(inTensor.getData(), dTensor.getData(), dw, inputDim, outputDim);
          dTensor.freeRef();
          inTensor.freeRef();
        }
        Delta<UUID> deltaBuf = buffer.get(weights.getId(), weights.getData());
        if (deltaBuf != null) {
          deltaBuf.addInPlace(dw);
          deltaBuf.freeRef();
        }
      }
      if (alive) {
        double[] wData = weights.getData();
        int[] inDims = inputData.getDimensions();
        TensorArray passback = new TensorArray(RefIntStream.range(0, batchSize)
            .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
              Tensor dTensor = delta.get(b);
              Tensor dxTensor = new Tensor(inDims);
              backwardInput(dTensor.getData(), wData, dxTensor.getData(), inputDim, outputDim);
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
      weights.freeRef();
      upstream.freeRef();
    }
  }
}
