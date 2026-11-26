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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaDevice;
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudnnHandle;
import com.simiacryptus.mindseye.lang.cudnn.GemmaCudaKernels;
import com.simiacryptus.mindseye.lang.cudnn.MemoryType;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefFunction;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

/**
 * cuBLAS-accelerated bias-free sequence-aware linear projection layer ({@code y = W x})
 * mapping {@code [inputDim, ...]} to {@code [outputDim, ...]} with GGUF-compatible {@code [inputDim, outputDim]} storage.
 */
@SuppressWarnings("serial")
public class LinearProjectionLayer extends LayerBase implements MultiPrecision {

  private final int inputDim;
  private final int outputDim;
  @Nonnull
  private final Tensor weights;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

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
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
    }
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

  @Override
  public Precision getPrecision() {
    return precision;
  }

  @Override
  public void setPrecision(final Precision precision) {
    this.precision = precision;
  }

  @Nonnull
  public Layer getCompatibilityLayer() {
    com.simiacryptus.mindseye.layers.java.LinearProjectionLayer cpu =
        new com.simiacryptus.mindseye.layers.java.LinearProjectionLayer(inputDim, outputDim, weights.addRef());
    cpu.setFrozen(isFrozen());
    return cpu;
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (!CudaSystem.isEnabled()) {
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(inObj);
      compatibilityLayer.freeRef();
      return result;
    }
    if (inObj.length != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("LinearProjectionLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] inDims = inputData.getDimensions();
    if (inDims.length == 0 || inDims[0] != inputDim) {
      inputData.freeRef();
      in0.freeRef();
      throw new IllegalArgumentException("Expected leading dimension " + inputDim + ", got " + Arrays.toString(inDims));
    }
    if (inDims.length > 3) {
      inputData.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(in0);
      compatibilityLayer.freeRef();
      return result;
    }

    int[] outDims = inDims.clone();
    outDims[0] = outputDim;
    int batchSize = inputData.length();
    int positionsPerSample = Tensor.length(inDims) / inputDim;
    int totalPositions = batchSize * positionsPerSample;
    final Precision prec = this.precision;

    CudaTensorList outputData = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inTensor = gpu.getTensor(inputData.addRef(), prec, MemoryType.Device, true);
      CudaMemory inMem = inTensor.getMemory(gpu.addRef());
      CudaMemory wMem = gpu.allocate((long) inputDim * outputDim * prec.size, MemoryType.Device, true);
      wMem.write(prec, weights.addRef());
      CudaMemory outMem = gpu.allocate((long) totalPositions * outputDim * prec.size, MemoryType.Device, true);

      GemmaCudaKernels.linearForward(
          gpu.getDeviceId(), prec, inMem.getPtr(), wMem.getPtr(), outMem.getPtr(),
          totalPositions, inputDim, outputDim);

      inMem.dirty();
      wMem.dirty();
      outMem.dirty();
      int d2 = outDims.length < 3 ? 1 : outDims[2];
      int d1 = outDims.length < 2 ? 1 : outDims[1];
      int d0 = outDims[0];
      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(
          prec, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
      wMem.freeRef();
      inMem.freeRef();
      inTensor.freeRef();
      gpu.freeRef();
      return new CudaTensorList(new CudaTensor(outMem, outDesc, prec), batchSize, outDims, prec);
    }, inputData.addRef(), weights.addRef()), inputData.addRef());

    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(
        inputData, weights.addRef(), inDims, batchSize, inputDim, outputDim, isFrozen(), prec, upstream, alive);
    return new Result(outputData, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("inputDim", inputDim);
    json.addProperty("outputDim", outputDim);
    json.addProperty("precision", precision.name());
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
    private final int[] inDims;
    private final int batchSize;
    private final int inputDim;
    private final int outputDim;
    private final boolean frozen;
    private final Precision precision;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final Tensor weights, final int[] inDims, final int batchSize,
                final int inputDim, final int outputDim, final boolean frozen, final Precision precision,
                final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.weights = weights;
      this.inDims = inDims;
      this.batchSize = batchSize;
      this.inputDim = inputDim;
      this.outputDim = outputDim;
      this.frozen = frozen;
      this.precision = precision;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      final boolean calcDw = !frozen && buffer != null;
      if (alive || calcDw) {
        final double[] dwHost = calcDw ? new double[inputDim * outputDim] : null;
        final int positionsPerSample = Tensor.length(inDims) / inputDim;
        final int totalPositions = batchSize * positionsPerSample;

        CudaTensorList passback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor dyTensor = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
          CudaMemory dyMem = dyTensor.getMemory(gpu.addRef());

          if (calcDw) {
            CudaTensor inTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
            CudaMemory inMem = inTensor.getMemory(gpu.addRef());
            CudaMemory dwMem = gpu.allocate((long) inputDim * outputDim * precision.size, MemoryType.Device, true);
            GemmaCudaKernels.linearBackwardWeights(
                gpu.getDeviceId(), precision, inMem.getPtr(), dyMem.getPtr(), dwMem.getPtr(),
                totalPositions, inputDim, outputDim);
            inMem.dirty();
            dwMem.dirty();
            GemmaCudaKernels.readElemAsDoubles(precision, dwMem.getPtr(), dwHost);
            dwMem.freeRef();
            inMem.freeRef();
            inTensor.freeRef();
          }

          if (alive) {
            CudaMemory wMem = gpu.allocate((long) inputDim * outputDim * precision.size, MemoryType.Device, true);
            wMem.write(precision, weights.addRef());
            CudaMemory dxMem = gpu.allocate((long) totalPositions * inputDim * precision.size, MemoryType.Device, true);
            GemmaCudaKernels.linearBackwardInput(
                gpu.getDeviceId(), precision, dyMem.getPtr(), wMem.getPtr(), dxMem.getPtr(),
                totalPositions, inputDim, outputDim);
            wMem.dirty();
            dxMem.dirty();
            wMem.freeRef();
            dyMem.dirty();
            dyMem.freeRef();
            dyTensor.freeRef();
            int d2 = inDims.length < 3 ? 1 : inDims[2];
            int d1 = inDims.length < 2 ? 1 : inDims[1];
            int d0 = inDims[0];
            CudaDevice.CudaTensorDescriptor dxDesc = gpu.newTensorDescriptor(
                precision, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
            gpu.freeRef();
            return new CudaTensorList(new CudaTensor(dxMem, dxDesc, precision), batchSize, inDims, precision);
          } else {
            dyMem.dirty();
            dyMem.freeRef();
            dyTensor.freeRef();
            gpu.freeRef();
            return null;
          }
        }, inputData.addRef(), weights.addRef(), delta.addRef()), delta.addRef());

        if (calcDw && buffer != null) {
          Delta<UUID> deltaBuf = buffer.get(weights.getId(), weights.getData());
          if (deltaBuf != null) {
            deltaBuf.addInPlace(dwHost);
            deltaBuf.freeRef();
          }
        }
        if (alive && passback != null) {
          DeltaSet<UUID> bufferCopy = buffer == null ? null : buffer.addRef();
          upstream.accept(bufferCopy, passback);
        } else if (passback != null) {
          passback.freeRef();
        }
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
