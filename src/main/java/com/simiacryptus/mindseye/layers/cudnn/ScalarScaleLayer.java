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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

/**
 * CUDA-accelerated bias-free scalar multiplication layer ({@code y = scale * x}).
 */
@SuppressWarnings("serial")
public class ScalarScaleLayer extends LayerBase implements MultiPrecision {

  @Nonnull
  private final Tensor weight;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  public ScalarScaleLayer(final double scale) {
    this(scale, false);
  }

  public ScalarScaleLayer(final double scale, final boolean frozen) {
    super();
    this.weight = new Tensor(1);
    this.weight.set(0, scale);
    this.frozen = frozen;
  }

  protected ScalarScaleLayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    Tensor loaded = Tensor.fromJson(json.get("weight"), resources);
    if (loaded == null) {
      throw new IllegalArgumentException("Missing weight in ScalarScaleLayer JSON");
    }
    this.weight = loaded;
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
    }
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
    return new com.simiacryptus.mindseye.layers.java.ScalarScaleLayer(weight.addRef(), isFrozen());
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
      throw new IllegalArgumentException("ScalarScaleLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] dims = inputData.getDimensions();
    if (dims.length < 1 || dims.length > 3) {
      inputData.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(in0);
      compatibilityLayer.freeRef();
      return result;
    }

    int batchSize = inputData.length();
    int totalElements = batchSize * Tensor.length(dims);
    final double scale = weight.get(0);
    final Precision prec = this.precision;

    CudaTensorList outputData = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inTensor = gpu.getTensor(inputData.addRef(), prec, MemoryType.Device, true);
      CudaMemory inMem = inTensor.getMemory(gpu.addRef());
      CudaMemory outMem = gpu.allocate((long) totalElements * prec.size, MemoryType.Device, true);
      GemmaCudaKernels.scalarScaleForward(gpu.getDeviceId(), prec, inMem.getPtr(), outMem.getPtr(), scale, totalElements);
      inMem.dirty();
      outMem.dirty();
      int d2 = dims.length < 3 ? 1 : dims[2];
      int d1 = dims.length < 2 ? 1 : dims[1];
      int d0 = dims[0];
      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(
          prec, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
      inMem.freeRef();
      inTensor.freeRef();
      gpu.freeRef();
      return new CudaTensorList(new CudaTensor(outMem, outDesc, prec), batchSize, dims, prec);
    }, inputData.addRef()), inputData.addRef());

    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(
        inputData, weight.addRef(), dims, batchSize, isFrozen(), prec, getId(), upstream, alive);
    return new Result(outputData, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.add("weight", weight.getJson(resources, dataSerializer));
    json.addProperty("precision", precision.name());
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
    private final int[] dims;
    private final int batchSize;
    private final boolean frozen;
    private final Precision precision;
    private final UUID layerId;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final Tensor weight, final int[] dims,
                final int batchSize, final boolean frozen, final Precision precision,
                final UUID layerId, final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.weight = weight;
      this.dims = dims;
      this.batchSize = batchSize;
      this.frozen = frozen;
      this.precision = precision;
      this.layerId = layerId;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      final boolean calcDw = !frozen && buffer != null;
      if (alive || calcDw) {
        final double[] gradOut = calcDw ? new double[1] : null;
        final double scale = weight.get(0);
        final int totalElements = batchSize * Tensor.length(dims);

        CudaTensorList passback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor dyTensor = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
          CudaMemory dyMem = dyTensor.getMemory(gpu.addRef());
          if (calcDw) {
            CudaTensor inTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
            CudaMemory inMem = inTensor.getMemory(gpu.addRef());
            CudaMemory dotMem = gpu.allocate(Double.BYTES, MemoryType.Device, false);
            GemmaCudaKernels.dotReduceToDouble(
                gpu.getDeviceId(), precision, dyMem.getPtr(), inMem.getPtr(), dotMem.getPtr(), totalElements);
            inMem.dirty();
            dotMem.dirty();
            GemmaCudaKernels.readDoubles(dotMem.getPtr(), gradOut);
            dotMem.freeRef();
            inMem.freeRef();
            inTensor.freeRef();
          }
          if (alive) {
            CudaMemory dxMem = gpu.allocate((long) totalElements * precision.size, MemoryType.Device, true);
            GemmaCudaKernels.scalarScaleForward(
                gpu.getDeviceId(), precision, dyMem.getPtr(), dxMem.getPtr(), scale, totalElements);
            dxMem.dirty();
            dyMem.dirty();
            dyMem.freeRef();
            dyTensor.freeRef();
            int d2 = dims.length < 3 ? 1 : dims[2];
            int d1 = dims.length < 2 ? 1 : dims[1];
            int d0 = dims[0];
            CudaDevice.CudaTensorDescriptor dxDesc = gpu.newTensorDescriptor(
                precision, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
            gpu.freeRef();
            return new CudaTensorList(new CudaTensor(dxMem, dxDesc, precision), batchSize, dims, precision);
          } else {
            dyMem.dirty();
            dyMem.freeRef();
            dyTensor.freeRef();
            gpu.freeRef();
            return null;
          }
        }, inputData.addRef(), delta.addRef()), delta.addRef());

        if (calcDw && buffer != null) {
          Delta<UUID> deltaBuf = buffer.get(layerId, weight.getData());
          if (deltaBuf != null) {
            deltaBuf.addInPlace(gradOut);
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
      weight.freeRef();
      upstream.freeRef();
    }
  }
}
