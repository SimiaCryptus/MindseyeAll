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
 * CUDA-accelerated tanh logit soft-capping activation layer: {@code cap * tanh(x / cap)} when {@code cap > 0}.
 */
@SuppressWarnings("serial")
public class SoftcapLayer extends LayerBase implements MultiPrecision {

  private final double cap;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  public SoftcapLayer(final double cap) {
    super();
    this.cap = cap;
    this.frozen = true;
  }

  protected SoftcapLayer(@Nonnull final JsonObject json) {
    super(json);
    this.cap = json.get("cap").getAsDouble();
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
    }
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SoftcapLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new SoftcapLayer(json);
  }

  public double getCap() {
    return cap;
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
    return new com.simiacryptus.mindseye.layers.java.SoftcapLayer(cap);
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
      throw new IllegalArgumentException("SoftcapLayer requires 1 input");
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
    final Precision prec = this.precision;

    CudaTensorList outputData = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inTensor = gpu.getTensor(inputData.addRef(), prec, MemoryType.Device, true);
      CudaMemory inMem = inTensor.getMemory(gpu.addRef());
      CudaMemory outMem = gpu.allocate((long) totalElements * prec.size, MemoryType.Device, true);
      GemmaCudaKernels.softcapForward(gpu.getDeviceId(), prec, inMem.getPtr(), outMem.getPtr(), cap, totalElements);
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
    Accumulator accumulator = new Accumulator(inputData, dims, batchSize, cap, prec, upstream, alive);
    return new Result(outputData, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("cap", cap);
    json.addProperty("precision", precision.name());
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
  public SoftcapLayer addRef() {
    return (SoftcapLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList inputData;
    private final int[] dims;
    private final int batchSize;
    private final double cap;
    private final Precision precision;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final int[] dims, final int batchSize, final double cap,
                final Precision precision, final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.dims = dims;
      this.batchSize = batchSize;
      this.cap = cap;
      this.precision = precision;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      if (alive) {
        final int totalElements = batchSize * Tensor.length(dims);
        CudaTensorList passback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor inTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
          CudaTensor dyTensor = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
          CudaMemory inMem = inTensor.getMemory(gpu.addRef());
          CudaMemory dyMem = dyTensor.getMemory(gpu.addRef());
          CudaMemory dxMem = gpu.allocate((long) totalElements * precision.size, MemoryType.Device, true);
          GemmaCudaKernels.softcapBackward(
              gpu.getDeviceId(), precision, inMem.getPtr(), dyMem.getPtr(), dxMem.getPtr(), cap, totalElements);
          inMem.dirty();
          dyMem.dirty();
          dxMem.dirty();
          int d2 = dims.length < 3 ? 1 : dims[2];
          int d1 = dims.length < 2 ? 1 : dims[1];
          int d0 = dims[0];
          CudaDevice.CudaTensorDescriptor dxDesc = gpu.newTensorDescriptor(
              precision, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
          inMem.freeRef();
          dyMem.freeRef();
          inTensor.freeRef();
          dyTensor.freeRef();
          gpu.freeRef();
          return new CudaTensorList(new CudaTensor(dxMem, dxDesc, precision), batchSize, dims, precision);
        }, inputData.addRef(), delta.addRef()), delta);
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
      inputData.freeRef();
      upstream.freeRef();
    }
  }
}
