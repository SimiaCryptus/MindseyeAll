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
import java.nio.FloatBuffer;
import java.util.Map;
import java.util.UUID;

/**
 * CUDA-accelerated Root Mean Square Layer Normalization (RMSNorm) layer supporting both
 * learned scale weights and bare unit-RMS normalization ({@code hasWeights == false}).
 */
@SuppressWarnings("serial")
public class RmsNormLayer extends LayerBase implements MultiPrecision {

  private final int normSize;
  private final double eps;
  private final boolean hasWeights;
  @Nullable
  private final Tensor weights;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

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

  protected RmsNormLayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    this.normSize = json.get("normSize").getAsInt();
    this.eps = json.get("eps").getAsDouble();
    this.hasWeights = json.get("hasWeights").getAsBoolean();
    this.weights = hasWeights ? Tensor.fromJson(json.get("weights"), resources) : null;
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
    }
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
    System.arraycopy(values, 0, weights.getData(), 0, normSize);
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
    com.simiacryptus.mindseye.layers.java.RmsNormLayer cpu =
        new com.simiacryptus.mindseye.layers.java.RmsNormLayer(
            normSize, eps, hasWeights, weights == null ? null : weights.addRef());
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
      throw new IllegalArgumentException("RmsNormLayer requires exactly 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] dims = inputData.getDimensions();
    int elemLen = Tensor.length(dims);
    if (elemLen % normSize != 0) {
      inputData.freeRef();
      in0.freeRef();
      throw new IllegalArgumentException("Input length " + elemLen + " is not divisible by normSize " + normSize);
    }
    if (dims.length < 1 || dims.length > 3) {
      inputData.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(in0);
      compatibilityLayer.freeRef();
      return result;
    }

    int batchSize = inputData.length();
    int totalChunks = batchSize * (elemLen / normSize);
    final Precision prec = this.precision;
    final double[] hostWeights = (hasWeights && weights != null) ? weights.getData() : null;

    CudaTensorList outputData = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inTensor = gpu.getTensor(inputData.addRef(), prec, MemoryType.Device, true);
      CudaMemory inMem = inTensor.getMemory(gpu.addRef());
      CudaMemory outMem = gpu.allocate((long) batchSize * elemLen * prec.size, MemoryType.Device, true);
      final CudaMemory wMem = hostWeights != null
          ? gpu.allocate((long) normSize * Double.BYTES, MemoryType.Device, true)
          : null;
      if (wMem != null) {
        GemmaCudaKernels.writeDoubles(wMem.getPtr(), hostWeights);
      }
      GemmaCudaKernels.rmsNormForward(
          gpu.getDeviceId(), prec,
          inMem.getPtr(), wMem == null ? null : wMem.getPtr(), outMem.getPtr(),
          totalChunks, normSize, eps, hostWeights != null);
      inMem.dirty();
      outMem.dirty();
      if (wMem != null) {
        wMem.dirty();
        wMem.freeRef();
      }
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
    boolean trainable = hasWeights && !isFrozen();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Tensor weightsRef = weights == null ? null : weights.addRef();
    Accumulator accumulator = new Accumulator(
        inputData, weightsRef, dims, batchSize, normSize, eps, hasWeights, isFrozen(), prec, getId(), upstream, alive);
    return new Result(outputData, accumulator, alive || trainable);
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("normSize", normSize);
    json.addProperty("eps", eps);
    json.addProperty("hasWeights", hasWeights);
    json.addProperty("precision", precision.name());
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
    private final int[] dims;
    private final int batchSize;
    private final int normSize;
    private final double eps;
    private final boolean hasWeights;
    private final boolean frozen;
    private final Precision precision;
    private final UUID layerId;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, @Nullable final Tensor weights, final int[] dims,
                final int batchSize, final int normSize, final double eps, final boolean hasWeights,
                final boolean frozen, final Precision precision, final UUID layerId,
                final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.weights = weights;
      this.dims = dims;
      this.batchSize = batchSize;
      this.normSize = normSize;
      this.eps = eps;
      this.hasWeights = hasWeights;
      this.frozen = frozen;
      this.precision = precision;
      this.layerId = layerId;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      final boolean calcDw = hasWeights && !frozen && weights != null && buffer != null;
      if (alive || calcDw) {
        final double[] wGradOut = calcDw ? new double[normSize] : null;
        final double[] hostWeights = (hasWeights && weights != null) ? weights.getData() : null;
        final int elemLen = Tensor.length(dims);
        final int totalChunks = batchSize * (elemLen / normSize);

        CudaTensorList passback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor inTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
          CudaTensor dyTensor = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
          CudaMemory inMem = inTensor.getMemory(gpu.addRef());
          CudaMemory dyMem = dyTensor.getMemory(gpu.addRef());
          final CudaMemory wMem = hostWeights != null
              ? gpu.allocate((long) normSize * Double.BYTES, MemoryType.Device, true)
              : null;
          if (wMem != null) {
            GemmaCudaKernels.writeDoubles(wMem.getPtr(), hostWeights);
          }
          CudaMemory dxMem = alive
              ? gpu.allocate((long) batchSize * elemLen * precision.size, MemoryType.Device, true)
              : null;
          CudaMemory dwMem = calcDw
              ? gpu.allocate((long) normSize * Double.BYTES, MemoryType.Device, false)
              : null;

          GemmaCudaKernels.rmsNormBackward(
              gpu.getDeviceId(), precision,
              inMem.getPtr(), dyMem.getPtr(),
              wMem == null ? null : wMem.getPtr(),
              dxMem == null ? null : dxMem.getPtr(),
              dwMem == null ? null : dwMem.getPtr(),
              totalChunks, normSize, eps, hostWeights != null);

          inMem.dirty();
          dyMem.dirty();
          if (wMem != null) {
            wMem.dirty();
            wMem.freeRef();
          }
          if (dwMem != null) {
            dwMem.dirty();
            GemmaCudaKernels.readDoubles(dwMem.getPtr(), wGradOut);
            dwMem.freeRef();
          }
          inMem.freeRef();
          dyMem.freeRef();
          inTensor.freeRef();
          dyTensor.freeRef();

          if (dxMem != null) {
            dxMem.dirty();
            int d2 = dims.length < 3 ? 1 : dims[2];
            int d1 = dims.length < 2 ? 1 : dims[1];
            int d0 = dims[0];
            CudaDevice.CudaTensorDescriptor dxDesc = gpu.newTensorDescriptor(
                precision, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
            gpu.freeRef();
            return new CudaTensorList(new CudaTensor(dxMem, dxDesc, precision), batchSize, dims, precision);
          } else {
            gpu.freeRef();
            return null;
          }
        }, inputData.addRef(), delta.addRef()), delta.addRef());

        if (calcDw && weights != null && buffer != null) {
          Delta<UUID> deltaBuf = buffer.get(layerId, weights.getData());
          if (deltaBuf != null) {
            deltaBuf.addInPlace(wGradOut);
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
      if (weights != null) {
        weights.freeRef();
      }
      upstream.freeRef();
    }
  }
}
