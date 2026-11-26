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

import com.google.gson.JsonArray;
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
 * CUDA-accelerated NeoX-style Rotary Positional Embedding (RoPE) layer supporting both standard
 * {@code theta} and Gemma 4 per-frequency scaling divisors ({@code rope_freqs.weight}).
 */
@SuppressWarnings("serial")
public class RopeLayer extends LayerBase implements MultiPrecision {

  private final int numHeads;
  private final int headSize;
  private final double ropeTheta;
  @Nullable
  private final double[] freqFactors;
  private int startPosition;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

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
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
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
    return new com.simiacryptus.mindseye.layers.java.RopeLayer(numHeads, headSize, ropeTheta, freqFactors, startPosition);
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
      throw new IllegalArgumentException("RopeLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] dims = inputData.getDimensions();
    int elemLen = Tensor.length(dims);
    int totalHeadDim = numHeads * headSize;
    if (elemLen % totalHeadDim != 0) {
      inputData.freeRef();
      in0.freeRef();
      throw new IllegalArgumentException("Input length " + elemLen + " is not divisible by " + totalHeadDim);
    }
    if (dims.length < 1 || dims.length > 3) {
      inputData.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(in0);
      compatibilityLayer.freeRef();
      return result;
    }

    int batchSize = inputData.length();
    int seqLen = elemLen / totalHeadDim;
    final int capturedStart = this.startPosition;
    final double[] capturedFactors = this.freqFactors;
    final Precision prec = this.precision;

    CudaTensorList outputData = runRopeGpu(
        inputData, dims, batchSize, seqLen, numHeads, headSize, ropeTheta, capturedFactors, capturedStart, prec, false);

    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(
        dims, batchSize, seqLen, numHeads, headSize, ropeTheta, capturedFactors, capturedStart, prec, upstream, alive);
    return new Result(outputData, accumulator, alive);
  }

  @Nonnull
  static CudaTensorList runRopeGpu(final TensorList srcList, final int[] dims, final int batchSize,
                                   final int seqLen, final int numHeads, final int headSize,
                                   final double ropeTheta, @Nullable final double[] factors,
                                   final int startPos, final Precision prec, final boolean inverse) {
    final int elemLen = seqLen * numHeads * headSize;
    return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor srcTensor = gpu.getTensor(srcList.addRef(), prec, MemoryType.Device, true);
      CudaMemory srcMem = srcTensor.getMemory(gpu.addRef());
      CudaMemory dstMem = gpu.allocate((long) batchSize * elemLen * prec.size, MemoryType.Device, true);
      final CudaMemory fMem = factors != null
          ? gpu.allocate((long) factors.length * Double.BYTES, MemoryType.Device, true)
          : null;
      if (fMem != null) {
        GemmaCudaKernels.writeDoubles(fMem.getPtr(), factors);
      }

      GemmaCudaKernels.ropeApply(
          gpu.getDeviceId(), prec, srcMem.getPtr(), dstMem.getPtr(), fMem == null ? null : fMem.getPtr(),
          batchSize, seqLen, numHeads, headSize, ropeTheta, startPos, inverse);

      srcMem.dirty();
      dstMem.dirty();
      if (fMem != null) {
        fMem.dirty();
        fMem.freeRef();
      }
      int d2 = dims.length < 3 ? 1 : dims[2];
      int d1 = dims.length < 2 ? 1 : dims[1];
      int d0 = dims[0];
      CudaDevice.CudaTensorDescriptor dstDesc = gpu.newTensorDescriptor(
          prec, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
      srcMem.freeRef();
      srcTensor.freeRef();
      gpu.freeRef();
      return new CudaTensorList(new CudaTensor(dstMem, dstDesc, prec), batchSize, dims, prec);
    }, srcList.addRef()), srcList);
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("numHeads", numHeads);
    json.addProperty("headSize", headSize);
    json.addProperty("ropeTheta", ropeTheta);
    json.addProperty("startPosition", startPosition);
    json.addProperty("precision", precision.name());
    if (freqFactors != null) {
      JsonArray arr = new JsonArray();
      for (int i = 0; i < freqFactors.length; i++) {
        arr.add(freqFactors[i]);
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
    private final int[] dims;
    private final int batchSize;
    private final int seqLen;
    private final int numHeads;
    private final int headSize;
    private final double ropeTheta;
    @Nullable
    private final double[] freqFactors;
    private final int startPos;
    private final Precision precision;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final int[] dims, final int batchSize, final int seqLen, final int numHeads, final int headSize,
                final double ropeTheta, @Nullable final double[] freqFactors, final int startPos,
                final Precision precision, final Result.Accumulator upstream, final boolean alive) {
      this.dims = dims;
      this.batchSize = batchSize;
      this.seqLen = seqLen;
      this.numHeads = numHeads;
      this.headSize = headSize;
      this.ropeTheta = ropeTheta;
      this.freqFactors = freqFactors;
      this.startPos = startPos;
      this.precision = precision;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      if (alive) {
        CudaTensorList passback = runRopeGpu(
            delta, dims, batchSize, seqLen, numHeads, headSize, ropeTheta, freqFactors, startPos, precision, true);
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
