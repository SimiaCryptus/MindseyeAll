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
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

/**
 * CUDA-accelerated chunk slicing layer extracting {@code [chunkIndex * chunkSize, (chunkIndex + 1) * chunkSize)}
 * along the leading dimension {@code totalChunks * chunkSize} across all sequence positions.
 */
@SuppressWarnings("serial")
public class ChunkSelectLayer extends LayerBase implements MultiPrecision {

  private final int chunkSize;
  private final int totalChunks;
  private final int chunkIndex;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  public ChunkSelectLayer(final int chunkSize, final int totalChunks, final int chunkIndex) {
    super();
    if (chunkSize <= 0 || totalChunks <= 0 || chunkIndex < 0 || chunkIndex >= totalChunks) {
      throw new IllegalArgumentException(
          "Invalid ChunkSelectLayer params: chunkSize=" + chunkSize + ", totalChunks=" + totalChunks + ", chunkIndex=" + chunkIndex);
    }
    this.chunkSize = chunkSize;
    this.totalChunks = totalChunks;
    this.chunkIndex = chunkIndex;
    this.frozen = true;
  }

  protected ChunkSelectLayer(@Nonnull final JsonObject json) {
    super(json);
    this.chunkSize = json.get("chunkSize").getAsInt();
    this.totalChunks = json.get("totalChunks").getAsInt();
    this.chunkIndex = json.get("chunkIndex").getAsInt();
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
    }
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ChunkSelectLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new ChunkSelectLayer(json);
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
    return new com.simiacryptus.mindseye.layers.java.ChunkSelectLayer(chunkSize, totalChunks, chunkIndex);
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
      throw new IllegalArgumentException("ChunkSelectLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] inDims = inputData.getDimensions();
    int expectedDim0 = chunkSize * totalChunks;
    if (inDims.length == 0 || inDims[0] != expectedDim0) {
      inputData.freeRef();
      in0.freeRef();
      throw new IllegalArgumentException("Expected leading dim " + expectedDim0 + ", got " + Arrays.toString(inDims));
    }
    if (inDims.length > 3) {
      inputData.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(in0);
      compatibilityLayer.freeRef();
      return result;
    }

    int[] outDims = inDims.clone();
    outDims[0] = chunkSize;
    int batchSize = inputData.length();
    int positionsPerSample = Tensor.length(inDims) / expectedDim0;
    int totalPositions = batchSize * positionsPerSample;
    final Precision prec = this.precision;

    CudaTensorList outputData = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inTensor = gpu.getTensor(inputData.addRef(), prec, MemoryType.Device, true);
      CudaMemory inMem = inTensor.getMemory(gpu.addRef());
      CudaMemory outMem = gpu.allocate((long) totalPositions * chunkSize * prec.size, MemoryType.Device, true);

      GemmaCudaKernels.chunkSelectForward(
          gpu.getDeviceId(), prec, inMem.getPtr(), outMem.getPtr(),
          totalPositions, chunkSize, totalChunks, chunkIndex);

      inMem.dirty();
      outMem.dirty();
      int d2 = outDims.length < 3 ? 1 : outDims[2];
      int d1 = outDims.length < 2 ? 1 : outDims[1];
      int d0 = outDims[0];
      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(
          prec, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
      inMem.freeRef();
      inTensor.freeRef();
      gpu.freeRef();
      return new CudaTensorList(new CudaTensor(outMem, outDesc, prec), batchSize, outDims, prec);
    }, inputData.addRef()), inputData);

    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(inDims, batchSize, totalPositions, chunkSize, totalChunks, chunkIndex, prec, upstream, alive);
    return new Result(outputData, accumulator, alive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("chunkSize", chunkSize);
    json.addProperty("totalChunks", totalChunks);
    json.addProperty("chunkIndex", chunkIndex);
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
  public ChunkSelectLayer addRef() {
    return (ChunkSelectLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final int[] inDims;
    private final int batchSize;
    private final int totalPositions;
    private final int chunkSize;
    private final int totalChunks;
    private final int chunkIndex;
    private final Precision precision;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final int[] inDims, final int batchSize, final int totalPositions,
                final int chunkSize, final int totalChunks, final int chunkIndex,
                final Precision precision, final Result.Accumulator upstream, final boolean alive) {
      this.inDims = inDims;
      this.batchSize = batchSize;
      this.totalPositions = totalPositions;
      this.chunkSize = chunkSize;
      this.totalChunks = totalChunks;
      this.chunkIndex = chunkIndex;
      this.precision = precision;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      if (alive) {
        final long inTotalElems = (long) totalPositions * chunkSize * totalChunks;
        CudaTensorList passback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor dyTensor = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
          CudaMemory dyMem = dyTensor.getMemory(gpu.addRef());
          CudaMemory dxMem = gpu.allocate(inTotalElems * precision.size, MemoryType.Device, false);

          GemmaCudaKernels.chunkSelectBackward(
              gpu.getDeviceId(), precision, dyMem.getPtr(), dxMem.getPtr(),
              totalPositions, chunkSize, totalChunks, chunkIndex);

          dyMem.dirty();
          dxMem.dirty();
          int d2 = inDims.length < 3 ? 1 : inDims[2];
          int d1 = inDims.length < 2 ? 1 : inDims[1];
          int d0 = inDims[0];
          CudaDevice.CudaTensorDescriptor dxDesc = gpu.newTensorDescriptor(
              precision, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
          dyMem.freeRef();
          dyTensor.freeRef();
          gpu.freeRef();
          return new CudaTensorList(new CudaTensor(dxMem, dxDesc, precision), batchSize, inDims, precision);
        }, delta.addRef()), delta);
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
