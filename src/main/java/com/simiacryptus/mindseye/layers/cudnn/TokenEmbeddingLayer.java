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
import java.util.Map;
import java.util.UUID;

/**
 * CUDA-accelerated token embedding lookup layer mapping integer token IDs {@code Tensor(seqLen)}
 * to {@code [embeddingDim, seqLen]} on GPU with optional scale factor {@code scale}.
 */
@SuppressWarnings("serial")
public class TokenEmbeddingLayer extends LayerBase implements MultiPrecision {

  private final int vocabSize;
  private final int embeddingDim;
  private final double scale;
  @Nonnull
  private final Tensor weights;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  public TokenEmbeddingLayer(final int vocabSize, final int embeddingDim) {
    this(vocabSize, embeddingDim, 1.0);
  }

  public TokenEmbeddingLayer(final int vocabSize, final int embeddingDim, final double scale) {
    super();
    if (vocabSize <= 0 || embeddingDim <= 0) {
      throw new IllegalArgumentException("Invalid embedding dimensions: " + vocabSize + " x " + embeddingDim);
    }
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.scale = scale;
    this.weights = new Tensor(embeddingDim, vocabSize);
    double std = 1.0 / Math.sqrt(embeddingDim);
    double[] w = this.weights.getData();
    for (int i = 0; i < w.length; i++) {
      w[i] = (1.0 - 2.0 * Util.R.get().nextDouble()) * std;
    }
  }

  public TokenEmbeddingLayer(final int vocabSize, final int embeddingDim, final double scale,
                             @Nonnull final Tensor weights) {
    super();
    if (weights.length() != vocabSize * embeddingDim) {
      weights.freeRef();
      throw new IllegalArgumentException("Embedding table size mismatch");
    }
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    this.scale = scale;
    this.weights = weights;
  }

  protected TokenEmbeddingLayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    this.vocabSize = json.get("vocabSize").getAsInt();
    this.embeddingDim = json.get("embeddingDim").getAsInt();
    this.scale = json.get("scale").getAsDouble();
    Tensor loaded = Tensor.fromJson(json.get("weights"), resources);
    if (loaded == null) {
      throw new IllegalArgumentException("Missing weights in TokenEmbeddingLayer JSON");
    }
    this.weights = loaded;
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
    }
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static TokenEmbeddingLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new TokenEmbeddingLayer(json, rs);
  }

  public int getVocabSize() {
    return vocabSize;
  }

  public int getEmbeddingDim() {
    return embeddingDim;
  }

  public double getScale() {
    return scale;
  }

  @Nonnull
  public Tensor getWeights() {
    return weights.addRef();
  }

  public void setWeights(@Nonnull final double[] values) {
    if (values.length != vocabSize * embeddingDim) {
      throw new IllegalArgumentException("Expected " + (vocabSize * embeddingDim) + " values, got " + values.length);
    }
    System.arraycopy(values, 0, weights.getData(), 0, values.length);
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
    com.simiacryptus.mindseye.layers.java.TokenEmbeddingLayer cpu =
        new com.simiacryptus.mindseye.layers.java.TokenEmbeddingLayer(vocabSize, embeddingDim, scale, weights.addRef());
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
      throw new IllegalArgumentException("TokenEmbeddingLayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] inDims = inputData.getDimensions();
    if (inDims.length < 1 || inDims.length > 3) {
      inputData.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(in0);
      compatibilityLayer.freeRef();
      return result;
    }

    int seqLen = Tensor.length(inDims);
    int[] outDims = new int[]{embeddingDim, seqLen};
    int batchSize = inputData.length();
    int totalTokens = batchSize * seqLen;
    final Precision prec = this.precision;
    final int[] errHolder = new int[1];

    CudaTensorList outputData = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inTensor = gpu.getTensor(inputData.addRef(), prec, MemoryType.Device, true);
      CudaMemory inMem = inTensor.getMemory(gpu.addRef());
      CudaMemory wMem = gpu.allocate((long) vocabSize * embeddingDim * prec.size, MemoryType.Device, true);
      wMem.write(prec, weights.addRef());
      CudaMemory outMem = gpu.allocate((long) totalTokens * embeddingDim * prec.size, MemoryType.Device, true);
      CudaMemory errMem = gpu.allocate(Integer.BYTES, MemoryType.Device, false);

      GemmaCudaKernels.tokenEmbeddingForward(
          gpu.getDeviceId(), prec, inMem.getPtr(), wMem.getPtr(), outMem.getPtr(), errMem.getPtr(),
          totalTokens, vocabSize, embeddingDim, scale);

      inMem.dirty();
      wMem.dirty();
      outMem.dirty();
      errMem.dirty();
      GemmaCudaKernels.readInts(errMem.getPtr(), errHolder);
      errMem.freeRef();
      wMem.freeRef();
      inMem.freeRef();
      inTensor.freeRef();

      if (errHolder[0] != 0) {
        outMem.freeRef();
        gpu.freeRef();
        throw new IndexOutOfBoundsException("Token index out of range [0, " + vocabSize + "): " + errHolder[0]);
      }

      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(
          prec, batchSize, 1, seqLen, embeddingDim, seqLen * embeddingDim, embeddingDim, embeddingDim, 1);
      gpu.freeRef();
      return new CudaTensorList(new CudaTensor(outMem, outDesc, prec), batchSize, outDims, prec);
    }, inputData.addRef(), weights.addRef()), inputData.addRef());

    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(
        inputData, weights.addRef(), inDims, batchSize, vocabSize, embeddingDim, scale, isFrozen(), prec, upstream, alive);
    return new Result(outputData, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("vocabSize", vocabSize);
    json.addProperty("embeddingDim", embeddingDim);
    json.addProperty("scale", scale);
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
  public TokenEmbeddingLayer addRef() {
    return (TokenEmbeddingLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList inputData;
    private final Tensor weights;
    private final int[] inDims;
    private final int batchSize;
    private final int vocabSize;
    private final int embeddingDim;
    private final double scale;
    private final boolean frozen;
    private final Precision precision;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final Tensor weights, final int[] inDims, final int batchSize,
                final int vocabSize, final int embeddingDim, final double scale, final boolean frozen,
                final Precision precision, final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.weights = weights;
      this.inDims = inDims;
      this.batchSize = batchSize;
      this.vocabSize = vocabSize;
      this.embeddingDim = embeddingDim;
      this.scale = scale;
      this.frozen = frozen;
      this.precision = precision;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      final boolean calcDw = !frozen && buffer != null;
      if (alive || calcDw) {
        final double[] dTableHost = calcDw ? new double[vocabSize * embeddingDim] : null;
        final int seqLen = Tensor.length(inDims);
        final int totalTokens = batchSize * seqLen;

        CudaTensorList zeroPassback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          if (calcDw) {
            CudaTensor inTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
            CudaTensor dyTensor = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
            CudaMemory inMem = inTensor.getMemory(gpu.addRef());
            CudaMemory dyMem = dyTensor.getMemory(gpu.addRef());
            CudaMemory dTableMem = gpu.allocate((long) vocabSize * embeddingDim * Double.BYTES, MemoryType.Device, false);

            GemmaCudaKernels.tokenEmbeddingBackward(
                gpu.getDeviceId(), precision, inMem.getPtr(), dyMem.getPtr(), dTableMem.getPtr(),
                totalTokens, vocabSize, embeddingDim, scale);

            inMem.dirty();
            dyMem.dirty();
            dTableMem.dirty();
            GemmaCudaKernels.readDoubles(dTableMem.getPtr(), dTableHost);
            dTableMem.freeRef();
            inMem.freeRef();
            dyMem.freeRef();
            inTensor.freeRef();
            dyTensor.freeRef();
          }

          if (alive) {
            CudaMemory zeroMem = gpu.allocate((long) totalTokens * precision.size, MemoryType.Device, false);
            zeroMem.dirty();
            int d2 = inDims.length < 3 ? 1 : inDims[2];
            int d1 = inDims.length < 2 ? 1 : inDims[1];
            int d0 = inDims[0];
            CudaDevice.CudaTensorDescriptor zDesc = gpu.newTensorDescriptor(
                precision, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
            gpu.freeRef();
            return new CudaTensorList(new CudaTensor(zeroMem, zDesc, precision), batchSize, inDims, precision);
          } else {
            gpu.freeRef();
            return null;
          }
        }, inputData.addRef(), delta.addRef()), delta.addRef());

        if (calcDw && buffer != null) {
          Delta<UUID> deltaBuf = buffer.get(weights.getId(), weights.getData());
          if (deltaBuf != null) {
            deltaBuf.addInPlace(dTableHost);
            deltaBuf.freeRef();
          }
        }
        if (alive && zeroPassback != null) {
          DeltaSet<UUID> bufferCopy = buffer == null ? null : buffer.addRef();
          upstream.accept(bufferCopy, zeroPassback);
        } else if (zeroPassback != null) {
          zeroPassback.freeRef();
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
