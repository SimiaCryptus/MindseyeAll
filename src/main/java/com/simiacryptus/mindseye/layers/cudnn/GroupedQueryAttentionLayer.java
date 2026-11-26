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
 * CUDA-accelerated Grouped-Query Attention (GQA) layer using an online-softmax FlashAttention-style
 * GPU kernel supporting causal masking, Gemma 4 sliding-window attention (SWA), shared KV layers,
 * {@code V = K} graphs, and both {@link Precision#Float} and {@link Precision#Double}.
 */
@SuppressWarnings("serial")
public class GroupedQueryAttentionLayer extends LayerBase implements MultiPrecision {

  private final int numHeads;
  private final int numKvHeads;
  private final int headSize;
  private final double scale;
  private final boolean causal;
  private final int slidingWindow;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  public GroupedQueryAttentionLayer(final int numHeads, final int numKvHeads, final int headSize,
                                    final double scale, final boolean causal, final int slidingWindow) {
    super();
    if (numHeads <= 0 || numKvHeads <= 0 || headSize <= 0 || numHeads % numKvHeads != 0) {
      throw new IllegalArgumentException(
          "Invalid GQA config: heads=" + numHeads + ", kvHeads=" + numKvHeads + ", headSize=" + headSize);
    }
    this.numHeads = numHeads;
    this.numKvHeads = numKvHeads;
    this.headSize = headSize;
    this.scale = scale;
    this.causal = causal;
    this.slidingWindow = slidingWindow;
    this.frozen = true;
  }

  protected GroupedQueryAttentionLayer(@Nonnull final JsonObject json) {
    super(json);
    this.numHeads = json.get("numHeads").getAsInt();
    this.numKvHeads = json.get("numKvHeads").getAsInt();
    this.headSize = json.get("headSize").getAsInt();
    this.scale = json.get("scale").getAsDouble();
    this.causal = json.get("causal").getAsBoolean();
    this.slidingWindow = json.get("slidingWindow").getAsInt();
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
    }
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static GroupedQueryAttentionLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new GroupedQueryAttentionLayer(json);
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
    return new com.simiacryptus.mindseye.layers.java.GroupedQueryAttentionLayer(
        numHeads, numKvHeads, headSize, scale, causal, slidingWindow);
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
    if (inObj.length != 3) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("GroupedQueryAttentionLayer requires 3 inputs (Q, K, V)");
    }
    Result qRes = inObj[0].addRef();
    Result kRes = inObj[1].addRef();
    Result vRes = inObj[2].addRef();
    RefUtil.freeRef(inObj);

    TensorList qList = qRes.getData();
    TensorList kList = kRes.getData();
    TensorList vList = vRes.getData();
    int[] qDims = qList.getDimensions();
    int[] kDims = kList.getDimensions();
    int[] vDims = vList.getDimensions();

    int qElem = Tensor.length(qDims);
    int kElem = Tensor.length(kDims);
    int vElem = Tensor.length(vDims);
    int qDim = numHeads * headSize;
    int kvDim = numKvHeads * headSize;
    if (qElem % qDim != 0 || kElem % kvDim != 0 || vElem != kElem) {
      qList.freeRef();
      kList.freeRef();
      vList.freeRef();
      qRes.freeRef();
      kRes.freeRef();
      vRes.freeRef();
      throw new IllegalArgumentException("Incompatible Q/K/V lengths: " + qElem + ", " + kElem + ", " + vElem);
    }
    if (qDims.length > 3 || kDims.length > 3 || vDims.length > 3) {
      qList.freeRef();
      kList.freeRef();
      vList.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(qRes, kRes, vRes);
      compatibilityLayer.freeRef();
      return result;
    }

    int batchSize = qList.length();
    int kBatch = kList.length();
    int vBatch = vList.length();
    int qSeqLen = qElem / qDim;
    int kvSeqLen = kElem / kvDim;
    final Precision prec = this.precision;

    boolean qAlive = qRes.isAlive();
    boolean kAlive = kRes.isAlive();
    boolean vAlive = vRes.isAlive();
    boolean anyAlive = qAlive || kAlive || vAlive;

    CudaTensor outTensor = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensor>) gpu -> {
      CudaTensor qGpu = gpu.getTensor(qList.addRef(), prec, MemoryType.Device, true);
      CudaTensor kGpu = gpu.getTensor(kList.addRef(), prec, MemoryType.Device, true);
      CudaTensor vGpu = gpu.getTensor(vList.addRef(), prec, MemoryType.Device, true);
      CudaMemory qMem = qGpu.getMemory(gpu.addRef());
      CudaMemory kMem = kGpu.getMemory(gpu.addRef());
      CudaMemory vMem = vGpu.getMemory(gpu.addRef());

      CudaMemory outMem = gpu.allocate((long) batchSize * qElem * prec.size, MemoryType.Device, true);

      GemmaCudaKernels.gqaForward(
          gpu.getDeviceId(), prec,
          qMem.getPtr(), kMem.getPtr(), vMem.getPtr(),
          outMem.getPtr(), null,
          batchSize, kBatch, vBatch, qSeqLen, kvSeqLen,
          numHeads, numKvHeads, headSize, scale, causal, slidingWindow);

      qMem.dirty();
      kMem.dirty();
      vMem.dirty();
      outMem.dirty();

      int d2 = qDims.length < 3 ? 1 : qDims[2];
      int d1 = qDims.length < 2 ? 1 : qDims[1];
      int d0 = qDims[0];
      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(
          prec, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);

      qMem.freeRef();
      kMem.freeRef();
      vMem.freeRef();
      qGpu.freeRef();
      kGpu.freeRef();
      vGpu.freeRef();
      gpu.freeRef();
      return new CudaTensor(outMem, outDesc, prec);
    }, qList.addRef(), kList.addRef(), vList.addRef()), qList.addRef());

    CudaTensorList outputData = new CudaTensorList(outTensor.addRef(), batchSize, qDims, prec);

    Accumulator accumulator = new Accumulator(
        qList, kList, vList, outTensor,
        qDims, kDims, vDims, batchSize, kBatch, vBatch, qSeqLen, kvSeqLen,
        qRes.getAccumulator(), kRes.getAccumulator(), vRes.getAccumulator(),
        qAlive, kAlive, vAlive,
        numHeads, numKvHeads, headSize, scale, causal, slidingWindow, prec);
    qRes.freeRef();
    kRes.freeRef();
    vRes.freeRef();
    return new Result(outputData, accumulator, anyAlive);
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("numHeads", numHeads);
    json.addProperty("numKvHeads", numKvHeads);
    json.addProperty("headSize", headSize);
    json.addProperty("scale", scale);
    json.addProperty("causal", causal);
    json.addProperty("slidingWindow", slidingWindow);
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
  public GroupedQueryAttentionLayer addRef() {
    return (GroupedQueryAttentionLayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList qList;
    private final TensorList kList;
    private final TensorList vList;
    private final CudaTensor outTensor;
    private final int[] qDims;
    private final int[] kDims;
    private final int[] vDims;
    private final int batchSize;
    private final int kBatch;
    private final int vBatch;
    private final int qSeqLen;
    private final int kvSeqLen;
    private final Result.Accumulator qAcc;
    private final Result.Accumulator kAcc;
    private final Result.Accumulator vAcc;
    private final boolean qAlive;
    private final boolean kAlive;
    private final boolean vAlive;
    private final int numHeads;
    private final int numKvHeads;
    private final int headSize;
    private final double scale;
    private final boolean causal;
    private final int slidingWindow;
    private final Precision precision;

    Accumulator(final TensorList qList, final TensorList kList, final TensorList vList,
                final CudaTensor outTensor,
                final int[] qDims, final int[] kDims, final int[] vDims,
                final int batchSize, final int kBatch, final int vBatch,
                final int qSeqLen, final int kvSeqLen,
                final Result.Accumulator qAcc, final Result.Accumulator kAcc, final Result.Accumulator vAcc,
                final boolean qAlive, final boolean kAlive, final boolean vAlive,
                final int numHeads, final int numKvHeads, final int headSize,
                final double scale, final boolean causal, final int slidingWindow,
                final Precision precision) {
      this.qList = qList;
      this.kList = kList;
      this.vList = vList;
      this.outTensor = outTensor;
      this.qDims = qDims;
      this.kDims = kDims;
      this.vDims = vDims;
      this.batchSize = batchSize;
      this.kBatch = kBatch;
      this.vBatch = vBatch;
      this.qSeqLen = qSeqLen;
      this.kvSeqLen = kvSeqLen;
      this.qAcc = qAcc;
      this.kAcc = kAcc;
      this.vAcc = vAcc;
      this.qAlive = qAlive;
      this.kAlive = kAlive;
      this.vAlive = vAlive;
      this.numHeads = numHeads;
      this.numKvHeads = numKvHeads;
      this.headSize = headSize;
      this.scale = scale;
      this.causal = causal;
      this.slidingWindow = slidingWindow;
      this.precision = precision;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      if (qAlive || kAlive || vAlive) {
        final int qElem = Tensor.length(qDims);
        final int kvElem = Tensor.length(kDims);
        final int dqBytes = qAlive ? (batchSize * qElem * precision.size) : 0;
        final int dkBytes = kAlive ? (kBatch * kvElem * precision.size) : 0;
        final int dvBytes = vAlive ? (vBatch * kvElem * precision.size) : 0;
        final int totalBytes = dqBytes + dkBytes + dvBytes;

        CudaMemory allGrads = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaMemory>) gpu -> {
          CudaTensor qGpu = gpu.getTensor(qList.addRef(), precision, MemoryType.Device, true);
          CudaTensor kGpu = gpu.getTensor(kList.addRef(), precision, MemoryType.Device, true);
          CudaTensor vGpu = gpu.getTensor(vList.addRef(), precision, MemoryType.Device, true);
          CudaTensor dOutGpu = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
          CudaTensor outDense = outTensor.getDense(gpu.addRef());

          CudaMemory qMem = qGpu.getMemory(gpu.addRef());
          CudaMemory kMem = kGpu.getMemory(gpu.addRef());
          CudaMemory vMem = vGpu.getMemory(gpu.addRef());
          CudaMemory outMem = outDense.getMemory(gpu.addRef());
          CudaMemory dOutMem = dOutGpu.getMemory(gpu.addRef());

          CudaMemory gradMem = gpu.allocate((long) totalBytes, MemoryType.Device, false);

          GemmaCudaKernels.gqaBackward(
              gpu.getDeviceId(), precision,
              qMem.getPtr(), kMem.getPtr(), vMem.getPtr(),
              outMem.getPtr(), null, dOutMem.getPtr(),
              qAlive ? gradMem.getPtr().withByteOffset(0L) : null,
              kAlive ? gradMem.getPtr().withByteOffset((long) dqBytes) : null,
              vAlive ? gradMem.getPtr().withByteOffset((long) (dqBytes + dkBytes)) : null,
              batchSize, kBatch, vBatch, qSeqLen, kvSeqLen,
              numHeads, numKvHeads, headSize, scale, causal, slidingWindow);

          qMem.dirty();
          kMem.dirty();
          vMem.dirty();
          outMem.dirty();
          dOutMem.dirty();
          gradMem.dirty();

          qMem.freeRef();
          kMem.freeRef();
          vMem.freeRef();
          outMem.freeRef();
          dOutMem.freeRef();
          outDense.freeRef();
          dOutGpu.freeRef();
          qGpu.freeRef();
          kGpu.freeRef();
          vGpu.freeRef();
          gpu.freeRef();
          return gradMem;
        }, qList.addRef(), kList.addRef(), vList.addRef(), outTensor.addRef(), delta.addRef()), delta.addRef());

        if (qAlive) {
          CudaTensorList dqList = wrapSlice(allGrads.addRef(), 0, batchSize, qDims, precision);
          DeltaSet<UUID> bufCopy = buffer == null ? null : buffer.addRef();
          qAcc.accept(bufCopy, dqList);
        }
        if (kAlive) {
          CudaTensorList dkList = wrapSlice(allGrads.addRef(), dqBytes, kBatch, kDims, precision);
          DeltaSet<UUID> bufCopy = buffer == null ? null : buffer.addRef();
          kAcc.accept(bufCopy, dkList);
        }
        if (vAlive) {
          CudaTensorList dvList = wrapSlice(allGrads.addRef(), dqBytes + dkBytes, vBatch, vDims, precision);
          DeltaSet<UUID> bufCopy = buffer == null ? null : buffer.addRef();
          vAcc.accept(bufCopy, dvList);
        }
        allGrads.freeRef();
      }
      delta.freeRef();
      if (buffer != null) {
        buffer.freeRef();
      }
    }

    @Nonnull
    private static CudaTensorList wrapSlice(final CudaMemory baseMem, final int byteOffset,
                                            final int batchCount, final int[] dims, final Precision prec) {
      return CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
        CudaMemory sliceMem = byteOffset == 0 ? baseMem.addRef() : baseMem.withByteOffset(byteOffset);
        int d2 = dims.length < 3 ? 1 : dims[2];
        int d1 = dims.length < 2 ? 1 : dims[1];
        int d0 = dims[0];
        CudaDevice.CudaTensorDescriptor desc = gpu.newTensorDescriptor(
            prec, batchCount, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
        gpu.freeRef();
        return new CudaTensorList(new CudaTensor(sliceMem, desc, prec), batchCount, dims, prec);
      }, baseMem));
    }

    @Override
    public void _free() {
      super._free();
      qList.freeRef();
      kList.freeRef();
      vList.freeRef();
      outTensor.freeRef();
      qAcc.freeRef();
      kAcc.freeRef();
      vAcc.freeRef();
    }
  }
}
