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

package com.simiacryptus.mindseye.models.gemma4;

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
import com.simiacryptus.ref.lang.RefIgnore;
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
 * CUDA/cuBLAS-accelerated differentiable Gemma 4 routed Mixture-of-Experts (MoE) layer (used in 26B-A4B models).
 * Executes pre-norms, router GEMM, top-k softmax routing, expert GeGLU/down projections, and post-norm
 * natively on the GPU in {@link Precision#Float} or {@link Precision#Double}, falling back to
 * {@link GemmaMoELayer} when {@link CudaSystem#isEnabled()} is {@code false}.
 */
@SuppressWarnings("serial")
public class CudaGemmaMoELayer extends LayerBase implements MultiPrecision {

  private final int dim;
  private final int expertCount;
  private final int expertUsedCount;
  private final int expertFF;
  private final double eps;
  private Precision precision = CudaSettings.INSTANCE().getDefaultPrecision();

  @Nonnull
  private final Tensor preFfwNorm2;
  @Nonnull
  private final Tensor ffnGateInpScale;
  @Nonnull
  private final Tensor ffnGateInp;
  @Nonnull
  private final Tensor ffnGateUpExps;
  @Nonnull
  private final Tensor ffnDownExps;
  @Nonnull
  private final Tensor ffnDownExpsScale;
  @Nonnull
  private final Tensor ffnPostNorm2;

  public CudaGemmaMoELayer(final int dim, final int expertCount, final int expertUsedCount,
                           final int expertFF, final double eps) {
    super();
    if (dim <= 0 || expertCount <= 0 || expertUsedCount <= 0 || expertUsedCount > expertCount || expertFF <= 0) {
      throw new IllegalArgumentException("Invalid CudaGemmaMoELayer configuration");
    }
    this.dim = dim;
    this.expertCount = expertCount;
    this.expertUsedCount = expertUsedCount;
    this.expertFF = expertFF;
    this.eps = eps;

    this.preFfwNorm2 = new Tensor(dim);
    this.preFfwNorm2.setAll(1.0);
    this.ffnGateInpScale = new Tensor(dim);
    this.ffnGateInpScale.setAll(1.0);
    this.ffnGateInp = new Tensor(dim, expertCount);
    initRandom(this.ffnGateInp.getData(), Math.sqrt(2.0 / (dim + expertCount)));
    this.ffnGateUpExps = new Tensor(dim, 2 * expertFF * expertCount);
    initRandom(this.ffnGateUpExps.getData(), Math.sqrt(2.0 / (dim + 2 * expertFF)));
    this.ffnDownExps = new Tensor(expertFF, dim * expertCount);
    initRandom(this.ffnDownExps.getData(), Math.sqrt(2.0 / (expertFF + dim)));
    this.ffnDownExpsScale = new Tensor(expertCount);
    this.ffnDownExpsScale.setAll(1.0);
    this.ffnPostNorm2 = new Tensor(dim);
    this.ffnPostNorm2.setAll(1.0);
  }

  public CudaGemmaMoELayer(final int dim, final int expertCount, final int expertUsedCount, final int expertFF,
                           final double eps,
                           @Nonnull final Tensor preFfwNorm2,
                           @Nonnull final Tensor ffnGateInpScale,
                           @Nonnull final Tensor ffnGateInp,
                           @Nonnull final Tensor ffnGateUpExps,
                           @Nonnull final Tensor ffnDownExps,
                           @Nonnull final Tensor ffnDownExpsScale,
                           @Nonnull final Tensor ffnPostNorm2) {
    super();
    this.dim = dim;
    this.expertCount = expertCount;
    this.expertUsedCount = expertUsedCount;
    this.expertFF = expertFF;
    this.eps = eps;
    this.preFfwNorm2 = preFfwNorm2;
    this.ffnGateInpScale = ffnGateInpScale;
    this.ffnGateInp = ffnGateInp;
    this.ffnGateUpExps = ffnGateUpExps;
    this.ffnDownExps = ffnDownExps;
    this.ffnDownExpsScale = ffnDownExpsScale;
    this.ffnPostNorm2 = ffnPostNorm2;
  }

  protected CudaGemmaMoELayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    this.dim = json.get("dim").getAsInt();
    this.expertCount = json.get("expertCount").getAsInt();
    this.expertUsedCount = json.get("expertUsedCount").getAsInt();
    this.expertFF = json.get("expertFF").getAsInt();
    this.eps = json.get("eps").getAsDouble();
    if (json.has("precision")) {
      this.precision = Precision.valueOf(json.get("precision").getAsString());
    }
    this.preFfwNorm2 = requireTensor(json, "preFfwNorm2", resources);
    this.ffnGateInpScale = requireTensor(json, "ffnGateInpScale", resources);
    this.ffnGateInp = requireTensor(json, "ffnGateInp", resources);
    this.ffnGateUpExps = requireTensor(json, "ffnGateUpExps", resources);
    this.ffnDownExps = requireTensor(json, "ffnDownExps", resources);
    this.ffnDownExpsScale = requireTensor(json, "ffnDownExpsScale", resources);
    this.ffnPostNorm2 = requireTensor(json, "ffnPostNorm2", resources);
  }

  private static Tensor requireTensor(final JsonObject json, final String key, final Map<CharSequence, byte[]> rs) {
    Tensor t = Tensor.fromJson(json.get(key), rs);
    if (t == null) {
      throw new IllegalArgumentException("Missing tensor " + key + " in CudaGemmaMoELayer JSON");
    }
    return t;
  }

  private static void initRandom(@Nonnull final double[] d, final double scale) {
    for (int i = 0; i < d.length; i++) {
      d[i] = (1.0 - 2.0 * Util.R.get().nextDouble()) * scale;
    }
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static CudaGemmaMoELayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new CudaGemmaMoELayer(json, rs);
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
    GemmaMoELayer cpu = new GemmaMoELayer(
        dim, expertCount, expertUsedCount, expertFF, eps,
        preFfwNorm2.addRef(), ffnGateInpScale.addRef(), ffnGateInp.addRef(),
        ffnGateUpExps.addRef(), ffnDownExps.addRef(), ffnDownExpsScale.addRef(), ffnPostNorm2.addRef());
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
      throw new IllegalArgumentException("CudaGemmaMoELayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int[] dims = inputData.getDimensions();
    int elemLen = Tensor.length(dims);
    if (elemLen % dim != 0) {
      inputData.freeRef();
      in0.freeRef();
      throw new IllegalArgumentException("Input length " + elemLen + " is not divisible by dim " + dim);
    }
    if (dims.length < 1 || dims.length > 3) {
      inputData.freeRef();
      Layer compatibilityLayer = getCompatibilityLayer();
      Result result = compatibilityLayer.eval(in0);
      compatibilityLayer.freeRef();
      return result;
    }

    int batchSize = inputData.length();
    int totalTokens = batchSize * (elemLen / dim);
    final Precision prec = this.precision;

    CudaTensorList outputData = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
      CudaTensor inTensor = gpu.getTensor(inputData.addRef(), prec, MemoryType.Device, true);
      CudaMemory inMem = inTensor.getMemory(gpu.addRef());
      CudaMemory outMem = gpu.allocate((long) totalTokens * dim * prec.size, MemoryType.Device, true);
      long wsBytes = GemmaCudaKernels.moeForwardWorkspaceBytes(prec, totalTokens, dim, expertCount, expertUsedCount, expertFF);
      CudaMemory wsMem = gpu.allocate(wsBytes, MemoryType.Device, true);

      CudaMemory preMem = gpu.allocate((long) dim * Double.BYTES, MemoryType.Device, true);
      GemmaCudaKernels.writeDoubles(preMem.getPtr(), preFfwNorm2.getData());
      CudaMemory gisMem = gpu.allocate((long) dim * Double.BYTES, MemoryType.Device, true);
      GemmaCudaKernels.writeDoubles(gisMem.getPtr(), ffnGateInpScale.getData());
      CudaMemory giMem = gpu.allocate((long) ffnGateInp.length() * prec.size, MemoryType.Device, true);
      giMem.write(prec, ffnGateInp.addRef());
      CudaMemory gueMem = gpu.allocate((long) ffnGateUpExps.length() * prec.size, MemoryType.Device, true);
      gueMem.write(prec, ffnGateUpExps.addRef());
      CudaMemory deMem = gpu.allocate((long) ffnDownExps.length() * prec.size, MemoryType.Device, true);
      deMem.write(prec, ffnDownExps.addRef());
      CudaMemory desMem = gpu.allocate((long) expertCount * Double.BYTES, MemoryType.Device, true);
      GemmaCudaKernels.writeDoubles(desMem.getPtr(), ffnDownExpsScale.getData());
      CudaMemory postMem = gpu.allocate((long) dim * Double.BYTES, MemoryType.Device, true);
      GemmaCudaKernels.writeDoubles(postMem.getPtr(), ffnPostNorm2.getData());

      GemmaCudaKernels.moeForward(
          gpu.getDeviceId(), prec,
          inMem.getPtr(), outMem.getPtr(), wsMem.getPtr(),
          preMem.getPtr(), gisMem.getPtr(), giMem.getPtr(),
          gueMem.getPtr(), deMem.getPtr(), desMem.getPtr(), postMem.getPtr(),
          totalTokens, dim, expertCount, expertUsedCount, expertFF, eps);

      inMem.dirty();
      outMem.dirty();
      wsMem.dirty();
      preMem.dirty();
      gisMem.dirty();
      giMem.dirty();
      gueMem.dirty();
      deMem.dirty();
      desMem.dirty();
      postMem.dirty();

      wsMem.freeRef();
      preMem.freeRef();
      gisMem.freeRef();
      giMem.freeRef();
      gueMem.freeRef();
      deMem.freeRef();
      desMem.freeRef();
      postMem.freeRef();
      inMem.freeRef();
      inTensor.freeRef();

      int d2 = dims.length < 3 ? 1 : dims[2];
      int d1 = dims.length < 2 ? 1 : dims[1];
      int d0 = dims[0];
      CudaDevice.CudaTensorDescriptor outDesc = gpu.newTensorDescriptor(
          prec, batchSize, d2, d1, d0, d2 * d1 * d0, d1 * d0, d0, 1);
      gpu.freeRef();
      return new CudaTensorList(new CudaTensor(outMem, outDesc, prec), batchSize, dims, prec);
    }, inputData.addRef(), ffnGateInp.addRef(), ffnGateUpExps.addRef(), ffnDownExps.addRef()), inputData.addRef());

    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();
    Accumulator accumulator = new Accumulator(
        inputData, dims, batchSize, dim, expertCount, expertUsedCount, expertFF, eps, isFrozen(), prec,
        preFfwNorm2.addRef(), ffnGateInpScale.addRef(), ffnGateInp.addRef(),
        ffnGateUpExps.addRef(), ffnDownExps.addRef(), ffnDownExpsScale.addRef(), ffnPostNorm2.addRef(),
        upstream, alive);
    return new Result(outputData, accumulator, alive || !isFrozen());
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, @Nonnull final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("dim", dim);
    json.addProperty("expertCount", expertCount);
    json.addProperty("expertUsedCount", expertUsedCount);
    json.addProperty("expertFF", expertFF);
    json.addProperty("eps", eps);
    json.addProperty("precision", precision.name());
    json.add("preFfwNorm2", preFfwNorm2.getJson(resources, dataSerializer));
    json.add("ffnGateInpScale", ffnGateInpScale.getJson(resources, dataSerializer));
    json.add("ffnGateInp", ffnGateInp.getJson(resources, dataSerializer));
    json.add("ffnGateUpExps", ffnGateUpExps.getJson(resources, dataSerializer));
    json.add("ffnDownExps", ffnDownExps.getJson(resources, dataSerializer));
    json.add("ffnDownExpsScale", ffnDownExpsScale.getJson(resources, dataSerializer));
    json.add("ffnPostNorm2", ffnPostNorm2.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList(
        preFfwNorm2.getData(),
        ffnGateInpScale.getData(),
        ffnGateInp.getData(),
        ffnGateUpExps.getData(),
        ffnDownExps.getData(),
        ffnDownExpsScale.getData(),
        ffnPostNorm2.getData()
    );
  }

  @Override
  public void _free() {
    preFfwNorm2.freeRef();
    ffnGateInpScale.freeRef();
    ffnGateInp.freeRef();
    ffnGateUpExps.freeRef();
    ffnDownExps.freeRef();
    ffnDownExpsScale.freeRef();
    ffnPostNorm2.freeRef();
    super._free();
  }

  @Nonnull
  @Override
  public CudaGemmaMoELayer addRef() {
    return (CudaGemmaMoELayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList inputData;
    private final int[] dims;
    private final int batchSize;
    private final int dim;
    private final int expertCount;
    private final int expertUsedCount;
    private final int expertFF;
    private final double eps;
    private final boolean frozen;
    private final Precision precision;
    private final Tensor preFfwNorm2;
    private final Tensor ffnGateInpScale;
    private final Tensor ffnGateInp;
    private final Tensor ffnGateUpExps;
    private final Tensor ffnDownExps;
    private final Tensor ffnDownExpsScale;
    private final Tensor ffnPostNorm2;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final int[] dims, final int batchSize,
                final int dim, final int expertCount, final int expertUsedCount,
                final int expertFF, final double eps, final boolean frozen, final Precision precision,
                final Tensor preFfwNorm2, final Tensor ffnGateInpScale, final Tensor ffnGateInp,
                final Tensor ffnGateUpExps, final Tensor ffnDownExps, final Tensor ffnDownExpsScale,
                final Tensor ffnPostNorm2, final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.dims = dims;
      this.batchSize = batchSize;
      this.dim = dim;
      this.expertCount = expertCount;
      this.expertUsedCount = expertUsedCount;
      this.expertFF = expertFF;
      this.eps = eps;
      this.frozen = frozen;
      this.precision = precision;
      this.preFfwNorm2 = preFfwNorm2;
      this.ffnGateInpScale = ffnGateInpScale;
      this.ffnGateInp = ffnGateInp;
      this.ffnGateUpExps = ffnGateUpExps;
      this.ffnDownExps = ffnDownExps;
      this.ffnDownExpsScale = ffnDownExpsScale;
      this.ffnPostNorm2 = ffnPostNorm2;
      this.upstream = upstream;
      this.alive = alive;
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      final boolean accW = !frozen && buffer != null;
      if (alive || accW) {
        final double[] gPre = accW ? new double[preFfwNorm2.length()] : null;
        final double[] gGis = accW ? new double[ffnGateInpScale.length()] : null;
        final double[] gGi = accW ? new double[ffnGateInp.length()] : null;
        final double[] gGue = accW ? new double[ffnGateUpExps.length()] : null;
        final double[] gDe = accW ? new double[ffnDownExps.length()] : null;
        final double[] gDes = accW ? new double[ffnDownExpsScale.length()] : null;
        final double[] gPost = accW ? new double[ffnPostNorm2.length()] : null;

        final int elemLen = Tensor.length(dims);
        final int totalTokens = batchSize * (elemLen / dim);

        CudaTensorList passback = CudaSystem.run(RefUtil.wrapInterface((RefFunction<CudnnHandle, CudaTensorList>) gpu -> {
          CudaTensor inTensor = gpu.getTensor(inputData.addRef(), precision, MemoryType.Device, true);
          CudaTensor dyTensor = gpu.getTensor(delta.addRef(), precision, MemoryType.Device, true);
          CudaMemory inMem = inTensor.getMemory(gpu.addRef());
          CudaMemory dyMem = dyTensor.getMemory(gpu.addRef());

          long wsBytes = GemmaCudaKernels.moeBackwardWorkspaceBytes(precision, totalTokens, dim, expertCount, expertUsedCount, expertFF);
          CudaMemory wsMem = gpu.allocate(wsBytes, MemoryType.Device, false);

          CudaMemory preMem = gpu.allocate((long) dim * Double.BYTES, MemoryType.Device, true);
          GemmaCudaKernels.writeDoubles(preMem.getPtr(), preFfwNorm2.getData());
          CudaMemory gisMem = gpu.allocate((long) dim * Double.BYTES, MemoryType.Device, true);
          GemmaCudaKernels.writeDoubles(gisMem.getPtr(), ffnGateInpScale.getData());
          CudaMemory giMem = gpu.allocate((long) ffnGateInp.length() * precision.size, MemoryType.Device, true);
          giMem.write(precision, ffnGateInp.addRef());
          CudaMemory gueMem = gpu.allocate((long) ffnGateUpExps.length() * precision.size, MemoryType.Device, true);
          gueMem.write(precision, ffnGateUpExps.addRef());
          CudaMemory deMem = gpu.allocate((long) ffnDownExps.length() * precision.size, MemoryType.Device, true);
          deMem.write(precision, ffnDownExps.addRef());
          CudaMemory desMem = gpu.allocate((long) expertCount * Double.BYTES, MemoryType.Device, true);
          GemmaCudaKernels.writeDoubles(desMem.getPtr(), ffnDownExpsScale.getData());
          CudaMemory postMem = gpu.allocate((long) dim * Double.BYTES, MemoryType.Device, true);
          GemmaCudaKernels.writeDoubles(postMem.getPtr(), ffnPostNorm2.getData());

          CudaMemory dxMem = alive ? gpu.allocate((long) totalTokens * dim * precision.size, MemoryType.Device, true) : null;

          CudaMemory gPreMem = accW ? gpu.allocate((long) gPre.length * Double.BYTES, MemoryType.Device, false) : null;
          CudaMemory gGisMem = accW ? gpu.allocate((long) gGis.length * Double.BYTES, MemoryType.Device, false) : null;
          CudaMemory gGiMem = accW ? gpu.allocate((long) gGi.length * precision.size, MemoryType.Device, false) : null;
          CudaMemory gGueMem = accW ? gpu.allocate((long) gGue.length * Double.BYTES, MemoryType.Device, false) : null;
          CudaMemory gDeMem = accW ? gpu.allocate((long) gDe.length * Double.BYTES, MemoryType.Device, false) : null;
          CudaMemory gDesMem = accW ? gpu.allocate((long) gDes.length * Double.BYTES, MemoryType.Device, false) : null;
          CudaMemory gPostMem = accW ? gpu.allocate((long) gPost.length * Double.BYTES, MemoryType.Device, false) : null;

          GemmaCudaKernels.moeBackward(
              gpu.getDeviceId(), precision,
              inMem.getPtr(), dyMem.getPtr(), dxMem == null ? null : dxMem.getPtr(),
              wsMem.getPtr(),
              preMem.getPtr(), gisMem.getPtr(), giMem.getPtr(),
              gueMem.getPtr(), deMem.getPtr(), desMem.getPtr(), postMem.getPtr(),
              gPreMem == null ? null : gPreMem.getPtr(),
              gGisMem == null ? null : gGisMem.getPtr(),
              gGiMem == null ? null : gGiMem.getPtr(),
              gGueMem == null ? null : gGueMem.getPtr(),
              gDeMem == null ? null : gDeMem.getPtr(),
              gDesMem == null ? null : gDesMem.getPtr(),
              gPostMem == null ? null : gPostMem.getPtr(),
              totalTokens, dim, expertCount, expertUsedCount, expertFF, eps, accW);

          inMem.dirty();
          dyMem.dirty();
          wsMem.dirty();
          preMem.dirty();
          gisMem.dirty();
          giMem.dirty();
          gueMem.dirty();
          deMem.dirty();
          desMem.dirty();
          postMem.dirty();

          if (gPreMem != null) {
            gPreMem.dirty();
            GemmaCudaKernels.readDoubles(gPreMem.getPtr(), gPre);
            gPreMem.freeRef();
          }
          if (gGisMem != null) {
            gGisMem.dirty();
            GemmaCudaKernels.readDoubles(gGisMem.getPtr(), gGis);
            gGisMem.freeRef();
          }
          if (gGiMem != null) {
            gGiMem.dirty();
            GemmaCudaKernels.readElemAsDoubles(precision, gGiMem.getPtr(), gGi);
            gGiMem.freeRef();
          }
          if (gGueMem != null) {
            gGueMem.dirty();
            GemmaCudaKernels.readDoubles(gGueMem.getPtr(), gGue);
            gGueMem.freeRef();
          }
          if (gDeMem != null) {
            gDeMem.dirty();
            GemmaCudaKernels.readDoubles(gDeMem.getPtr(), gDe);
            gDeMem.freeRef();
          }
          if (gDesMem != null) {
            gDesMem.dirty();
            GemmaCudaKernels.readDoubles(gDesMem.getPtr(), gDes);
            gDesMem.freeRef();
          }
          if (gPostMem != null) {
            gPostMem.dirty();
            GemmaCudaKernels.readDoubles(gPostMem.getPtr(), gPost);
            gPostMem.freeRef();
          }

          wsMem.freeRef();
          preMem.freeRef();
          gisMem.freeRef();
          giMem.freeRef();
          gueMem.freeRef();
          deMem.freeRef();
          desMem.freeRef();
          postMem.freeRef();
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
        }, inputData.addRef(), delta.addRef(), ffnGateInp.addRef(), ffnGateUpExps.addRef(), ffnDownExps.addRef()), delta.addRef());

        if (accW && buffer != null) {
          accumulateParam(buffer, preFfwNorm2, gPre);
          accumulateParam(buffer, ffnGateInpScale, gGis);
          accumulateParam(buffer, ffnGateInp, gGi);
          accumulateParam(buffer, ffnGateUpExps, gGue);
          accumulateParam(buffer, ffnDownExps, gDe);
          accumulateParam(buffer, ffnDownExpsScale, gDes);
          accumulateParam(buffer, ffnPostNorm2, gPost);
        }

        if (alive && passback != null) {
          DeltaSet<UUID> bufCopy = buffer == null ? null : buffer.addRef();
          upstream.accept(bufCopy, passback);
        } else if (passback != null) {
          passback.freeRef();
        }
      }
      delta.freeRef();
      if (buffer != null) {
        buffer.freeRef();
      }
    }

    private static void accumulateParam(@RefIgnore @Nonnull final DeltaSet<UUID> buffer,
                                        @RefIgnore @Nonnull final Tensor param, @Nonnull final double[] grad) {
      Delta<UUID> d = buffer.get(param.getId(), param.getData());
      if (d != null) {
        d.addInPlace(grad);
        d.freeRef();
      }
    }

    @Override
    public void _free() {
      super._free();
      inputData.freeRef();
      preFfwNorm2.freeRef();
      ffnGateInpScale.freeRef();
      ffnGateInp.freeRef();
      ffnGateUpExps.freeRef();
      ffnDownExps.freeRef();
      ffnDownExpsScale.freeRef();
      ffnPostNorm2.freeRef();
      upstream.freeRef();
    }
  }
}
