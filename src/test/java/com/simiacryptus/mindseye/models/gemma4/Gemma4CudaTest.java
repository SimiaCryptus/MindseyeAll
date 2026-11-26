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
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.lang.cudnn.CudaSystem;
import com.simiacryptus.mindseye.lang.cudnn.CudaTensorList;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.ChunkSelectLayer;
import com.simiacryptus.mindseye.layers.cudnn.GeluActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.GroupedQueryAttentionLayer;
import com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer;
import com.simiacryptus.mindseye.layers.cudnn.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.cudnn.ProductLayer;
import com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer;
import com.simiacryptus.mindseye.layers.cudnn.RopeLayer;
import com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer;
import com.simiacryptus.mindseye.layers.cudnn.SoftcapLayer;
import com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer;
import com.simiacryptus.mindseye.layers.cudnn.TokenEmbeddingLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefCollectors;
import com.simiacryptus.ref.wrappers.RefList;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/**
 * GPU &amp; CPU-compatibility tests for all CUDA/cuDNN/cuBLAS Gemma 4 layers and full CUDA networks.
 */
@RefIgnore
public class Gemma4CudaTest {

  private static Tensor randomTensor(final long seed, final int... dims) {
    Random rng = new Random(seed);
    int len = Tensor.length(dims);
    double[] data = new double[len];
    for (int i = 0; i < len; i++) {
      data[i] = (rng.nextDouble() * 2.0 - 1.0) * 0.8;
    }
    return new Tensor(data, dims);
  }

  private static Tensor[] copyTensors(final Tensor[] src) {
    Tensor[] out = new Tensor[src.length];
    for (int i = 0; i < src.length; i++) {
      out[i] = src[i].copy();
    }
    return out;
  }

  private static double evalWeightedSum(final Layer layer, final double[] outWeights, final Tensor... inputs) {
    Result res = layer.eval(copyTensors(inputs));
    if (CudaSystem.isEnabled() && !(layer instanceof PipelineNetwork)) {
      TensorList tl = res.getData();
      Assertions.assertInstanceOf(CudaTensorList.class, tl,
          "Expected CudaTensorList output from " + layer.getClass().getName());
      tl.freeRef();
    }
    Tensor out = Result.getData0(res);
    try {
      double[] y = out.getData();
      double sum = 0.0;
      for (int i = 0; i < y.length; i++) {
        sum += y[i] * outWeights[i];
      }
      return sum;
    } finally {
      out.freeRef();
    }
  }

  private static void assertCentralDifferenceAndJson(final Layer layer, final boolean checkInputGrad,
                                                     final double tol, final Tensor... inputs) {
    try {
      // 1. JSON round-trip check
      Map<CharSequence, byte[]> resources = new HashMap<CharSequence, byte[]>();
      JsonObject json = layer.getJson(resources, Tensor.json_precision).getAsJsonObject();
      Layer restored = Layer.fromJson(json, resources);
      try {
        Result origRes = layer.eval(copyTensors(inputs));
        if (CudaSystem.isEnabled()) {
          TensorList tl = origRes.getData();
          Assertions.assertInstanceOf(CudaTensorList.class, tl);
          tl.freeRef();
        }
        Result restRes = restored.eval(copyTensors(inputs));
        Tensor origOut = Result.getData0(origRes);
        Tensor restOut = Result.getData0(restRes);
        try {
          double[] a = origOut.getData();
          double[] b = restOut.getData();
          Assertions.assertEquals(a.length, b.length);
          for (int i = 0; i < a.length; i++) {
            Assertions.assertEquals(a[i], b[i], 1e-4, "JSON round-trip mismatch at " + i);
          }
        } finally {
          origOut.freeRef();
          restOut.freeRef();
        }
      } finally {
        restored.freeRef();
      }

      // 2. Central finite-difference check for input and parameter gradients
      Result fwd = layer.eval(copyTensors(inputs));
      Tensor fwdOut = Result.getData0(fwd.addRef());
      int outLen = fwdOut.length();
      int[] outDims = fwdOut.getDimensions();
      fwdOut.freeRef();
      fwd.freeRef();

      Tensor seedTensor = randomTensor(1234L, outLen);
      double[] upstreamSeed = seedTensor.copyData();
      seedTensor.freeRef();

      final double[][] analyticalInputGrads = new double[inputs.length][];
      Result[] wrappedInputs = new Result[inputs.length];
      for (int idx = 0; idx < inputs.length; idx++) {
        final int slot = idx;
        analyticalInputGrads[slot] = new double[inputs[slot].length()];
        Result.Accumulator acc = new Result.Accumulator() {
          @Override
          public void accept(final DeltaSet<UUID> buffer, final TensorList data) {
            if (CudaSystem.isEnabled()) {
              Assertions.assertInstanceOf(CudaTensorList.class, data,
                  "Expected GPU-resident CudaTensorList passback from " + layer.getClass().getSimpleName());
            }
            Tensor d0 = data.get(0);
            System.arraycopy(d0.getData(), 0, analyticalInputGrads[slot], 0, analyticalInputGrads[slot].length);
            d0.freeRef();
            data.freeRef();
            if (buffer != null) {
              buffer.freeRef();
            }
          }

          @Override
          public void _free() {
            super._free();
          }
        };
        wrappedInputs[idx] = new Result(new TensorArray(inputs[idx].copy()), acc, true);
      }

      Result eval = layer.eval(wrappedInputs);
      DeltaSet<UUID> deltaSet = new DeltaSet<UUID>();
      eval.accumulate(deltaSet.addRef(), new TensorArray(new Tensor(upstreamSeed.clone(), outDims)));
      eval.freeRef();

      double eps = 1e-6;
      if (checkInputGrad) {
        for (int idx = 0; idx < inputs.length; idx++) {
          double[] x = inputs[idx].getData();
          for (int i = 0; i < x.length; i++) {
            double orig = x[i];
            x[i] = orig + eps;
            double fPlus = evalWeightedSum(layer, upstreamSeed, inputs);
            x[i] = orig - eps;
            double fMinus = evalWeightedSum(layer, upstreamSeed, inputs);
            x[i] = orig;
            double numGrad = (fPlus - fMinus) / (2.0 * eps);
            Assertions.assertEquals(numGrad, analyticalInputGrads[idx][i], tol,
                layer.getClass().getSimpleName() + " input[" + idx + "] grad mismatch at " + i);
          }
        }
      }

      if (!layer.isFrozen()) {
        RefList<double[]> states = layer.state();
        for (int s = 0; s < states.size(); s++) {
          double[] param = states.get(s);
          double[] analyticalParamGrad = null;
          RefList<Delta<UUID>> allDeltas = deltaSet.stream().collect(RefCollectors.toList());
          for (int d = 0; d < allDeltas.size(); d++) {
            Delta<UUID> item = allDeltas.get(d);
            if (item.target == param) {
              analyticalParamGrad = item.getDelta();
            }
            item.freeRef();
          }
          allDeltas.freeRef();
          Assertions.assertNotNull(analyticalParamGrad, "Missing parameter gradient for state " + s);

          for (int i = 0; i < param.length; i++) {
            double orig = param[i];
            param[i] = orig + eps;
            double fPlus = evalWeightedSum(layer, upstreamSeed, inputs);
            param[i] = orig - eps;
            double fMinus = evalWeightedSum(layer, upstreamSeed, inputs);
            param[i] = orig;
            double numGrad = (fPlus - fMinus) / (2.0 * eps);
            Assertions.assertEquals(numGrad, analyticalParamGrad[i], tol,
                layer.getClass().getSimpleName() + " param[" + s + "] grad mismatch at " + i);
          }
        }
        states.freeRef();
      }
      deltaSet.freeRef();
    } finally {
      layer.freeRef();
      RefUtil.freeRef(inputs);
    }
  }

  private static void assertGpuLayerMatchesCpuReference(final Layer gpuLayer, final Layer cpuLayer,
                                                        final double tol, final boolean checkInputGrad,
                                                        final Tensor... inputs) {
    try {
      Result gpuFwd = gpuLayer.eval(copyTensors(inputs));
      if (CudaSystem.isEnabled()) {
        TensorList tl = gpuFwd.getData();
        Assertions.assertInstanceOf(CudaTensorList.class, tl);
        tl.freeRef();
      }
      Result cpuFwd = cpuLayer.eval(copyTensors(inputs));
      Tensor gpuOut = Result.getData0(gpuFwd);
      Tensor cpuOut = Result.getData0(cpuFwd);
      int[] outDims = cpuOut.getDimensions();
      int outLen = cpuOut.length();
      try {
        Assertions.assertArrayEquals(outDims, gpuOut.getDimensions());
        for (int i = 0; i < outLen; i++) {
          Assertions.assertEquals(cpuOut.getData()[i], gpuOut.getData()[i], tol,
              gpuLayer.getClass().getSimpleName() + " forward mismatch at " + i);
        }
      } finally {
        gpuOut.freeRef();
        cpuOut.freeRef();
      }

      Tensor seedTensor = randomTensor(4321L, outLen);
      double[] seed = seedTensor.copyData();
      seedTensor.freeRef();

      final double[][] gpuInGrads = new double[inputs.length][];
      final double[][] cpuInGrads = new double[inputs.length][];
      Result[] gpuWrapped = new Result[inputs.length];
      Result[] cpuWrapped = new Result[inputs.length];
      for (int idx = 0; idx < inputs.length; idx++) {
        final int slot = idx;
        gpuInGrads[slot] = new double[inputs[slot].length()];
        cpuInGrads[slot] = new double[inputs[slot].length()];
        gpuWrapped[idx] = new Result(new TensorArray(inputs[slot].copy()), new Result.Accumulator() {
          @Override
          public void accept(final DeltaSet<UUID> buffer, final TensorList data) {
            Tensor t = data.get(0);
            System.arraycopy(t.getData(), 0, gpuInGrads[slot], 0, gpuInGrads[slot].length);
            t.freeRef();
            data.freeRef();
            if (buffer != null) {
              buffer.freeRef();
            }
          }

          @Override
          public void _free() {
            super._free();
          }
        }, true);
        cpuWrapped[idx] = new Result(new TensorArray(inputs[slot].copy()), new Result.Accumulator() {
          @Override
          public void accept(final DeltaSet<UUID> buffer, final TensorList data) {
            Tensor t = data.get(0);
            System.arraycopy(t.getData(), 0, cpuInGrads[slot], 0, cpuInGrads[slot].length);
            t.freeRef();
            data.freeRef();
            if (buffer != null) {
              buffer.freeRef();
            }
          }

          @Override
          public void _free() {
            super._free();
          }
        }, true);
      }

      Result gRes = gpuLayer.eval(gpuWrapped);
      DeltaSet<UUID> gDeltas = new DeltaSet<UUID>();
      gRes.accumulate(gDeltas.addRef(), new TensorArray(new Tensor(seed.clone(), outDims)));
      gRes.freeRef();

      Result cRes = cpuLayer.eval(cpuWrapped);
      DeltaSet<UUID> cDeltas = new DeltaSet<UUID>();
      cRes.accumulate(cDeltas.addRef(), new TensorArray(new Tensor(seed.clone(), outDims)));
      cRes.freeRef();

      if (checkInputGrad) {
        for (int idx = 0; idx < inputs.length; idx++) {
          for (int i = 0; i < gpuInGrads[idx].length; i++) {
            Assertions.assertEquals(cpuInGrads[idx][i], gpuInGrads[idx][i], tol,
                gpuLayer.getClass().getSimpleName() + " dx[" + idx + "] mismatch at " + i);
          }
        }
      }

      gDeltas.freeRef();
      cDeltas.freeRef();
    } finally {
      gpuLayer.freeRef();
      cpuLayer.freeRef();
      RefUtil.freeRef(inputs);
    }
  }

  @Test
  @Tag("UnitTest")
  public void testCudaPrimitiveLayersDoubleDerivativesAndJson() {
    RmsNormLayer rmsLearned = new RmsNormLayer(4, 1e-5, true);
    rmsLearned.setPrecision(Precision.Double);
    rmsLearned.setWeights(new double[]{1.2, 0.8, -0.5, 1.1});
    assertCentralDifferenceAndJson(rmsLearned, true, 1e-5, randomTensor(101L, 8, 3));

    RmsNormLayer rmsBare = new RmsNormLayer(4, 1e-5, false);
    rmsBare.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(rmsBare, true, 1e-5, randomTensor(102L, 8, 2));

    GeluActivationLayer gelu = new GeluActivationLayer();
    gelu.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(gelu, true, 1e-5, randomTensor(103L, 6, 2));

    SoftcapLayer softcap = new SoftcapLayer(5.0);
    softcap.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(softcap, true, 1e-5, randomTensor(104L, 6, 2));

    ScalarScaleLayer scale = new ScalarScaleLayer(0.75, false);
    scale.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(scale, true, 1e-5, randomTensor(105L, 5, 2));

    LinearProjectionLayer proj = new LinearProjectionLayer(6, 4);
    proj.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(proj, true, 1e-5, randomTensor(106L, 6, 3));

    RopeLayer rope = new RopeLayer(2, 4, 10000.0, new double[]{1.0, 2.0}, 3);
    rope.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(rope, true, 1e-5, randomTensor(107L, 8, 3));

    ChunkSelectLayer slice = new ChunkSelectLayer(3, 4, 2);
    slice.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(slice, true, 1e-5, randomTensor(108L, 12, 3));

    GroupedQueryAttentionLayer gqaSwa = new GroupedQueryAttentionLayer(4, 2, 4, 1.0, true, 2);
    gqaSwa.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(
        gqaSwa, true, 1e-5,
        randomTensor(109L, 16, 4),
        randomTensor(110L, 8, 4),
        randomTensor(111L, 8, 4)
    );

    // Tied TokenEmbeddingLayer + LinearProjectionLayer shared weight gradient accumulation on GPU
    Tensor sharedTable = randomTensor(112L, 4, 8);
    PipelineNetwork tiedNet = new PipelineNetwork(1);
    DAGNode tokIn = tiedNet.getInput(0);
    TokenEmbeddingLayer tokEmb = new TokenEmbeddingLayer(8, 4, 2.0, sharedTable.addRef());
    tokEmb.setPrecision(Precision.Double);
    DAGNode embNode = tiedNet.add(tokEmb, tokIn);
    LinearProjectionLayer lmHead = new LinearProjectionLayer(4, 8, sharedTable);
    lmHead.setPrecision(Precision.Double);
    DAGNode logitsNode = tiedNet.add(lmHead, embNode);
    tiedNet.setHead(logitsNode);
    assertCentralDifferenceAndJson(tiedNet, false, 1e-5, new Tensor(new double[]{1, 5, 2}, 3));
  }

  @Test
  @Tag("UnitTest")
  public void testCudaPrimitiveLayersFloatVsCpuReference() {
    RmsNormLayer gpuRms = new RmsNormLayer(8, 1e-5, true);
    gpuRms.setPrecision(Precision.Float);
    gpuRms.setWeights(new double[]{1.1, 0.9, -0.7, 1.3, 0.5, -1.1, 0.8, 1.2});
    assertGpuLayerMatchesCpuReference(gpuRms.addRef(), gpuRms.getCompatibilityLayer(), 1e-4, true, randomTensor(201L, 16, 3));
    gpuRms.freeRef();

    LinearProjectionLayer gpuProj = new LinearProjectionLayer(8, 6);
    gpuProj.setPrecision(Precision.Float);
    assertGpuLayerMatchesCpuReference(gpuProj.addRef(), gpuProj.getCompatibilityLayer(), 1e-4, true, randomTensor(202L, 8, 4));
    gpuProj.freeRef();

    RopeLayer gpuRope = new RopeLayer(2, 4, 10000.0, new double[]{1.5, 0.75}, 2);
    gpuRope.setPrecision(Precision.Float);
    assertGpuLayerMatchesCpuReference(gpuRope.addRef(), gpuRope.getCompatibilityLayer(), 1e-4, true, randomTensor(203L, 8, 4));
    gpuRope.freeRef();

    GroupedQueryAttentionLayer gpuGqa = new GroupedQueryAttentionLayer(4, 2, 4, 1.0, true, 3);
    gpuGqa.setPrecision(Precision.Float);
    assertGpuLayerMatchesCpuReference(
        gpuGqa.addRef(), gpuGqa.getCompatibilityLayer(), 1e-4, true,
        randomTensor(204L, 16, 5), randomTensor(205L, 8, 5), randomTensor(206L, 8, 5));
    gpuGqa.freeRef();

    // 2D sequence SumInputsLayer and ProductLayer on GPU
    SumInputsLayer gpuSum = new SumInputsLayer();
    gpuSum.setPrecision(Precision.Float);
    gpuSum.setParallel(false);
    assertGpuLayerMatchesCpuReference(
        gpuSum.addRef(), gpuSum.getCompatibilityLayer(), 1e-5, true,
        randomTensor(207L, 8, 3), randomTensor(208L, 8, 3));
    gpuSum.freeRef();

    ProductLayer gpuProd = new ProductLayer();
    gpuProd.setPrecision(Precision.Float);
    assertGpuLayerMatchesCpuReference(
        gpuProd.addRef(), gpuProd.getCompatibilityLayer(), 1e-5, true,
        randomTensor(209L, 8, 3), randomTensor(210L, 8, 3));
    gpuProd.freeRef();
  }

  @Test
  @Tag("UnitTest")
  public void testCudaGemmaMoELayerDerivativesAndEquivalence() {
    CudaGemmaMoELayer moeDouble = new CudaGemmaMoELayer(6, 4, 2, 5, 1e-5);
    moeDouble.setPrecision(Precision.Double);
    assertCentralDifferenceAndJson(moeDouble, true, 1e-5, randomTensor(301L, 6, 2));

    CudaGemmaMoELayer moeFloat = new CudaGemmaMoELayer(8, 4, 2, 6, 1e-5);
    moeFloat.setPrecision(Precision.Float);
    assertGpuLayerMatchesCpuReference(
        moeFloat.addRef(), moeFloat.getCompatibilityLayer(), 1e-4, true, randomTensor(302L, 8, 3));
    moeFloat.freeRef();
  }

  @Test
  @Tag("UnitTest")
  public void testFullCudaGemma4NetworksAndFineTuning() {
    // 1. Dense + PLE + SWA + SharedKV + V=K + Softcap + Tied Embeddings on GPU
    Gemma4Config denseConfig = new Gemma4Config.Builder()
        .setEmbeddingLength(16)
        .setDefaultFeedForwardLength(32)
        .setNumberOfLayers(4)
        .setNumberOfHeads(4)
        .setKvHeadsPerLayer(new int[]{2, 1, 2, 1})
        .setHeadSizes(8, 4)
        .setSlidingWindow(4)
        .setIsSWA(new boolean[]{true, false, true, false})
        .setSharedKvLayers(2)
        .setEmbeddingLengthPerLayer(6)
        .setVocabularySize(24)
        .setContextLength(16)
        .setLogitSoftcapping(30.0f)
        .build();
    Gemma4Weights denseWeights = Gemma4Weights.randomInit(denseConfig, 12345L, false, true);
    Gemma4Model denseModel = new Gemma4Model(denseConfig, null, denseWeights);

    Tensor tokenInput = new Tensor(new double[]{2, 7, 13, 4, 19, 11}, 6);
    PipelineNetwork cpuDenseNet = denseModel.toNetwork();
    PipelineNetwork cudaDenseFloatNet = denseModel.toCudaNetwork(Precision.Float);
    PipelineNetwork cudaDenseDoubleNet = denseModel.toCudaNetwork(Precision.Double);
    try {
      Result cpuRes = cpuDenseNet.eval(tokenInput.copy());
      Result gpuFloatRes = cudaDenseFloatNet.eval(tokenInput.copy());
      Result gpuDoubleRes = cudaDenseDoubleNet.eval(tokenInput.copy());
      if (CudaSystem.isEnabled()) {
        TensorList fList = gpuFloatRes.getData();
        TensorList dList = gpuDoubleRes.getData();
        Assertions.assertInstanceOf(CudaTensorList.class, fList);
        Assertions.assertInstanceOf(CudaTensorList.class, dList);
        fList.freeRef();
        dList.freeRef();
      }
      Tensor cpuOut = Result.getData0(cpuRes);
      Tensor gpuFloatOut = Result.getData0(gpuFloatRes);
      Tensor gpuDoubleOut = Result.getData0(gpuDoubleRes);
      try {
        for (int i = 0; i < cpuOut.length(); i++) {
          Assertions.assertEquals(cpuOut.getData()[i], gpuFloatOut.getData()[i], 5e-4,
              "CUDA Float Dense+PLE network mismatch at " + i);
          Assertions.assertEquals(cpuOut.getData()[i], gpuDoubleOut.getData()[i], 1e-9,
              "CUDA Double Dense+PLE network mismatch at " + i);
        }
      } finally {
        cpuOut.freeRef();
        gpuFloatOut.freeRef();
        gpuDoubleOut.freeRef();
      }

      // Verify full CUDA network serialization/deserialization
      Layer serializedCopy = cudaDenseFloatNet.copy(SerialPrecision.Float);
      Tensor copyOut = Result.getData0(serializedCopy.eval(tokenInput));
      serializedCopy.freeRef();
      Assertions.assertEquals(24 * 6, copyOut.length());
      copyOut.freeRef();
    } finally {
      cpuDenseNet.freeRef();
      cudaDenseFloatNet.freeRef();
      cudaDenseDoubleNet.freeRef();
    }

    // 2. Mixture-of-Experts (26B-A4B style) full CUDA network equivalence
    Gemma4Config moeConfig = new Gemma4Config.Builder()
        .setEmbeddingLength(16)
        .setDefaultFeedForwardLength(24)
        .setNumberOfLayers(2)
        .setNumberOfHeads(4)
        .setDefaultKvHeads(2)
        .setHeadSizes(8, 4)
        .setSlidingWindow(4)
        .setIsSWA(new boolean[]{true, false})
        .setMoE(4, 2, 10)
        .setVocabularySize(20)
        .setContextLength(16)
        .setLogitSoftcapping(20.0f)
        .build();
    Gemma4Weights moeWeights = Gemma4Weights.randomInit(moeConfig, 67890L, true, false);
    Gemma4Model moeModel = new Gemma4Model(moeConfig, null, moeWeights);

    Tensor moeTokens = new Tensor(new double[]{1, 9, 5, 14, 3}, 5);
    PipelineNetwork cpuMoeNet = moeModel.toNetwork();
    PipelineNetwork cudaMoeNet = moeModel.toCudaNetwork(Precision.Float);
    try {
      Result cpuRes = cpuMoeNet.eval(moeTokens.copy());
      Result gpuRes = cudaMoeNet.eval(moeTokens);
      if (CudaSystem.isEnabled()) {
        TensorList mList = gpuRes.getData();
        Assertions.assertInstanceOf(CudaTensorList.class, mList);
        mList.freeRef();
      }
      Tensor cpuOut = Result.getData0(cpuRes);
      Tensor gpuOut = Result.getData0(gpuRes);
      try {
        for (int i = 0; i < cpuOut.length(); i++) {
          Assertions.assertEquals(cpuOut.getData()[i], gpuOut.getData()[i], 5e-4,
              "CUDA MoE network mismatch at " + i);
        }
      } finally {
        cpuOut.freeRef();
        gpuOut.freeRef();
      }
    } finally {
      cpuMoeNet.freeRef();
      cudaMoeNet.freeRef();
    }

    // 3. End-to-end GPU training with IterativeTrainer + cuDNN MeanSqLossLayer reduces loss
    Gemma4Config trainCfg = new Gemma4Config.Builder()
        .setEmbeddingLength(12)
        .setDefaultFeedForwardLength(20)
        .setNumberOfLayers(2)
        .setNumberOfHeads(2)
        .setDefaultKvHeads(1)
        .setHeadSizes(6, 4)
        .setSlidingWindow(4)
        .setIsSWA(new boolean[]{true, false})
        .setEmbeddingLengthPerLayer(4)
        .setVocabularySize(10)
        .setContextLength(8)
        .setLogitSoftcapping(15.0f)
        .build();
    Gemma4Weights trainWeights = Gemma4Weights.randomInit(trainCfg, 55L, false, true);
    Gemma4Model trainModel = new Gemma4Model(trainCfg, null, trainWeights);
    PipelineNetwork cudaTrainNet = trainModel.toCudaNetwork(Precision.Float);

    Tensor inputTokens = new Tensor(new double[]{2, 4}, 2);
    Tensor targetLogits = new Tensor(10, 2);
    targetLogits.set(new int[]{5, 0}, 4.0);
    targetLogits.set(new int[]{7, 1}, 4.0);

    Tensor[][] trainingData = new Tensor[][]{{inputTokens, targetLogits}};
    SimpleLossNetwork supervised = new SimpleLossNetwork(cudaTrainNet, new MeanSqLossLayer());
    ArrayTrainable trainable = new ArrayTrainable(trainingData, supervised);

    PointSample initialSample = trainable.measure(null);
    double initialLoss = initialSample.sum;
    initialSample.freeRef();

    IterativeTrainer trainer = new IterativeTrainer(trainable);
    try {
      trainer.setOrientation(new GradientDescent());
      trainer.setLineSearchFactory(name -> new StaticLearningRate(0.01));
      trainer.setMaxIterations(5);
      trainer.setTimeout(30, TimeUnit.SECONDS);
      double finalLoss = trainer.run();
      Assertions.assertTrue(finalLoss < initialLoss,
          "Expected GPU fine-tuning to reduce loss: initial=" + initialLoss + ", final=" + finalLoss);
    } finally {
      trainer.freeRef();
    }
  }
}
