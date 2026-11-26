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
import com.simiacryptus.mindseye.lang.Delta;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.java.ChunkSelectLayer;
import com.simiacryptus.mindseye.layers.java.GeluActivationLayer;
import com.simiacryptus.mindseye.layers.java.GroupedQueryAttentionLayer;
import com.simiacryptus.mindseye.layers.java.LinearProjectionLayer;
import com.simiacryptus.mindseye.layers.java.RmsNormLayer;
import com.simiacryptus.mindseye.layers.java.RopeLayer;
import com.simiacryptus.mindseye.layers.java.ScalarScaleLayer;
import com.simiacryptus.mindseye.layers.java.SoftcapLayer;
import com.simiacryptus.mindseye.layers.java.TokenEmbeddingLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.ToleranceStatistics;
import com.simiacryptus.mindseye.test.unit.SingleDerivativeTester;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefList;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

/**
 * Finite-difference derivative, serialization, and reference-counting tests for all Gemma 4 differentiable CPU layers.
 */
@RefIgnore
public class Gemma4LayersTest {

  private static Tensor randomTensor(final long seed, final int... dims) {
    Random rng = new Random(seed);
    int len = Tensor.length(dims);
    double[] data = new double[len];
    for (int i = 0; i < len; i++) {
      data[i] = (rng.nextDouble() * 2.0 - 1.0) * 0.8;
    }
    return new Tensor(data, dims);
  }

  private static void verifyDerivativesAndJson(final Layer layer, final Tensor... inputs) {
    try {
      // 1. JSON serialization round-trip equivalence
      Map<CharSequence, byte[]> resources = new HashMap<>();
      JsonObject json = layer.getJson(resources, Tensor.json_precision).getAsJsonObject();
      Layer restored = Layer.fromJson(json, resources);
      try {
        Result origRes = layer.eval(copyTensors(inputs));
        Result restRes = restored.eval(copyTensors(inputs));
        Tensor origOut = Result.getData0(origRes);
        Tensor restOut = Result.getData0(restRes);
        try {
          double[] a = origOut.getData();
          double[] b = restOut.getData();
          Assertions.assertEquals(a.length, b.length);
          for (int i = 0; i < a.length; i++) {
            Assertions.assertEquals(a[i], b[i], 1e-4, "Serialization output mismatch at " + i);
          }
        } finally {
          origOut.freeRef();
          restOut.freeRef();
        }
      } finally {
        restored.freeRef();
      }

      // 2. MindsEye SingleDerivativeTester validation (feedback + weights + frozen/unfrozen)
      SingleDerivativeTester tester = new SingleDerivativeTester(1e-2, 1e-5);
      tester.setVerbose(false);
      NullNotebookOutput log = new NullNotebookOutput();
      ToleranceStatistics stats = tester.test(log, layer.addRef(), copyTensors(inputs));
      tester.freeRef();
      Assertions.assertNotNull(stats);
    } finally {
      layer.freeRef();
      RefUtil.freeRef(inputs);
    }
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

  /**
   * High-precision central finite-difference check (< 1e-6 error) for both input and weight gradients.
   */
  private static void assertCentralDifferenceExact(final Layer layer, final boolean checkInputGrad, final Tensor... inputs) {
    try {
      Result fwd = layer.eval(copyTensors(inputs));
      Tensor fwdOut = Result.getData0(fwd.addRef());
      int outLen = fwdOut.length();
      int[] outDims = fwdOut.getDimensions();
      fwdOut.freeRef();
      fwd.freeRef();

      Tensor seedTensor = randomTensor(999L, outLen);
      double[] upstreamSeed = seedTensor.copyData();
      seedTensor.freeRef();
      double[][] analyticalInputGrads = new double[inputs.length][];
      Result[] wrappedInputs = new Result[inputs.length];
      for (int idx = 0; idx < inputs.length; idx++) {
        final int slot = idx;
        analyticalInputGrads[slot] = new double[inputs[slot].length()];
        Result.Accumulator acc = new Result.Accumulator() {
          @Override
          public void accept(final DeltaSet<UUID> buffer, final TensorList data) {
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
      DeltaSet<UUID> deltaSet = new DeltaSet<>();
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
            Assertions.assertEquals(numGrad, analyticalInputGrads[idx][i], 1e-5,
                layer.getClass().getSimpleName() + " input[" + idx + "] grad mismatch at " + i);
          }
        }
      }

      if (!layer.isFrozen()) {
        RefList<double[]> states = layer.state();
        for (int s = 0; s < states.size(); s++) {
          double[] param = states.get(s);
          double[] analyticalParamGrad = null;
          RefList<Delta<UUID>> allDeltas = deltaSet.stream().collect(com.simiacryptus.ref.wrappers.RefCollectors.toList());
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
            Assertions.assertEquals(numGrad, analyticalParamGrad[i], 1e-5,
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

  @Test
  @Tag("UnitTest")
  public void testRmsNormGeluSoftcapAndScalarScaleDerivatives() {
    RmsNormLayer rmsLearned = new RmsNormLayer(4, 1e-5, true);
    rmsLearned.setWeights(new double[]{1.2, 0.8, -0.5, 1.1});
    assertCentralDifferenceExact(rmsLearned.addRef(), true, randomTensor(11L, 8, 3));
    verifyDerivativesAndJson(rmsLearned, randomTensor(12L, 8, 2));

    RmsNormLayer rmsBare = new RmsNormLayer(4, 1e-5, false);
    assertCentralDifferenceExact(rmsBare.addRef(), true, randomTensor(13L, 8, 2));
    verifyDerivativesAndJson(rmsBare, randomTensor(14L, 8, 2));

    GeluActivationLayer gelu = new GeluActivationLayer();
    assertCentralDifferenceExact(gelu.addRef(), true, randomTensor(15L, 6, 2));
    verifyDerivativesAndJson(gelu, randomTensor(16L, 6, 2));

    SoftcapLayer softcap = new SoftcapLayer(5.0);
    assertCentralDifferenceExact(softcap.addRef(), true, randomTensor(17L, 6, 2));
    verifyDerivativesAndJson(softcap, randomTensor(18L, 6, 2));

    ScalarScaleLayer scale = new ScalarScaleLayer(0.75, false);
    assertCentralDifferenceExact(scale.addRef(), true, randomTensor(19L, 5, 2));
    verifyDerivativesAndJson(scale, randomTensor(20L, 5, 2));
  }

  @Test
  @Tag("UnitTest")
  public void testLinearProjectionRopeChunkSelectAndEmbeddingDerivatives() {
    LinearProjectionLayer proj = new LinearProjectionLayer(6, 4);
    assertCentralDifferenceExact(proj.addRef(), true, randomTensor(21L, 6, 3));
    verifyDerivativesAndJson(proj, randomTensor(22L, 6, 3));

    RopeLayer ropeWithFactors = new RopeLayer(2, 4, 10000.0, new double[]{1.0, 2.0}, 3);
    assertCentralDifferenceExact(ropeWithFactors.addRef(), true, randomTensor(23L, 8, 3));
    verifyDerivativesAndJson(ropeWithFactors, randomTensor(24L, 8, 3));

    ChunkSelectLayer slice = new ChunkSelectLayer(3, 4, 2);
    assertCentralDifferenceExact(slice.addRef(), true, randomTensor(25L, 12, 3));
    verifyDerivativesAndJson(slice, randomTensor(26L, 12, 3));

    // TokenEmbeddingLayer + tied LM head shared weight gradient check
    Tensor sharedTable = randomTensor(27L, 4, 8);
    PipelineNetwork tiedNet = new PipelineNetwork(1);
    DAGNode tokIn = tiedNet.getInput(0);
    com.simiacryptus.mindseye.network.DAGNode emb = tiedNet.add(
        new TokenEmbeddingLayer(8, 4, 2.0, sharedTable.addRef()), tokIn);
    com.simiacryptus.mindseye.network.DAGNode logits = tiedNet.add(
        new LinearProjectionLayer(4, 8, sharedTable), emb);
    tiedNet.setHead(logits);
    assertCentralDifferenceExact(tiedNet, false, new Tensor(new double[]{1, 5, 2}, 3));
  }

  @Test
  @Tag("UnitTest")
  public void testGroupedQueryAttentionAndMoEDerivatives() {
    // GQA causal + slidingWindow=2
    GroupedQueryAttentionLayer gqaSwa = new GroupedQueryAttentionLayer(4, 2, 4, 1.0, true, 2);
    assertCentralDifferenceExact(
        gqaSwa.addRef(),
        true,
        randomTensor(31L, 16, 4),
        randomTensor(32L, 8, 4),
        randomTensor(33L, 8, 4)
    );
    verifyDerivativesAndJson(
        gqaSwa,
        randomTensor(34L, 16, 3),
        randomTensor(35L, 8, 3),
        randomTensor(36L, 8, 3)
    );

    // GemmaMoELayer: routed top-2 out of 4 experts across sequence length 2
    GemmaMoELayer moe = new GemmaMoELayer(6, 4, 2, 5, 1e-5);
    assertCentralDifferenceExact(moe.addRef(), true, randomTensor(41L, 6, 2));

    Map<CharSequence, byte[]> rs = new HashMap<>();
    JsonObject moeJson = moe.getJson(rs, Tensor.json_precision).getAsJsonObject();
    Layer moeRestored = Layer.fromJson(moeJson, rs);
    Tensor sampleIn = randomTensor(42L, 6, 2);
    Tensor out1 = Result.getData0(moe.eval(sampleIn.copy()));
    Tensor out2 = Result.getData0(moeRestored.eval(sampleIn));
    for (int i = 0; i < out1.length(); i++) {
      Assertions.assertEquals(out1.getData()[i], out2.getData()[i], 1e-4);
    }
    out1.freeRef();
    out2.freeRef();
    moeRestored.freeRef();
    moe.freeRef();
  }
}
