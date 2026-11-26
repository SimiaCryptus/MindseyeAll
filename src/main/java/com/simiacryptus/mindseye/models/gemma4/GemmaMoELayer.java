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
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.java.GeluActivationLayer;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * Differentiable Gemma 4 routed Mixture-of-Experts (MoE) branch (used in 26B-A4B models).
 * <p>
 * Implements the exact Gemma 4 expert sub-block:
 * <ol>
 *   <li>{@code moeInput = rmsnorm(x, preFfwNorm2, eps)}</li>
 *   <li>Router: {@code routerIn = rmsnorm_no_weight(x) / sqrt(dim) * ffnGateInpScale},
 *       {@code probs = softmax(ffnGateInp @ routerIn)}, top-{@code expertUsedCount} selection</li>
 *   <li>Selected experts: {@code gateUp = ffnGateUpExps[e] @ moeInput},
 *       {@code hidden = gelu(gate) * up}, {@code down = ffnDownExps[e] @ hidden},
 *       {@code moeOut += prob[e] * ffnDownExpsScale[e] * down}</li>
 *   <li>{@code output = rmsnorm(moeOut, ffnPostNorm2, eps)}</li>
 * </ol>
 */
@SuppressWarnings("serial")
public class GemmaMoELayer extends LayerBase {

  private final int dim;
  private final int expertCount;
  private final int expertUsedCount;
  private final int expertFF;
  private final double eps;

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

  public GemmaMoELayer(final int dim, final int expertCount, final int expertUsedCount,
                       final int expertFF, final double eps) {
    super();
    if (dim <= 0 || expertCount <= 0 || expertUsedCount <= 0 || expertUsedCount > expertCount || expertFF <= 0) {
      throw new IllegalArgumentException("Invalid GemmaMoELayer configuration");
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

  public GemmaMoELayer(final int dim, final int expertCount, final int expertUsedCount, final int expertFF,
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

  protected GemmaMoELayer(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> resources) {
    super(json);
    this.dim = json.get("dim").getAsInt();
    this.expertCount = json.get("expertCount").getAsInt();
    this.expertUsedCount = json.get("expertUsedCount").getAsInt();
    this.expertFF = json.get("expertFF").getAsInt();
    this.eps = json.get("eps").getAsDouble();
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
      throw new IllegalArgumentException("Missing tensor " + key + " in GemmaMoELayer JSON");
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
  public static GemmaMoELayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new GemmaMoELayer(json, rs);
  }

  @RefIgnore
  static void forwardSequence(@Nonnull final double[] xAll, @Nonnull final double[] outAll,
                              final int dim, final int expertCount, final int topK,
                              final int expertFF, final double eps,
                              @Nonnull final double[] preNormW, @Nonnull final double[] gateInpScale,
                              @Nonnull final double[] gateInpW, @Nonnull final double[] gateUpExpsW,
                              @Nonnull final double[] downExpsW, @Nonnull final double[] downExpsScale,
                              @Nonnull final double[] postNormW) {
    if (xAll.length % dim != 0) {
      throw new IllegalArgumentException("Input length " + xAll.length + " is not divisible by dim " + dim);
    }
    int seqLen = xAll.length / dim;
    int gateUpDim = 2 * expertFF;
    double invSqrtDim = 1.0 / Math.sqrt(dim);

    double[] moeInput = new double[dim];
    double[] routerIn = new double[dim];
    double[] probs = new double[expertCount];
    double[] workProbs = new double[expertCount];
    int[] topExperts = new int[topK];
    double[] topProbs = new double[topK];
    double[] gateUp = new double[gateUpDim];
    double[] hidden = new double[expertFF];
    double[] moeOut = new double[dim];

    for (int t = 0; t < seqLen; t++) {
      int base = t * dim;
      double ss = 0.0;
      for (int i = 0; i < dim; i++) {
        double xi = xAll[base + i];
        ss += xi * xi;
      }
      double rX = 1.0 / Math.sqrt(ss / dim + eps);
      double routerRms = rX * invSqrtDim;
      for (int i = 0; i < dim; i++) {
        double xi = xAll[base + i];
        moeInput[i] = preNormW[i] * rX * xi;
        routerIn[i] = xi * routerRms * gateInpScale[i];
      }

      double maxLogit = Double.NEGATIVE_INFINITY;
      for (int e = 0; e < expertCount; e++) {
        int wOff = e * dim;
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
          sum += gateInpW[wOff + i] * routerIn[i];
        }
        probs[e] = sum;
        if (sum > maxLogit) {
          maxLogit = sum;
        }
      }
      double sumExp = 0.0;
      for (int e = 0; e < expertCount; e++) {
        double ex = Math.exp(probs[e] - maxLogit);
        probs[e] = ex;
        sumExp += ex;
      }
      for (int e = 0; e < expertCount; e++) {
        probs[e] /= sumExp;
        workProbs[e] = probs[e];
      }

      for (int k = 0; k < topK; k++) {
        int bestIdx = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int e = 0; e < expertCount; e++) {
          if (workProbs[e] > bestVal) {
            bestVal = workProbs[e];
            bestIdx = e;
          }
        }
        topExperts[k] = bestIdx;
        topProbs[k] = bestVal;
        workProbs[bestIdx] = Double.NEGATIVE_INFINITY;
      }

      Arrays.fill(moeOut, 0.0);
      for (int k = 0; k < topK; k++) {
        int e = topExperts[k];
        double weight = topProbs[k] * downExpsScale[e];
        int guBase = e * gateUpDim * dim;
        for (int j = 0; j < gateUpDim; j++) {
          int rowOff = guBase + j * dim;
          double sum = 0.0;
          for (int i = 0; i < dim; i++) {
            sum += gateUpExpsW[rowOff + i] * moeInput[i];
          }
          gateUp[j] = sum;
        }
        for (int j = 0; j < expertFF; j++) {
          hidden[j] = GeluActivationLayer.gelu(gateUp[j]) * gateUp[expertFF + j];
        }
        int downBase = e * dim * expertFF;
        for (int i = 0; i < dim; i++) {
          int rowOff = downBase + i * expertFF;
          double sum = 0.0;
          for (int j = 0; j < expertFF; j++) {
            sum += downExpsW[rowOff + j] * hidden[j];
          }
          moeOut[i] += weight * sum;
        }
      }

      double ssOut = 0.0;
      for (int i = 0; i < dim; i++) {
        ssOut += moeOut[i] * moeOut[i];
      }
      double rOut = 1.0 / Math.sqrt(ssOut / dim + eps);
      for (int i = 0; i < dim; i++) {
        outAll[base + i] = postNormW[i] * rOut * moeOut[i];
      }
    }
  }

  @RefIgnore
  static void backwardSequence(@Nonnull final double[] xAll, @Nonnull final double[] dOutAll,
                               @Nullable final double[] dxAll, final boolean accumulateWeights,
                               final int dim, final int expertCount, final int topK,
                               final int expertFF, final double eps,
                               @Nonnull final double[] preNormW, @Nonnull final double[] gateInpScale,
                               @Nonnull final double[] gateInpW, @Nonnull final double[] gateUpExpsW,
                               @Nonnull final double[] downExpsW, @Nonnull final double[] downExpsScale,
                               @Nonnull final double[] postNormW,
                               @Nullable final double[] gPreNormW, @Nullable final double[] gGateInpScale,
                               @Nullable final double[] gGateInpW, @Nullable final double[] gGateUpExpsW,
                               @Nullable final double[] gDownExpsW, @Nullable final double[] gDownExpsScale,
                               @Nullable final double[] gPostNormW) {
    int seqLen = xAll.length / dim;
    int gateUpDim = 2 * expertFF;
    double invSqrtDim = 1.0 / Math.sqrt(dim);

    double[] moeInput = new double[dim];
    double[] routerIn = new double[dim];
    double[] probs = new double[expertCount];
    double[] workProbs = new double[expertCount];
    int[] topExperts = new int[topK];
    double[] topProbs = new double[topK];
    double[][] savedGateUp = new double[topK][gateUpDim];
    double[][] savedHidden = new double[topK][expertFF];
    double[][] savedDown = new double[topK][dim];
    double[] moeOut = new double[dim];
    double[] dMoeOut = new double[dim];
    double[] dMoeInput = new double[dim];
    double[] dProb = new double[expertCount];
    double[] dRouterIn = new double[dim];

    for (int t = 0; t < seqLen; t++) {
      int base = t * dim;
      double ss = 0.0;
      for (int i = 0; i < dim; i++) {
        double xi = xAll[base + i];
        ss += xi * xi;
      }
      double rX = 1.0 / Math.sqrt(ss / dim + eps);
      double routerRms = rX * invSqrtDim;
      for (int i = 0; i < dim; i++) {
        double xi = xAll[base + i];
        moeInput[i] = preNormW[i] * rX * xi;
        routerIn[i] = xi * routerRms * gateInpScale[i];
      }

      double maxLogit = Double.NEGATIVE_INFINITY;
      for (int e = 0; e < expertCount; e++) {
        int wOff = e * dim;
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
          sum += gateInpW[wOff + i] * routerIn[i];
        }
        probs[e] = sum;
        if (sum > maxLogit) {
          maxLogit = sum;
        }
      }
      double sumExp = 0.0;
      for (int e = 0; e < expertCount; e++) {
        double ex = Math.exp(probs[e] - maxLogit);
        probs[e] = ex;
        sumExp += ex;
      }
      for (int e = 0; e < expertCount; e++) {
        probs[e] /= sumExp;
        workProbs[e] = probs[e];
      }

      for (int k = 0; k < topK; k++) {
        int bestIdx = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int e = 0; e < expertCount; e++) {
          if (workProbs[e] > bestVal) {
            bestVal = workProbs[e];
            bestIdx = e;
          }
        }
        topExperts[k] = bestIdx;
        topProbs[k] = bestVal;
        workProbs[bestIdx] = Double.NEGATIVE_INFINITY;
      }

      Arrays.fill(moeOut, 0.0);
      for (int k = 0; k < topK; k++) {
        int e = topExperts[k];
        double weight = topProbs[k] * downExpsScale[e];
        int guBase = e * gateUpDim * dim;
        for (int j = 0; j < gateUpDim; j++) {
          int rowOff = guBase + j * dim;
          double sum = 0.0;
          for (int i = 0; i < dim; i++) {
            sum += gateUpExpsW[rowOff + i] * moeInput[i];
          }
          savedGateUp[k][j] = sum;
        }
        for (int j = 0; j < expertFF; j++) {
          savedHidden[k][j] = GeluActivationLayer.gelu(savedGateUp[k][j]) * savedGateUp[k][expertFF + j];
        }
        int downBase = e * dim * expertFF;
        for (int i = 0; i < dim; i++) {
          int rowOff = downBase + i * expertFF;
          double sum = 0.0;
          for (int j = 0; j < expertFF; j++) {
            sum += downExpsW[rowOff + j] * savedHidden[k][j];
          }
          savedDown[k][i] = sum;
          moeOut[i] += weight * sum;
        }
      }

      // 1. Backward through postNorm (ffnPostNorm2)
      double ssOut = 0.0;
      for (int i = 0; i < dim; i++) {
        ssOut += moeOut[i] * moeOut[i];
      }
      double rOut = 1.0 / Math.sqrt(ssOut / dim + eps);
      if (accumulateWeights && gPostNormW != null) {
        for (int i = 0; i < dim; i++) {
          gPostNormW[i] += dOutAll[base + i] * rOut * moeOut[i];
        }
      }
      double dotOut = 0.0;
      for (int i = 0; i < dim; i++) {
        dotOut += (dOutAll[base + i] * postNormW[i]) * moeOut[i];
      }
      double coeffOut = (rOut * rOut / dim) * dotOut;
      for (int i = 0; i < dim; i++) {
        double ui = dOutAll[base + i] * postNormW[i];
        dMoeOut[i] = rOut * (ui - coeffOut * moeOut[i]);
      }

      // 2. Backward through selected experts
      Arrays.fill(dMoeInput, 0.0);
      Arrays.fill(dProb, 0.0);
      for (int k = 0; k < topK; k++) {
        int e = topExperts[k];
        double prob = topProbs[k];
        double downScale = downExpsScale[e];
        double finalW = prob * downScale;

        double dotDown = 0.0;
        for (int i = 0; i < dim; i++) {
          dotDown += dMoeOut[i] * savedDown[k][i];
        }
        dProb[e] += downScale * dotDown;
        if (accumulateWeights && gDownExpsScale != null) {
          gDownExpsScale[e] += prob * dotDown;
        }

        double[] dHidden = new double[expertFF];
        int downBase = e * dim * expertFF;
        for (int i = 0; i < dim; i++) {
          double dDownI = finalW * dMoeOut[i];
          int rowOff = downBase + i * expertFF;
          if (accumulateWeights && gDownExpsW != null) {
            for (int j = 0; j < expertFF; j++) {
              gDownExpsW[rowOff + j] += dDownI * savedHidden[k][j];
            }
          }
          for (int j = 0; j < expertFF; j++) {
            dHidden[j] += dDownI * downExpsW[rowOff + j];
          }
        }

        double[] dGateUp = new double[gateUpDim];
        for (int j = 0; j < expertFF; j++) {
          double gVal = savedGateUp[k][j];
          double uVal = savedGateUp[k][expertFF + j];
          dGateUp[j] = dHidden[j] * uVal * GeluActivationLayer.geluGrad(gVal);
          dGateUp[expertFF + j] = dHidden[j] * GeluActivationLayer.gelu(gVal);
        }

        int guBase = e * gateUpDim * dim;
        for (int j = 0; j < gateUpDim; j++) {
          double dgu = dGateUp[j];
          int rowOff = guBase + j * dim;
          if (accumulateWeights && gGateUpExpsW != null) {
            for (int i = 0; i < dim; i++) {
              gGateUpExpsW[rowOff + i] += dgu * moeInput[i];
            }
          }
          for (int i = 0; i < dim; i++) {
            dMoeInput[i] += dgu * gateUpExpsW[rowOff + i];
          }
        }
      }

      // 3. Backward through router softmax & linear gateInp
      double probDot = 0.0;
      for (int e = 0; e < expertCount; e++) {
        probDot += probs[e] * dProb[e];
      }
      Arrays.fill(dRouterIn, 0.0);
      for (int e = 0; e < expertCount; e++) {
        double dLogit = probs[e] * (dProb[e] - probDot);
        int wOff = e * dim;
        if (accumulateWeights && gGateInpW != null) {
          for (int i = 0; i < dim; i++) {
            gGateInpW[wOff + i] += dLogit * routerIn[i];
          }
        }
        for (int i = 0; i < dim; i++) {
          dRouterIn[i] += dLogit * gateInpW[wOff + i];
        }
      }

      if (accumulateWeights && gGateInpScale != null) {
        for (int i = 0; i < dim; i++) {
          gGateInpScale[i] += dRouterIn[i] * xAll[base + i] * routerRms;
        }
      }

      // 4. Backward through preFfwNorm2 and router bare RMS scaling into x
      if (accumulateWeights && gPreNormW != null) {
        for (int i = 0; i < dim; i++) {
          gPreNormW[i] += dMoeInput[i] * rX * xAll[base + i];
        }
      }

      if (dxAll != null) {
        double dotPre = 0.0;
        double dotRouter = 0.0;
        for (int i = 0; i < dim; i++) {
          double xi = xAll[base + i];
          dotPre += (dMoeInput[i] * preNormW[i]) * xi;
          dotRouter += (dRouterIn[i] * gateInpScale[i]) * xi;
        }
        double coeffPre = (rX * rX / dim) * dotPre;
        double coeffRouter = (rX * rX / dim) * dotRouter;
        for (int i = 0; i < dim; i++) {
          double xi = xAll[base + i];
          double dxPre = rX * (dMoeInput[i] * preNormW[i] - coeffPre * xi);
          double dxRouter = routerRms * (dRouterIn[i] * gateInpScale[i] - coeffRouter * xi);
          dxAll[base + i] = dxPre + dxRouter;
        }
      }
    }
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (inObj.length != 1) {
      RefUtil.freeRef(inObj);
      throw new IllegalArgumentException("GemmaMoELayer requires 1 input");
    }
    Result in0 = inObj[0].addRef();
    RefUtil.freeRef(inObj);
    TensorList inputData = in0.getData();
    int batchSize = inputData.length();

    double[] preW = preFfwNorm2.getData();
    double[] gisW = ffnGateInpScale.getData();
    double[] giW = ffnGateInp.getData();
    double[] gueW = ffnGateUpExps.getData();
    double[] deW = ffnDownExps.getData();
    double[] desW = ffnDownExpsScale.getData();
    double[] postW = ffnPostNorm2.getData();

    TensorArray outputData = new TensorArray(RefIntStream.range(0, batchSize)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
          Tensor inTensor = inputData.get(b);
          Tensor outTensor = new Tensor(inTensor.getDimensions());
          forwardSequence(
              inTensor.getData(), outTensor.getData(),
              dim, expertCount, expertUsedCount, expertFF, eps,
              preW, gisW, giW, gueW, deW, desW, postW
          );
          inTensor.freeRef();
          return outTensor;
        }, inputData.addRef()))
        .toArray(Tensor[]::new));

    boolean alive = in0.isAlive();
    Result.Accumulator upstream = in0.getAccumulator();
    in0.freeRef();

    Accumulator accumulator = new Accumulator(
        inputData, dim, expertCount, expertUsedCount, expertFF, eps, isFrozen(),
        preFfwNorm2.addRef(), ffnGateInpScale.addRef(), ffnGateInp.addRef(),
        ffnGateUpExps.addRef(), ffnDownExps.addRef(), ffnDownExpsScale.addRef(), ffnPostNorm2.addRef(),
        upstream, alive
    );
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
  public GemmaMoELayer addRef() {
    return (GemmaMoELayer) super.addRef();
  }

  private static final class Accumulator extends Result.Accumulator {
    private final TensorList inputData;
    private final int dim;
    private final int expertCount;
    private final int expertUsedCount;
    private final int expertFF;
    private final double eps;
    private final boolean frozen;
    private final Tensor preFfwNorm2;
    private final Tensor ffnGateInpScale;
    private final Tensor ffnGateInp;
    private final Tensor ffnGateUpExps;
    private final Tensor ffnDownExps;
    private final Tensor ffnDownExpsScale;
    private final Tensor ffnPostNorm2;
    private final Result.Accumulator upstream;
    private final boolean alive;

    Accumulator(final TensorList inputData, final int dim, final int expertCount,
                final int expertUsedCount, final int expertFF, final double eps, final boolean frozen,
                final Tensor preFfwNorm2, final Tensor ffnGateInpScale, final Tensor ffnGateInp,
                final Tensor ffnGateUpExps, final Tensor ffnDownExps, final Tensor ffnDownExpsScale,
                final Tensor ffnPostNorm2, final Result.Accumulator upstream, final boolean alive) {
      this.inputData = inputData;
      this.dim = dim;
      this.expertCount = expertCount;
      this.expertUsedCount = expertUsedCount;
      this.expertFF = expertFF;
      this.eps = eps;
      this.frozen = frozen;
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
      int batchSize = inputData.length();
      boolean accW = !frozen && buffer != null;

      double[] gPre = accW ? new double[preFfwNorm2.length()] : null;
      double[] gGis = accW ? new double[ffnGateInpScale.length()] : null;
      double[] gGi = accW ? new double[ffnGateInp.length()] : null;
      double[] gGue = accW ? new double[ffnGateUpExps.length()] : null;
      double[] gDe = accW ? new double[ffnDownExps.length()] : null;
      double[] gDes = accW ? new double[ffnDownExpsScale.length()] : null;
      double[] gPost = accW ? new double[ffnPostNorm2.length()] : null;

      int[] inDims = inputData.getDimensions();
      int inLen = Tensor.length(inDims);
      double[][] dxBatch = alive ? new double[batchSize][inLen] : null;

      for (int b = 0; b < batchSize; b++) {
        Tensor inTensor = inputData.get(b);
        Tensor dTensor = delta.get(b);
        backwardSequence(
            inTensor.getData(), dTensor.getData(),
            alive ? dxBatch[b] : null, accW,
            dim, expertCount, expertUsedCount, expertFF, eps,
            preFfwNorm2.getData(), ffnGateInpScale.getData(), ffnGateInp.getData(),
            ffnGateUpExps.getData(), ffnDownExps.getData(), ffnDownExpsScale.getData(), ffnPostNorm2.getData(),
            gPre, gGis, gGi, gGue, gDe, gDes, gPost
        );
        dTensor.freeRef();
        inTensor.freeRef();
      }

      if (accW) {
        accumulateParam(buffer, preFfwNorm2, gPre);
        accumulateParam(buffer, ffnGateInpScale, gGis);
        accumulateParam(buffer, ffnGateInp, gGi);
        accumulateParam(buffer, ffnGateUpExps, gGue);
        accumulateParam(buffer, ffnDownExps, gDe);
        accumulateParam(buffer, ffnDownExpsScale, gDes);
        accumulateParam(buffer, ffnPostNorm2, gPost);
      }

      if (alive) {
        TensorArray passback = new TensorArray(RefIntStream.range(0, batchSize)
            .mapToObj(b -> new Tensor(dxBatch[b], inDims))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> bufCopy = buffer == null ? null : buffer.addRef();
        upstream.accept(bufCopy, passback);
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
