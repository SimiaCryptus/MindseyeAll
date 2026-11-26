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

import com.simiacryptus.ref.lang.RefIgnore;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.function.IntFunction;

/**
 * Container and loader for all Gemma 4 model parameters (dense, SWA, shared KV, V=K, PLE, and MoE).
 */
@RefIgnore
public final class Gemma4Weights {

  public final GGUFTensor tokenEmbeddingTable;
  public final float[][] rmsAttWeight;
  public final GGUFTensor[] wq;
  public final GGUFTensor[] wk;
  public final GGUFTensor[] wv;
  public final GGUFTensor[] wo;
  public final float[][] attnQNorm;
  public final float[][] attnKNorm;
  public final float[][] postAttentionNorm;
  public final float[][] rmsFfnWeight;
  public final GGUFTensor[] w1;
  public final GGUFTensor[] w2;
  public final GGUFTensor[] w3;
  public final float[][] postFfwNorm;
  public final float[] rmsFinalWeight;
  public final float[] layerOutputScale;
  @Nullable
  public final float[] ropeFreqFactors;
  public final float[] freqCisRealFull;
  public final float[] freqCisImagFull;
  public final float[] freqCisRealSwa;
  public final float[] freqCisImagSwa;
  public final GGUFTensor wcls;

  // Per-layer embedding weights (nullable)
  @Nullable
  public final GGUFTensor perLayerTokenEmbd;
  @Nullable
  public final GGUFTensor perLayerModelProj;
  @Nullable
  public final float[] perLayerProjNorm;
  @Nullable
  public final GGUFTensor[] perLayerInpGate;
  @Nullable
  public final GGUFTensor[] perLayerProj;
  @Nullable
  public final float[][] perLayerPostNorm;

  // MoE weights (nullable when dense)
  @Nullable
  public final GGUFTensor[] ffnGateInp;
  @Nullable
  public final float[][] ffnGateInpScale;
  @Nullable
  public final GGUFTensor[] ffnGateUpExps;
  @Nullable
  public final GGUFTensor[] ffnDownExps;
  @Nullable
  public final float[][] ffnDownExpsScale;
  @Nullable
  public final float[][] ffnPostNorm1;
  @Nullable
  public final float[][] preFfwNorm2;
  @Nullable
  public final float[][] ffnPostNorm2;

  public Gemma4Weights(@Nonnull final GGUFTensor tokenEmbeddingTable,
                       @Nonnull final float[][] rmsAttWeight,
                       @Nonnull final GGUFTensor[] wq,
                       @Nonnull final GGUFTensor[] wk,
                       @Nonnull final GGUFTensor[] wv,
                       @Nonnull final GGUFTensor[] wo,
                       @Nonnull final float[][] attnQNorm,
                       @Nonnull final float[][] attnKNorm,
                       @Nonnull final float[][] postAttentionNorm,
                       @Nonnull final float[][] rmsFfnWeight,
                       @Nonnull final GGUFTensor[] w1,
                       @Nonnull final GGUFTensor[] w2,
                       @Nonnull final GGUFTensor[] w3,
                       @Nonnull final float[][] postFfwNorm,
                       @Nonnull final float[] rmsFinalWeight,
                       @Nonnull final float[] layerOutputScale,
                       @Nullable final float[] ropeFreqFactors,
                       @Nonnull final float[] freqCisRealFull,
                       @Nonnull final float[] freqCisImagFull,
                       @Nonnull final float[] freqCisRealSwa,
                       @Nonnull final float[] freqCisImagSwa,
                       @Nonnull final GGUFTensor wcls,
                       @Nullable final GGUFTensor perLayerTokenEmbd,
                       @Nullable final GGUFTensor perLayerModelProj,
                       @Nullable final float[] perLayerProjNorm,
                       @Nullable final GGUFTensor[] perLayerInpGate,
                       @Nullable final GGUFTensor[] perLayerProj,
                       @Nullable final float[][] perLayerPostNorm,
                       @Nullable final GGUFTensor[] ffnGateInp,
                       @Nullable final float[][] ffnGateInpScale,
                       @Nullable final GGUFTensor[] ffnGateUpExps,
                       @Nullable final GGUFTensor[] ffnDownExps,
                       @Nullable final float[][] ffnDownExpsScale,
                       @Nullable final float[][] ffnPostNorm1,
                       @Nullable final float[][] preFfwNorm2,
                       @Nullable final float[][] ffnPostNorm2) {
    this.tokenEmbeddingTable = tokenEmbeddingTable;
    this.rmsAttWeight = rmsAttWeight;
    this.wq = wq;
    this.wk = wk;
    this.wv = wv;
    this.wo = wo;
    this.attnQNorm = attnQNorm;
    this.attnKNorm = attnKNorm;
    this.postAttentionNorm = postAttentionNorm;
    this.rmsFfnWeight = rmsFfnWeight;
    this.w1 = w1;
    this.w2 = w2;
    this.w3 = w3;
    this.postFfwNorm = postFfwNorm;
    this.rmsFinalWeight = rmsFinalWeight;
    this.layerOutputScale = layerOutputScale;
    this.ropeFreqFactors = ropeFreqFactors;
    this.freqCisRealFull = freqCisRealFull;
    this.freqCisImagFull = freqCisImagFull;
    this.freqCisRealSwa = freqCisRealSwa;
    this.freqCisImagSwa = freqCisImagSwa;
    this.wcls = wcls;
    this.perLayerTokenEmbd = perLayerTokenEmbd;
    this.perLayerModelProj = perLayerModelProj;
    this.perLayerProjNorm = perLayerProjNorm;
    this.perLayerInpGate = perLayerInpGate;
    this.perLayerProj = perLayerProj;
    this.perLayerPostNorm = perLayerPostNorm;
    this.ffnGateInp = ffnGateInp;
    this.ffnGateInpScale = ffnGateInpScale;
    this.ffnGateUpExps = ffnGateUpExps;
    this.ffnDownExps = ffnDownExps;
    this.ffnDownExpsScale = ffnDownExpsScale;
    this.ffnPostNorm1 = ffnPostNorm1;
    this.preFfwNorm2 = preFfwNorm2;
    this.ffnPostNorm2 = ffnPostNorm2;
  }

  public boolean hasTiedEmbeddings() {
    return wcls == tokenEmbeddingTable;
  }

  @Nonnull
  public static float[][] precomputeFreqsCis(final int contextLength, final int headSize, final double theta) {
    int halfHead = headSize / 2;
    float[] cr = new float[contextLength * halfHead];
    float[] ci = new float[contextLength * halfHead];
    int n = 0;
    for (int pos = 0; pos < contextLength; ++pos) {
      for (int i = 0; i < headSize; i += 2) {
        float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
        float val = pos * freq;
        cr[n] = (float) Math.cos(val);
        ci[n] = (float) Math.sin(val);
        n++;
      }
    }
    return new float[][]{cr, ci};
  }

  @Nonnull
  public static float[][] precomputeFreqsCisFromFreqs(final int contextLength, final int headSize,
                                                      final double ropeTheta, @Nonnull final float[] ropeFreqFactors) {
    int halfHead = headSize / 2;
    if (ropeFreqFactors.length != halfHead) {
      throw new IllegalArgumentException("ropeFreqFactors length " + ropeFreqFactors.length + " != " + halfHead);
    }
    float[] cr = new float[contextLength * halfHead];
    float[] ci = new float[contextLength * halfHead];
    int n = 0;
    for (int pos = 0; pos < contextLength; ++pos) {
      for (int i = 0; i < halfHead; i++) {
        float baseFreq = (float) (1.0 / Math.pow(ropeTheta, (2.0 * i) / headSize));
        float val = pos * baseFreq / ropeFreqFactors[i];
        cr[n] = (float) Math.cos(val);
        ci[n] = (float) Math.sin(val);
        n++;
      }
    }
    return new float[][]{cr, ci};
  }

  @Nonnull
  public static Gemma4Weights fromGGUF(@Nonnull final Map<String, GGUF.GGMLTensorEntry> tensorEntries,
                                       @Nonnull final Gemma4Config config) {
    float[][] ropeSwa = precomputeFreqsCis(config.contextLength, config.headSizeSWA, config.ropeThetaSWA);
    GGUF.GGMLTensorEntry ropeFreqEntry = tensorEntries.get("rope_freqs.weight");
    float[] ropeFreqFactors = null;
    float[][] ropeFull;
    if (ropeFreqEntry != null) {
      ropeFreqFactors = toFloatArray(ropeFreqEntry);
      ropeFull = precomputeFreqsCisFromFreqs(config.contextLength, config.headSizeFull, config.ropeTheta, ropeFreqFactors);
    } else {
      ropeFull = precomputeFreqsCis(config.contextLength, config.headSizeFull, config.ropeTheta);
    }

    int nLayers = config.numberOfLayers;
    GGUFTensor tokenEmbeddingTable = loadRequired(tensorEntries, "token_embd.weight");

    float[] layerOutputScale = new float[nLayers];
    for (int i = 0; i < nLayers; i++) {
      GGUF.GGMLTensorEntry scaleEntry = tensorEntries.get("blk." + i + ".layer_output_scale.weight");
      layerOutputScale[i] = scaleEntry != null ? toFloatArray(scaleEntry)[0] : 1.0f;
    }

    GGUFTensor perLayerTokenEmbd = null;
    GGUFTensor perLayerModelProj = null;
    float[] perLayerProjNorm = null;
    GGUFTensor[] perLayerInpGate = null;
    GGUFTensor[] perLayerProj = null;
    float[][] perLayerPostNorm = null;
    if (config.embeddingLengthPerLayer > 0 && tensorEntries.containsKey("per_layer_token_embd.weight")) {
      perLayerTokenEmbd = loadRequired(tensorEntries, "per_layer_token_embd.weight");
      perLayerModelProj = loadRequired(tensorEntries, "per_layer_model_proj.weight");
      perLayerProjNorm = toFloatArray(requireEntry(tensorEntries, "per_layer_proj_norm.weight"));
      perLayerInpGate = loadOptionalQuantizedArray(nLayers, i -> tensorEntries.get("blk." + i + ".inp_gate.weight"));
      perLayerProj = loadOptionalQuantizedArray(nLayers, i -> tensorEntries.get("blk." + i + ".proj.weight"));
      perLayerPostNorm = loadOptionalFloatArray(nLayers, i -> tensorEntries.get("blk." + i + ".post_norm.weight"));
    }

    GGUFTensor[] wk = loadOptionalQuantizedArray(nLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight"));
    GGUFTensor[] wv = loadOptionalQuantizedArray(nLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight"));
    float[][] attnKNorm = loadOptionalFloatArray(nLayers, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight"));

    GGUFTensor[] ffnGateInp = null;
    float[][] ffnGateInpScale = null;
    GGUFTensor[] ffnGateUpExps = null;
    GGUFTensor[] ffnDownExps = null;
    float[][] ffnDownExpsScale = null;
    float[][] ffnPostNorm1 = null;
    float[][] preFfwNorm2 = null;
    float[][] ffnPostNorm2 = null;

    if (config.isMoE()) {
      ffnGateInp = loadOptionalQuantizedArray(nLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.weight"));
      ffnGateInpScale = loadOptionalFloatArray(nLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.scale"));
      ffnGateUpExps = loadOptionalQuantizedArray(nLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate_up_exps.weight"));
      ffnDownExps = loadOptionalQuantizedArray(nLayers, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight"));
      ffnDownExpsScale = loadOptionalFloatArray(nLayers, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.scale"));
      ffnPostNorm1 = loadOptionalFloatArray(nLayers, i -> tensorEntries.get("blk." + i + ".post_ffw_norm_1.weight"));
      preFfwNorm2 = loadOptionalFloatArray(nLayers, i -> tensorEntries.get("blk." + i + ".pre_ffw_norm_2.weight"));
      ffnPostNorm2 = loadOptionalFloatArray(nLayers, i -> tensorEntries.get("blk." + i + ".post_ffw_norm_2.weight"));
    }

    GGUFTensor wcls = tensorEntries.containsKey("output.weight")
        ? loadRequired(tensorEntries, "output.weight")
        : tokenEmbeddingTable;

    return new Gemma4Weights(
        tokenEmbeddingTable,
        loadRequiredFloatArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".attn_norm.weight")),
        loadRequiredQuantizedArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".attn_q.weight")),
        wk,
        wv,
        loadRequiredQuantizedArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".attn_output.weight")),
        loadRequiredFloatArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".attn_q_norm.weight")),
        attnKNorm,
        loadRequiredFloatArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".post_attention_norm.weight")),
        loadRequiredFloatArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".ffn_norm.weight")),
        loadRequiredQuantizedArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".ffn_gate.weight")),
        loadRequiredQuantizedArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".ffn_down.weight")),
        loadRequiredQuantizedArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".ffn_up.weight")),
        loadRequiredFloatArray(nLayers, i -> requireEntry(tensorEntries, "blk." + i + ".post_ffw_norm.weight")),
        toFloatArray(requireEntry(tensorEntries, "output_norm.weight")),
        layerOutputScale,
        ropeFreqFactors,
        ropeFull[0], ropeFull[1],
        ropeSwa[0], ropeSwa[1],
        wcls,
        perLayerTokenEmbd, perLayerModelProj, perLayerProjNorm,
        perLayerInpGate, perLayerProj, perLayerPostNorm,
        ffnGateInp, ffnGateInpScale, ffnGateUpExps, ffnDownExps, ffnDownExpsScale,
        ffnPostNorm1, preFfwNorm2, ffnPostNorm2
    );
  }

  /**
   * Initializes deterministic random weights for testing, training from scratch, or synthetic GGUF generation.
   */
  @Nonnull
  public static Gemma4Weights randomInit(@Nonnull final Gemma4Config config, final long seed,
                                         final boolean includeAllVWeights, final boolean tiedEmbeddings) {
    Random rng = new Random(seed);
    int nLayers = config.numberOfLayers;
    int dim = config.embeddingLength;

    GGUFTensor tokenEmbd = randomTensor(rng, config.vocabularySize * dim, 0.08f);
    GGUFTensor wcls = tiedEmbeddings ? tokenEmbd : randomTensor(rng, config.vocabularySize * dim, 0.08f);

    float[][] rmsAtt = new float[nLayers][];
    GGUFTensor[] wq = new GGUFTensor[nLayers];
    GGUFTensor[] wk = new GGUFTensor[nLayers];
    GGUFTensor[] wv = new GGUFTensor[nLayers];
    GGUFTensor[] wo = new GGUFTensor[nLayers];
    float[][] qNorm = new float[nLayers][];
    float[][] kNorm = new float[nLayers][];
    float[][] postAttNorm = new float[nLayers][];
    float[][] rmsFfn = new float[nLayers][];
    GGUFTensor[] w1 = new GGUFTensor[nLayers];
    GGUFTensor[] w2 = new GGUFTensor[nLayers];
    GGUFTensor[] w3 = new GGUFTensor[nLayers];
    float[][] postFfwNorm = new float[nLayers][];
    float[] layerOutputScale = new float[nLayers];

    for (int l = 0; l < nLayers; l++) {
      int qDim = config.queryDim(l);
      int kvDim = config.kvDim(l);
      int hSize = config.headSize(l);
      int ffnDim = config.feedForwardLength[l];

      rmsAtt[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
      wq[l] = randomTensor(rng, qDim * dim, 0.12f);
      qNorm[l] = randomPositiveArray(rng, hSize, 0.8f, 1.2f);
      if (config.hasKv(l)) {
        wk[l] = randomTensor(rng, kvDim * dim, 0.12f);
        wv[l] = (includeAllVWeights || (l % 2 == 0)) ? randomTensor(rng, kvDim * dim, 0.12f) : null;
        kNorm[l] = randomPositiveArray(rng, hSize, 0.8f, 1.2f);
      }
      wo[l] = randomTensor(rng, dim * qDim, 0.12f);
      postAttNorm[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
      rmsFfn[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
      w1[l] = randomTensor(rng, ffnDim * dim, 0.12f);
      w2[l] = randomTensor(rng, dim * ffnDim, 0.12f);
      w3[l] = randomTensor(rng, ffnDim * dim, 0.12f);
      postFfwNorm[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
      layerOutputScale[l] = 0.9f + 0.1f * rng.nextFloat();
    }

    float[] rmsFinal = randomPositiveArray(rng, dim, 0.8f, 1.2f);
    float[] ropeFreqFactors = new float[config.headSizeFull / 2];
    for (int i = 0; i < ropeFreqFactors.length; i++) {
      ropeFreqFactors[i] = 1.0f + 0.25f * i;
    }
    float[][] ropeFull = precomputeFreqsCisFromFreqs(config.contextLength, config.headSizeFull, config.ropeTheta, ropeFreqFactors);
    float[][] ropeSwa = precomputeFreqsCis(config.contextLength, config.headSizeSWA, config.ropeThetaSWA);

    GGUFTensor plTok = null;
    GGUFTensor plModelProj = null;
    float[] plProjNorm = null;
    GGUFTensor[] plInpGate = null;
    GGUFTensor[] plProj = null;
    float[][] plPostNorm = null;
    if (config.embeddingLengthPerLayer > 0) {
      int plDim = config.embeddingLengthPerLayer;
      int plTotal = plDim * nLayers;
      plTok = randomTensor(rng, config.vocabularySize * plTotal, 0.08f);
      plModelProj = randomTensor(rng, plTotal * dim, 0.12f);
      plProjNorm = randomPositiveArray(rng, plDim, 0.8f, 1.2f);
      plInpGate = new GGUFTensor[nLayers];
      plProj = new GGUFTensor[nLayers];
      plPostNorm = new float[nLayers][];
      for (int l = 0; l < nLayers; l++) {
        plInpGate[l] = randomTensor(rng, plDim * dim, 0.12f);
        plProj[l] = randomTensor(rng, dim * plDim, 0.12f);
        plPostNorm[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
      }
    }

    GGUFTensor[] ffnGateInp = null;
    float[][] ffnGateInpScale = null;
    GGUFTensor[] ffnGateUpExps = null;
    GGUFTensor[] ffnDownExps = null;
    float[][] ffnDownExpsScale = null;
    float[][] ffnPostNorm1 = null;
    float[][] preFfwNorm2 = null;
    float[][] ffnPostNorm2 = null;
    if (config.isMoE()) {
      int nExp = config.expertCount;
      int expFF = config.expertFeedForwardLength;
      ffnGateInp = new GGUFTensor[nLayers];
      ffnGateInpScale = new float[nLayers][];
      ffnGateUpExps = new GGUFTensor[nLayers];
      ffnDownExps = new GGUFTensor[nLayers];
      ffnDownExpsScale = new float[nLayers][];
      ffnPostNorm1 = new float[nLayers][];
      preFfwNorm2 = new float[nLayers][];
      ffnPostNorm2 = new float[nLayers][];
      for (int l = 0; l < nLayers; l++) {
        ffnGateInp[l] = randomTensor(rng, nExp * dim, 0.25f);
        ffnGateInpScale[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
        ffnGateUpExps[l] = randomTensor(rng, nExp * 2 * expFF * dim, 0.12f);
        ffnDownExps[l] = randomTensor(rng, nExp * dim * expFF, 0.12f);
        ffnDownExpsScale[l] = randomPositiveArray(rng, nExp, 0.8f, 1.2f);
        ffnPostNorm1[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
        preFfwNorm2[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
        ffnPostNorm2[l] = randomPositiveArray(rng, dim, 0.8f, 1.2f);
      }
    }

    return new Gemma4Weights(
        tokenEmbd, rmsAtt, wq, wk, wv, wo, qNorm, kNorm, postAttNorm, rmsFfn,
        w1, w2, w3, postFfwNorm, rmsFinal, layerOutputScale,
        ropeFreqFactors, ropeFull[0], ropeFull[1], ropeSwa[0], ropeSwa[1], wcls,
        plTok, plModelProj, plProjNorm, plInpGate, plProj, plPostNorm,
        ffnGateInp, ffnGateInpScale, ffnGateUpExps, ffnDownExps, ffnDownExpsScale,
        ffnPostNorm1, preFfwNorm2, ffnPostNorm2
    );
  }

  private static GGUFTensor.ArrayFloatTensor randomTensor(final Random rng, final int size, final float scale) {
    float[] arr = new float[size];
    for (int i = 0; i < size; i++) {
      arr[i] = (rng.nextFloat() * 2f - 1f) * scale;
    }
    return new GGUFTensor.ArrayFloatTensor(arr);
  }

  private static float[] randomPositiveArray(final Random rng, final int size, final float min, final float max) {
    float[] arr = new float[size];
    for (int i = 0; i < size; i++) {
      arr[i] = min + rng.nextFloat() * (max - min);
    }
    return arr;
  }

  private static GGUF.GGMLTensorEntry requireEntry(final Map<String, GGUF.GGMLTensorEntry> map, final String name) {
    GGUF.GGMLTensorEntry entry = map.get(name);
    if (entry == null) {
      throw new IllegalArgumentException("Missing required GGUF tensor: " + name);
    }
    return entry;
  }

  private static GGUFTensor loadRequired(final Map<String, GGUF.GGMLTensorEntry> map, final String name) {
    return GGUFTensor.fromEntry(requireEntry(map, name));
  }

  private static GGUFTensor[] loadRequiredQuantizedArray(final int size, final IntFunction<GGUF.GGMLTensorEntry> fn) {
    GGUFTensor[] arr = new GGUFTensor[size];
    for (int i = 0; i < size; i++) {
      arr[i] = GGUFTensor.fromEntry(fn.apply(i));
    }
    return arr;
  }

  private static GGUFTensor[] loadOptionalQuantizedArray(final int size, final IntFunction<GGUF.GGMLTensorEntry> fn) {
    GGUFTensor[] arr = new GGUFTensor[size];
    for (int i = 0; i < size; i++) {
      GGUF.GGMLTensorEntry e = fn.apply(i);
      arr[i] = e != null ? GGUFTensor.fromEntry(e) : null;
    }
    return arr;
  }

  private static float[][] loadRequiredFloatArray(final int size, final IntFunction<GGUF.GGMLTensorEntry> fn) {
    float[][] arr = new float[size][];
    for (int i = 0; i < size; i++) {
      arr[i] = toFloatArray(fn.apply(i));
    }
    return arr;
  }

  private static float[][] loadOptionalFloatArray(final int size, final IntFunction<GGUF.GGMLTensorEntry> fn) {
    float[][] arr = new float[size][];
    for (int i = 0; i < size; i++) {
      GGUF.GGMLTensorEntry e = fn.apply(i);
      arr[i] = e != null ? toFloatArray(e) : null;
    }
    return arr;
  }

  private static float[] toFloatArray(@Nonnull final GGUF.GGMLTensorEntry entry) {
    FloatBuffer buf = entry.toFloatBuffer();
    float[] out = new float[buf.remaining()];
    int pos = buf.position();
    buf.get(out);
    buf.position(pos);
    return out;
  }

  @Nullable
  public double[] getRopeFreqFactorsAsDouble() {
    if (ropeFreqFactors == null) {
      return null;
    }
    double[] out = new double[ropeFreqFactors.length];
    for (int i = 0; i < out.length; i++) {
      out[i] = ropeFreqFactors[i];
    }
    return out;
  }
}
