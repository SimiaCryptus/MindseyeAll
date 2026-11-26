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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefIgnore;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

/**
 * High-level Gemma 4 model combining:
 * <ul>
 *   <li>GGUF model loading (all quantizations: F32, F16, BF16, Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, Q6_K, Q8_0, MXFP4)</li>
 *   <li>Conversion to a differentiable MindsEye {@link PipelineNetwork} ({@link #toNetwork()}) for training/backprop</li>
 *   <li>Fast stateful KV-cached autoregressive token generation &amp; CLI ({@link #forward}, {@link #generateTokens}, {@link #main})</li>
 * </ul>
 */
@RefIgnore
public final class Gemma4Model {

  public static final int DEFAULT_MAX_TOKENS = 1024;
  private static final String ANSI_CYAN = "\033[36m";
  private static final String ANSI_GREY = "\033[90m";
  private static final String ANSI_RESET = "\033[0m";

  private final Gemma4Config config;
  @Nullable
  private final GemmaTokenizer tokenizer;
  private final Gemma4Weights weights;

  public Gemma4Model(@Nonnull final Gemma4Config config,
                     @Nullable final GemmaTokenizer tokenizer,
                     @Nonnull final Gemma4Weights weights) {
    this.config = config;
    this.tokenizer = tokenizer;
    this.weights = weights;
  }

  @Nonnull
  public Gemma4Config configuration() {
    return config;
  }

  @Nullable
  public GemmaTokenizer tokenizer() {
    return tokenizer;
  }

  @Nonnull
  public Gemma4Weights weights() {
    return weights;
  }

  /**
   * Builds a differentiable MindsEye {@link PipelineNetwork} representing this Gemma 4 model.
   */
  @Nonnull
  public PipelineNetwork toNetwork() {
    return Gemma4Network.buildNetwork(config, weights);
  }

  /**
   * Builds a differentiable MindsEye {@link PipelineNetwork} starting from continuous embeddings.
   */
  @Nonnull
  public PipelineNetwork toEmbeddingNetwork() {
    return Gemma4Network.buildFromEmbeddingsNetwork(config, weights);
  }

  /**
   * Builds a CUDA/cuDNN-accelerated differentiable MindsEye {@link PipelineNetwork} representing this Gemma 4 model.
   */
  @Nonnull
  public PipelineNetwork toCudaNetwork() {
    return Gemma4Network.buildCudaNetwork(config, weights, CudaSettings.INSTANCE().getDefaultPrecision());
  }

  /**
   * Builds a CUDA/cuDNN-accelerated differentiable MindsEye {@link PipelineNetwork} at the specified GPU {@link Precision}.
   */
  @Nonnull
  public PipelineNetwork toCudaNetwork(@Nonnull final Precision precision) {
    return Gemma4Network.buildCudaNetwork(config, weights, precision);
  }

  /**
   * Builds a CUDA/cuDNN-accelerated differentiable MindsEye {@link PipelineNetwork} starting from continuous embeddings.
   */
  @Nonnull
  public PipelineNetwork toCudaEmbeddingNetwork() {
    return Gemma4Network.buildCudaFromEmbeddingsNetwork(config, weights, CudaSettings.INSTANCE().getDefaultPrecision());
  }

  /**
   * Builds a CUDA/cuDNN-accelerated differentiable MindsEye {@link PipelineNetwork} starting from continuous embeddings
   * at the specified GPU {@link Precision}.
   */
  @Nonnull
  public PipelineNetwork toCudaEmbeddingNetwork(@Nonnull final Precision precision) {
    return Gemma4Network.buildCudaFromEmbeddingsNetwork(config, weights, precision);
  }

  @Nonnull
  public State createNewState() {
    return createNewState(false);
  }

  @Nonnull
  public State createNewState(final boolean fp32KvCache) {
    State state = new State(config, fp32KvCache);
    if (tokenizer != null) {
      Integer bos = tokenizer.getSpecialTokens().get("<bos>");
      if (bos != null) {
        state.latestToken = bos;
      }
    }
    return state;
  }

  @Nonnull
  public static Gemma4Model load(@Nonnull final Path ggufPath) throws IOException {
    return load(ggufPath, -1);
  }

  @Nonnull
  public static Gemma4Model load(@Nonnull final Path ggufPath, final int contextLength) throws IOException {
    try (FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
      GGUF gguf = GGUF.loadModel(fileChannel);
      Map<String, GGUF.GGMLTensorEntry> entries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
      return fromGGUF(gguf, entries, contextLength);
    }
  }

  @Nonnull
  public static Gemma4Model fromGGUF(@Nonnull final GGUF gguf,
                                     @Nonnull final Map<String, GGUF.GGMLTensorEntry> entries,
                                     final int contextLength) {
    Gemma4Config config = Gemma4Config.fromGGUF(gguf, contextLength);
    GemmaTokenizer tokenizer = gguf.getMetadata().containsKey("tokenizer.ggml.tokens")
        ? GemmaTokenizer.fromGGUF(gguf.getMetadata())
        : null;
    Gemma4Weights weights = Gemma4Weights.fromGGUF(entries, config);
    return new Gemma4Model(config, tokenizer, weights);
  }

  /**
   * Mutable KV-cache and scratch-buffer state for token-by-token autoregressive inference.
   */
  @RefIgnore
  public static final class State {
    public final GGUFTensor.ArrayFloatTensor x;
    public final GGUFTensor.ArrayFloatTensor xb;
    public final GGUFTensor.ArrayFloatTensor xbK;
    public final GGUFTensor.ArrayFloatTensor xb2;
    public final GGUFTensor.ArrayFloatTensor hb;
    public final GGUFTensor.ArrayFloatTensor hb2;
    public final GGUFTensor.ArrayFloatTensor q;
    public final GGUFTensor.ArrayFloatTensor k;
    public final GGUFTensor.ArrayFloatTensor v;
    public final GGUFTensor.ArrayFloatTensor att;
    public final GGUFTensor.ArrayFloatTensor logits;
    public final GGUFTensor[] keyCache;
    public final GGUFTensor[] valueCache;
    @Nullable
    public final GGUFTensor.ArrayFloatTensor perLayerInputs;
    @Nullable
    public final GGUFTensor.ArrayFloatTensor plGate;
    @Nullable
    public final GGUFTensor.ArrayFloatTensor plProj;
    @Nullable
    public final GGUFTensor.ArrayFloatTensor routerLogits;
    @Nullable
    public final GGUFTensor.ArrayFloatTensor moeInput;
    @Nullable
    public final GGUFTensor.ArrayFloatTensor moeOutput;
    @Nullable
    public final GGUFTensor.ArrayFloatTensor expertGateUp;
    @Nullable
    public final GGUFTensor.ArrayFloatTensor expertDown;
    public int latestToken;

    public State(@Nonnull final Gemma4Config config, final boolean fp32KvCache) {
      int maxQueryDim = config.numberOfHeads * Math.max(config.headSizeFull, config.headSizeSWA);
      int maxKvDim = IntStream.range(0, config.numberOfLayers).map(config::kvDim).max().orElse(1);
      int maxHiddenDim = Math.max(1, config.maxHiddenDim());

      this.x = GGUFTensor.ArrayFloatTensor.allocate(config.embeddingLength);
      this.xb = GGUFTensor.ArrayFloatTensor.allocate(config.embeddingLength);
      this.xbK = GGUFTensor.ArrayFloatTensor.allocate(maxQueryDim);
      this.xb2 = GGUFTensor.ArrayFloatTensor.allocate(config.embeddingLength);
      this.hb = GGUFTensor.ArrayFloatTensor.allocate(maxHiddenDim);
      this.hb2 = GGUFTensor.ArrayFloatTensor.allocate(maxHiddenDim);
      this.q = GGUFTensor.ArrayFloatTensor.allocate(maxQueryDim);
      this.k = GGUFTensor.ArrayFloatTensor.allocate(maxKvDim);
      this.v = GGUFTensor.ArrayFloatTensor.allocate(maxKvDim);
      this.att = GGUFTensor.ArrayFloatTensor.allocate(config.numberOfHeads, config.contextLength);
      this.logits = GGUFTensor.ArrayFloatTensor.allocate(config.vocabularySize);

      int plDim = config.embeddingLengthPerLayer;
      this.perLayerInputs = plDim > 0 ? GGUFTensor.ArrayFloatTensor.allocate(plDim * config.numberOfLayers) : null;
      this.plGate = plDim > 0 ? GGUFTensor.ArrayFloatTensor.allocate(plDim) : null;
      this.plProj = plDim > 0 ? GGUFTensor.ArrayFloatTensor.allocate(config.embeddingLength) : null;

      if (config.isMoE()) {
        this.routerLogits = GGUFTensor.ArrayFloatTensor.allocate(config.expertCount);
        this.moeInput = GGUFTensor.ArrayFloatTensor.allocate(config.embeddingLength);
        this.moeOutput = GGUFTensor.ArrayFloatTensor.allocate(config.embeddingLength);
        this.expertGateUp = GGUFTensor.ArrayFloatTensor.allocate(2 * config.expertFeedForwardLength);
        this.expertDown = GGUFTensor.ArrayFloatTensor.allocate(config.embeddingLength);
      } else {
        this.routerLogits = null;
        this.moeInput = null;
        this.moeOutput = null;
        this.expertGateUp = null;
        this.expertDown = null;
      }

      this.keyCache = new GGUFTensor[config.nLayerKvFromStart];
      this.valueCache = new GGUFTensor[config.nLayerKvFromStart];
      for (int l = 0; l < config.nLayerKvFromStart; l++) {
        int kvDim = config.kvDim(l);
        int kvPositions = config.kvCachePositions(l);
        this.keyCache[l] = fp32KvCache
            ? GGUFTensor.ArrayFloatTensor.allocate(kvPositions, kvDim)
            : GGUFTensor.F16Tensor.allocate(kvPositions, kvDim);
        this.valueCache[l] = fp32KvCache
            ? GGUFTensor.ArrayFloatTensor.allocate(kvPositions, kvDim)
            : GGUFTensor.F16Tensor.allocate(kvPositions, kvDim);
      }
    }
  }

  static float gelu(final float x) {
    return (float) (0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x))));
  }

  static void rmsnorm(@Nonnull final GGUFTensor out, final int outOffset,
                      @Nonnull final GGUFTensor x, final int xOffset,
                      @Nonnull final float[] weight, final int size, final float rmsNormEps) {
    float ss = 0f;
    for (int i = 0; i < size; i++) {
      float xi = x.getFloat(xOffset + i);
      ss += xi * xi;
    }
    ss = (float) (1.0 / Math.sqrt(ss / size + rmsNormEps));
    for (int i = 0; i < size; i++) {
      out.setFloat(outOffset + i, weight[i] * ss * x.getFloat(xOffset + i));
    }
  }

  static void rmsnormNoWeight(@Nonnull final GGUFTensor out, final int outOffset,
                              @Nonnull final GGUFTensor x, final int xOffset,
                              final int size, final float rmsNormEps) {
    float ss = 0f;
    for (int i = 0; i < size; i++) {
      float xi = x.getFloat(xOffset + i);
      ss += xi * xi;
    }
    ss = (float) (1.0 / Math.sqrt(ss / size + rmsNormEps));
    for (int i = 0; i < size; i++) {
      out.setFloat(outOffset + i, ss * x.getFloat(xOffset + i));
    }
  }

  /**
   * Executes one KV-cached forward step for {@code token} at {@code position}, returning logits over the vocabulary.
   */
  @Nonnull
  public GGUFTensor forward(@Nonnull final State state, final int token, final int position) {
    int dim = config.embeddingLength;
    float sqrtDim = (float) Math.sqrt(dim);

    weights.tokenEmbeddingTable.copyTo((long) token * dim, state.x, 0, dim);
    state.x.mapInPlace(0, dim, v -> v * sqrtDim);

    int plDim = config.embeddingLengthPerLayer;
    int plTotal = plDim * config.numberOfLayers;
    if (plDim > 0 && weights.perLayerTokenEmbd != null && weights.perLayerModelProj != null
        && weights.perLayerProjNorm != null && state.perLayerInputs != null) {
      float sqrtPlDim = (float) Math.sqrt(plDim);
      float projScale = (float) (1.0 / Math.sqrt(dim));
      float inputScale = (float) (1.0 / Math.sqrt(2.0));

      weights.perLayerModelProj.matmul(state.x, state.perLayerInputs, plTotal, dim);
      state.perLayerInputs.mapInPlace(0, plTotal, v -> v * projScale);
      for (int l = 0; l < config.numberOfLayers; l++) {
        rmsnorm(state.perLayerInputs, l * plDim, state.perLayerInputs, l * plDim,
            weights.perLayerProjNorm, plDim, config.rmsNormEps);
      }

      long tokEmbOffset = (long) token * plTotal;
      for (int i = 0; i < plTotal; i++) {
        float tokEmb = weights.perLayerTokenEmbd.getFloat(tokEmbOffset + i) * sqrtPlDim;
        state.perLayerInputs.setFloat(i, state.perLayerInputs.getFloat(i) + tokEmb);
      }
      state.perLayerInputs.mapInPlace(0, plTotal, v -> v * inputScale);
    }

    for (int l = 0; l < config.numberOfLayers; l++) {
      boolean layerIsSWA = config.isSWA[l];
      int headSize = config.headSize(l);
      int halfHead = headSize / 2;
      int queryDim = config.queryDim(l);
      int kvDim = config.kvDim(l);
      int hiddenDim = config.feedForwardLength[l];

      rmsnorm(state.xb, 0, state.x, 0, weights.rmsAttWeight[l], dim, config.rmsNormEps);

      weights.wq[l].matmul(state.xb, state.q, queryDim, dim);
      for (int h = 0; h < config.numberOfHeads; h++) {
        rmsnorm(state.q, h * headSize, state.q, h * headSize, weights.attnQNorm[l], headSize, config.rmsNormEps);
      }

      float[] freqsReal = layerIsSWA ? weights.freqCisRealSwa : weights.freqCisRealFull;
      float[] freqsImag = layerIsSWA ? weights.freqCisImagSwa : weights.freqCisImagFull;
      for (int h = 0; h < config.numberOfHeads; ++h) {
        int poffset = h * headSize;
        for (int ic = 0; ic < halfHead; ic++) {
          float fcr = freqsReal[position * halfHead + ic];
          float fci = freqsImag[position * halfHead + ic];
          float v0 = state.q.getFloat(poffset + ic);
          float v1 = state.q.getFloat(poffset + ic + halfHead);
          state.q.setFloat(poffset + ic, v0 * fcr - v1 * fci);
          state.q.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr);
        }
      }

      int kvLayer = config.kvSourceLayer(l);
      int nKvHeads = config.numberOfKeyValueHeads(l);
      int kvMul = config.numberOfHeads / nKvHeads;
      if (config.hasKv(l)) {
        weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
        if (weights.wv[l] != null) {
          weights.wv[l].matmul(state.xb, state.v, kvDim, dim);
        } else {
          state.k.copyTo(0L, state.v, 0, kvDim);
        }

        for (int h = 0; h < nKvHeads; h++) {
          rmsnorm(state.k, h * headSize, state.k, h * headSize, weights.attnKNorm[l], headSize, config.rmsNormEps);
          rmsnormNoWeight(state.v, h * headSize, state.v, h * headSize, headSize, config.rmsNormEps);
        }

        for (int h = 0; h < nKvHeads; ++h) {
          int poffset = h * headSize;
          for (int ic = 0; ic < halfHead; ic++) {
            float fcr = freqsReal[position * halfHead + ic];
            float fci = freqsImag[position * halfHead + ic];
            float v0 = state.k.getFloat(poffset + ic);
            float v1 = state.k.getFloat(poffset + ic + halfHead);
            state.k.setFloat(poffset + ic, v0 * fcr - v1 * fci);
            state.k.setFloat(poffset + ic + halfHead, v0 * fci + v1 * fcr);
          }
        }

        int kvPos = config.kvCacheIndex(l, position);
        state.k.copyTo(0L, state.keyCache[kvLayer], kvPos * kvDim, kvDim);
        state.v.copyTo(0L, state.valueCache[kvLayer], kvPos * kvDim, kvDim);
      }

      final int attStart = layerIsSWA ? Math.max(0, position - config.slidingWindow + 1) : 0;
      final int finalKvLayer = kvLayer;
      final int finalKvDim = kvDim;
      final int finalLayer = l;

      IntStream.range(0, config.numberOfHeads).parallel().forEach(h -> {
        int qOffset = h * headSize;
        int attOffset = h * config.contextLength;
        int kvHeadOffset = (h / kvMul) * headSize;
        for (int t = attStart; t <= position; t++) {
          int keyCacheOffset = config.kvCacheIndex(finalLayer, t) * finalKvDim + kvHeadOffset;
          float score = state.q.dot(qOffset, state.keyCache[finalKvLayer], keyCacheOffset, headSize);
          state.att.setFloat(attOffset + t, score);
        }
        state.att.softmaxInPlace(attOffset + attStart, position - attStart + 1);
        int xbOffset = h * headSize;
        state.xbK.fillInPlace(xbOffset, headSize, 0f);
        for (int t = attStart; t <= position; t++) {
          int vOffset = config.kvCacheIndex(finalLayer, t) * finalKvDim + kvHeadOffset;
          float a = state.att.getFloat(attOffset + t);
          state.xbK.saxpyInPlace(xbOffset, state.valueCache[finalKvLayer], vOffset, headSize, a);
        }
      });

      weights.wo[l].matmul(state.xbK, state.xb2, dim, queryDim);
      rmsnorm(state.xb2, 0, state.xb2, 0, weights.postAttentionNorm[l], dim, config.rmsNormEps);
      state.x.addInPlace(0, state.xb2, 0, dim);

      boolean isMoELayer = config.isMoE() && weights.ffnGateInp != null && weights.ffnGateInp[l] != null;
      if (isMoELayer) {
        rmsnorm(state.xb, 0, state.x, 0, weights.rmsFfnWeight[l], dim, config.rmsNormEps);
        weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim);
        weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim);
        state.hb.mapInPlace(0, hiddenDim, Gemma4Model::gelu);
        state.hb.multiplyInPlace(0, state.hb2, 0, hiddenDim);
        weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim);
        rmsnorm(state.xb, 0, state.xb, 0, weights.ffnPostNorm1[l], dim, config.rmsNormEps);

        rmsnorm(state.moeInput, 0, state.x, 0, weights.preFfwNorm2[l], dim, config.rmsNormEps);
        float ss = state.x.reduce(0, dim, 0f, (acc, xi) -> acc + xi * xi);
        float rmsScale = (float) (1.0 / Math.sqrt(ss / dim + config.rmsNormEps)) / (float) Math.sqrt(dim);
        for (int i = 0; i < dim; i++) {
          state.xb2.setFloat(i, state.x.getFloat(i) * rmsScale * weights.ffnGateInpScale[l][i]);
        }
        weights.ffnGateInp[l].matmul(state.xb2, state.routerLogits, config.expertCount, dim);
        state.routerLogits.softmaxInPlace(0, config.expertCount);

        int nExperts = config.expertCount;
        int topK = config.expertUsedCount;
        int[] topExperts = new int[topK];
        float[] topProbs = new float[topK];
        for (int ki = 0; ki < topK; ki++) {
          int bestIdx = 0;
          float bestVal = Float.NEGATIVE_INFINITY;
          for (int ei = 0; ei < nExperts; ei++) {
            float val = state.routerLogits.getFloat(ei);
            if (val > bestVal) {
              bestVal = val;
              bestIdx = ei;
            }
          }
          topExperts[ki] = bestIdx;
          topProbs[ki] = bestVal;
          state.routerLogits.setFloat(bestIdx, Float.NEGATIVE_INFINITY);
        }

        int expertFF = config.expertFeedForwardLength;
        int gateUpDim = 2 * expertFF;
        state.moeOutput.fillInPlace(0, dim, 0f);
        for (int ki = 0; ki < topK; ki++) {
          int expertIdx = topExperts[ki];
          float prob = topProbs[ki];
          float downScale = weights.ffnDownExpsScale[l][expertIdx];
          int gateUpOffset = expertIdx * gateUpDim * dim;
          weights.ffnGateUpExps[l].matmul(state.moeInput, state.expertGateUp, gateUpDim, dim, gateUpOffset);
          state.expertGateUp.mapInPlace(0, expertFF, Gemma4Model::gelu);
          for (int i = 0; i < expertFF; i++) {
            state.expertGateUp.setFloat(i, state.expertGateUp.getFloat(i) * state.expertGateUp.getFloat(expertFF + i));
          }
          int downOffset = expertIdx * dim * expertFF;
          weights.ffnDownExps[l].matmul(state.expertGateUp, state.expertDown, dim, expertFF, downOffset);
          state.moeOutput.saxpyInPlace(0, state.expertDown, 0, dim, prob * downScale);
        }

        rmsnorm(state.moeOutput, 0, state.moeOutput, 0, weights.ffnPostNorm2[l], dim, config.rmsNormEps);
        state.xb.addInPlace(0, state.moeOutput, 0, dim);
        rmsnorm(state.xb, 0, state.xb, 0, weights.postFfwNorm[l], dim, config.rmsNormEps);
        state.x.addInPlace(0, state.xb, 0, dim);
      } else {
        rmsnorm(state.xb, 0, state.x, 0, weights.rmsFfnWeight[l], dim, config.rmsNormEps);
        weights.w1[l].matmul(state.xb, state.hb, hiddenDim, dim);
        weights.w3[l].matmul(state.xb, state.hb2, hiddenDim, dim);
        state.hb.mapInPlace(0, hiddenDim, Gemma4Model::gelu);
        state.hb.multiplyInPlace(0, state.hb2, 0, hiddenDim);
        weights.w2[l].matmul(state.hb, state.xb, dim, hiddenDim);
        rmsnorm(state.xb, 0, state.xb, 0, weights.postFfwNorm[l], dim, config.rmsNormEps);
        state.x.addInPlace(0, state.xb, 0, dim);
      }

      if (plDim > 0 && weights.perLayerInpGate != null && weights.perLayerInpGate[l] != null
          && state.plGate != null && state.plProj != null && state.perLayerInputs != null) {
        weights.perLayerInpGate[l].matmul(state.x, state.plGate, plDim, dim);
        state.plGate.mapInPlace(0, plDim, Gemma4Model::gelu);
        int plOffset = l * plDim;
        for (int i = 0; i < plDim; i++) {
          state.plGate.setFloat(i, state.plGate.getFloat(i) * state.perLayerInputs.getFloat(plOffset + i));
        }
        weights.perLayerProj[l].matmul(state.plGate, state.plProj, dim, plDim);
        rmsnorm(state.plProj, 0, state.plProj, 0, weights.perLayerPostNorm[l], dim, config.rmsNormEps);
        state.x.addInPlace(0, state.plProj, 0, dim);
      }

      float scale = weights.layerOutputScale[l];
      if (scale != 1.0f) {
        state.x.mapInPlace(0, dim, v -> v * scale);
      }
    }

    rmsnorm(state.x, 0, state.x, 0, weights.rmsFinalWeight, dim, config.rmsNormEps);
    weights.wcls.matmul(state.x, state.logits, config.vocabularySize, dim);
    if (config.logitSoftcapping > 0f) {
      float cap = config.logitSoftcapping;
      state.logits.mapInPlace(0, config.vocabularySize, v -> cap * (float) Math.tanh(v / cap));
    }
    return state.logits;
  }

  /**
   * Generates tokens autoregressively using the KV cache.
   */
  @Nonnull
  public List<Integer> generateTokens(@Nonnull final State state,
                                      final int startPosition,
                                      @Nonnull final List<Integer> promptTokens,
                                      @Nonnull final Set<Integer> stopTokens,
                                      final int requestedMaxTokens,
                                      @Nonnull final GemmaSampler sampler,
                                      final boolean echo,
                                      @Nullable final IntConsumer onTokenGenerated) {
    int maxTokens = (requestedMaxTokens < 0 || config.contextLength < requestedMaxTokens)
        ? config.contextLength
        : requestedMaxTokens;
    List<Integer> generatedTokens = new ArrayList<>(Math.max(16, maxTokens));
    int token = state.latestToken;
    int promptIndex = 0;
    for (int position = startPosition; position < maxTokens; ++position) {
      forward(state, token, position);
      int nextToken;
      if (promptIndex < promptTokens.size()) {
        nextToken = promptTokens.get(promptIndex++);
        if (echo && tokenizer != null) {
          System.err.print(GemmaTokenizer.replaceControlCharacters(tokenizer.decode(List.of(nextToken))));
        }
      } else {
        nextToken = sampler.sampleToken(state.logits);
        if (echo && tokenizer != null) {
          System.err.print(GemmaTokenizer.replaceControlCharacters(tokenizer.decode(List.of(nextToken))));
        }
        generatedTokens.add(nextToken);
        if (onTokenGenerated != null) {
          onTokenGenerated.accept(nextToken);
        }
        if (stopTokens.contains(nextToken)) {
          state.latestToken = nextToken;
          break;
        }
      }
      state.latestToken = token = nextToken;
    }
    return generatedTokens;
  }

  /**
   * High-level single-turn chat/instruct text generation helper.
   */
  @Nonnull
  public String generate(@Nonnull final String prompt,
                         @Nullable final String systemPrompt,
                         final int maxNewTokens,
                         final float temperature,
                         final float topP,
                         final long seed,
                         final boolean think) {
    if (tokenizer == null) {
      throw new IllegalStateException("Model was initialized without a GemmaTokenizer");
    }
    State state = createNewState();
    GemmaChatFormat chatFormat = new GemmaChatFormat(tokenizer);
    List<Integer> promptTokens = new ArrayList<>();
    if (think) {
      promptTokens.addAll(chatFormat.encodeSystemThinkingTurn(systemPrompt));
    } else if (systemPrompt != null) {
      promptTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.SYSTEM, systemPrompt)));
    }
    promptTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.USER, prompt)));
    promptTokens.addAll(chatFormat.encodeHeader(new GemmaChatFormat.Message(GemmaChatFormat.Role.MODEL, "")));

    Set<Integer> stopTokens = chatFormat.getStopTokens();
    GemmaSampler sampler = GemmaSampler.selectSampler(config.vocabularySize, temperature, topP, seed);
    int limit = Math.min(config.contextLength, promptTokens.size() + maxNewTokens);
    List<Integer> responseTokens = generateTokens(state, 0, promptTokens, stopTokens, limit, sampler, false, null);
    if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.get(responseTokens.size() - 1))) {
      responseTokens.remove(responseTokens.size() - 1);
    }
    List<Integer> visible = GemmaChatFormat.visibleTokens(tokenizer, responseTokens, think);
    return tokenizer.decode(visible);
  }

  private static IntConsumer streamingPrinter(final GemmaTokenizer tokenizer, final Options options) {
    if (!options.stream || tokenizer == null) {
      return token -> {
      };
    }
    Integer channelOpen = tokenizer.getSpecialTokens().get("<|channel>");
    Integer channelClose = tokenizer.getSpecialTokens().get("<channel|>");
    if (channelOpen == null || channelClose == null) {
      return token -> {
        if (!tokenizer.isSpecialToken(token)) {
          System.out.print(tokenizer.decode(List.of(token)));
        }
      };
    }
    final boolean thinkEnabled = options.think;
    final PrintStream thoughtOut = options.thinkInline ? System.out : System.err;
    final boolean ansi = options.colors;
    final boolean[] inChannel = {false};
    final boolean[] emitted = {false};
    return token -> {
      if (token == channelOpen) {
        if (thinkEnabled) {
          if (ansi) {
            thoughtOut.print(ANSI_GREY);
          }
          thoughtOut.println("[Start thinking]");
        }
        inChannel[0] = true;
        emitted[0] = false;
        return;
      }
      if (token == channelClose) {
        if (thinkEnabled) {
          if (emitted[0]) {
            thoughtOut.println();
          }
          thoughtOut.println("[End thinking]");
          if (ansi) {
            thoughtOut.print(ANSI_RESET);
          }
          thoughtOut.println();
        }
        inChannel[0] = false;
        emitted[0] = false;
        return;
      }
      if (!tokenizer.isSpecialToken(token)) {
        String text = tokenizer.decode(List.of(token));
        if (inChannel[0]) {
          if (thinkEnabled) {
            thoughtOut.print(text);
            emitted[0] = true;
          }
        } else {
          System.out.print(text);
        }
      }
    };
  }

  static void runInteractive(final Gemma4Model model, final GemmaSampler sampler, final Options options) {
    GemmaTokenizer tok = model.tokenizer();
    if (tok == null) {
      throw new IllegalStateException("Tokenizer required for interactive chat");
    }
    State state = null;
    GemmaChatFormat chatFormat = new GemmaChatFormat(tok);
    List<Integer> conversationTokens = new ArrayList<>();
    if (options.think) {
      conversationTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt));
    } else if (options.systemPrompt != null) {
      conversationTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.SYSTEM, options.systemPrompt)));
    }
    int startPosition = 0;
    Scanner in = new Scanner(System.in);
    while (true) {
      System.out.print("> ");
      System.out.flush();
      if (!in.hasNextLine()) {
        break;
      }
      String userText = in.nextLine();
      if ("/quit".equals(userText) || "/exit".equals(userText)) {
        break;
      }
      if ("/context".equals(userText)) {
        System.out.printf("%d out of %d context tokens used (%d remaining)%n",
            conversationTokens.size(), options.maxTokens, options.maxTokens - conversationTokens.size());
        continue;
      }
      if (state == null) {
        state = model.createNewState();
      }
      conversationTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.USER, userText)));
      conversationTokens.addAll(chatFormat.encodeHeader(new GemmaChatFormat.Message(GemmaChatFormat.Role.MODEL, "")));

      Set<Integer> stopTokens = chatFormat.getStopTokens();
      IntConsumer printer = streamingPrinter(tok, options);
      List<Integer> responseTokens = model.generateTokens(
          state, startPosition,
          conversationTokens.subList(startPosition, conversationTokens.size()),
          stopTokens, options.maxTokens, sampler, options.echo, printer
      );
      Integer stopToken = null;
      if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.get(responseTokens.size() - 1))) {
        stopToken = responseTokens.remove(responseTokens.size() - 1);
      }
      List<Integer> visible = GemmaChatFormat.visibleTokens(tok, responseTokens, options.think);
      conversationTokens.addAll(responseTokens);
      if (stopToken != null) {
        conversationTokens.add(stopToken);
        if (stopToken == chatFormat.getEndOfTurn()) {
          conversationTokens.addAll(tok.encode("\n"));
        }
      }
      startPosition = conversationTokens.size();
      if (!options.stream) {
        System.out.println(tok.decode(visible));
      } else {
        System.out.println();
      }
      if (stopToken == null) {
        System.err.println("Ran out of context length...");
        break;
      }
    }
  }

  static void runInstructOnce(final Gemma4Model model, final GemmaSampler sampler, final Options options) {
    GemmaTokenizer tok = model.tokenizer();
    if (tok == null) {
      throw new IllegalStateException("Tokenizer required for instruct mode");
    }
    State state = model.createNewState();
    GemmaChatFormat chatFormat = new GemmaChatFormat(tok);
    List<Integer> promptTokens = new ArrayList<>();

    if (options.suffix != null) {
      promptTokens.addAll(chatFormat.encodeFillInTheMiddle(options.prompt, options.suffix));
    } else {
      if (options.think) {
        promptTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt));
      } else if (options.systemPrompt != null) {
        promptTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.SYSTEM, options.systemPrompt)));
      }
      promptTokens.addAll(chatFormat.encodeMessage(new GemmaChatFormat.Message(GemmaChatFormat.Role.USER, options.prompt)));
      promptTokens.addAll(chatFormat.encodeHeader(new GemmaChatFormat.Message(GemmaChatFormat.Role.MODEL, "")));
    }

    Set<Integer> stopTokens = chatFormat.getStopTokens();
    IntConsumer printer = streamingPrinter(tok, options);
    long t0 = System.nanoTime();
    List<Integer> responseTokens = model.generateTokens(state, 0, promptTokens, stopTokens, options.maxTokens, sampler, options.echo, printer);
    long elapsedNanos = System.nanoTime() - t0;
    if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.get(responseTokens.size() - 1))) {
      responseTokens.remove(responseTokens.size() - 1);
    }
    List<Integer> visible = GemmaChatFormat.visibleTokens(tok, responseTokens, options.think);
    if (!options.stream) {
      System.out.println(tok.decode(visible));
    } else {
      System.out.println();
    }
    String prefix = options.colors ? ANSI_CYAN : "";
    String suffix = options.colors ? ANSI_RESET : "";
    System.err.printf("%scontext: %d/%d total: %.2f s (%d prompt + %d generated)%s%n",
        prefix, promptTokens.size() + responseTokens.size(), model.configuration().contextLength,
        elapsedNanos / 1_000_000_000.0, promptTokens.size(), responseTokens.size(), suffix);
  }

  @RefIgnore
  public static final class Options {
    public final Path modelPath;
    public final String prompt;
    public final String suffix;
    public final String systemPrompt;
    public final boolean interactive;
    public final float temperature;
    public final float topp;
    public final long seed;
    public final int maxTokens;
    public final boolean stream;
    public final boolean echo;
    public final boolean think;
    public final boolean thinkInline;
    public final boolean colors;

    public Options(final Path modelPath, final String prompt, final String suffix, final String systemPrompt,
                   final boolean interactive, final float temperature, final float topp, final long seed,
                   final int maxTokens, final boolean stream, final boolean echo,
                   final boolean think, final boolean thinkInline, final boolean colors) {
      if (modelPath == null) {
        throw new IllegalArgumentException("Missing argument: --model <path> is required");
      }
      if (!interactive && prompt == null) {
        throw new IllegalArgumentException("Missing argument: --prompt is required in --instruct mode");
      }
      if (temperature < 0f) {
        throw new IllegalArgumentException("--temperature must be non-negative");
      }
      if (topp < 0f || topp > 1f) {
        throw new IllegalArgumentException("--top-p must be within [0, 1]");
      }
      this.modelPath = modelPath;
      this.prompt = prompt;
      this.suffix = suffix;
      this.systemPrompt = systemPrompt;
      this.interactive = interactive;
      this.temperature = temperature;
      this.topp = topp;
      this.seed = seed;
      this.maxTokens = maxTokens;
      this.stream = stream;
      this.echo = echo;
      this.think = think;
      this.thinkInline = thinkInline;
      this.colors = colors;
    }

    public static Options parseOptions(@Nonnull final String[] args) {
      String prompt = null;
      String suffix = null;
      String systemPrompt = null;
      float temperature = 1f;
      float topp = 0.95f;
      Path modelPath = null;
      long seed = System.nanoTime();
      int maxTokens = DEFAULT_MAX_TOKENS;
      boolean interactive = false;
      boolean stream = true;
      boolean echo = false;
      boolean think = false;
      boolean thinkInline = false;
      String colorMode = "auto";

      for (int i = 0; i < args.length; i++) {
        String opt = args[i];
        if ("--interactive".equals(opt) || "--chat".equals(opt) || "-i".equals(opt)) {
          interactive = true;
        } else if ("--instruct".equals(opt)) {
          interactive = false;
        } else {
          String val;
          if (opt.contains("=")) {
            String[] parts = opt.split("=", 2);
            opt = parts[0];
            val = parts[1];
          } else {
            if (i + 1 >= args.length) {
              throw new IllegalArgumentException("Missing argument for option " + opt);
            }
            val = args[++i];
          }
          switch (opt) {
            case "--prompt":
            case "-p":
              prompt = val;
              break;
            case "--suffix":
              suffix = val;
              break;
            case "--system-prompt":
            case "-sp":
              systemPrompt = val;
              break;
            case "--temperature":
            case "--temp":
              temperature = Float.parseFloat(val);
              break;
            case "--top-p":
              topp = Float.parseFloat(val);
              break;
            case "--model":
            case "-m":
              modelPath = Path.of(val);
              break;
            case "--seed":
            case "-s":
              seed = Long.parseLong(val);
              break;
            case "--max-tokens":
            case "-n":
              maxTokens = Integer.parseInt(val);
              break;
            case "--stream":
              stream = "true".equalsIgnoreCase(val) || "on".equalsIgnoreCase(val);
              break;
            case "--echo":
              echo = "true".equalsIgnoreCase(val) || "on".equalsIgnoreCase(val);
              break;
            case "--color":
              colorMode = val.toLowerCase(Locale.ROOT);
              break;
            case "--think": {
              String tm = val.toLowerCase(Locale.ROOT);
              thinkInline = "inline".equals(tm) || "stdout".equals(tm);
              think = "on".equals(tm) || "true".equals(tm) || thinkInline;
              break;
            }
            default:
              throw new IllegalArgumentException("Unknown option: " + opt);
          }
        }
      }
      boolean colors = "on".equals(colorMode)
          || ("auto".equals(colorMode) && System.console() != null && System.getenv("NO_COLOR") == null);
      return new Options(modelPath, prompt, suffix, systemPrompt, interactive, temperature, topp, seed,
          maxTokens, stream, echo, think, thinkInline, colors);
    }
  }

  public static void main(@Nonnull final String[] args) throws IOException {
    Options options = Options.parseOptions(args);
    Gemma4Model model = load(options.modelPath, options.maxTokens);
    GemmaSampler sampler = GemmaSampler.selectSampler(
        model.configuration().vocabularySize, options.temperature, options.topp, options.seed);
    if (options.interactive) {
      runInteractive(model, sampler, options);
    } else {
      runInstructOnce(model, sampler, options);
    }
  }
}
