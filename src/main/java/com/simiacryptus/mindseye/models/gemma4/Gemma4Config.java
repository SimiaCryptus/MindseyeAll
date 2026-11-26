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
import java.util.Arrays;
import java.util.Map;

/**
 * Architectural configuration for Gemma 4 model families (E2B, E4B, 31B dense, and 26B-A4B MoE).
 */
@RefIgnore
public final class Gemma4Config {

  public final int embeddingLength;
  public final int[] feedForwardLength;
  public final int numberOfLayers;
  public final int numberOfHeads;
  public final int[] numberOfKeyValueHeadsPerLayer;
  public final int vocabularySize;
  public final int contextLength;
  public final float rmsNormEps;
  public final float ropeTheta;
  public final float ropeThetaSWA;
  public final int headSizeFull;
  public final int headSizeSWA;
  public final int slidingWindow;
  public final float logitSoftcapping;
  public final boolean[] isSWA;
  public final int nLayerKvFromStart;
  public final int embeddingLengthPerLayer;
  public final int expertCount;
  public final int expertUsedCount;
  public final int expertFeedForwardLength;

  public Gemma4Config(final int embeddingLength,
                      @Nonnull final int[] feedForwardLength,
                      final int numberOfLayers,
                      final int numberOfHeads,
                      @Nonnull final int[] numberOfKeyValueHeadsPerLayer,
                      final int vocabularySize,
                      final int contextLength,
                      final float rmsNormEps,
                      final float ropeTheta,
                      final float ropeThetaSWA,
                      final int headSizeFull,
                      final int headSizeSWA,
                      final int slidingWindow,
                      final float logitSoftcapping,
                      @Nonnull final boolean[] isSWA,
                      final int nLayerKvFromStart,
                      final int embeddingLengthPerLayer,
                      final int expertCount,
                      final int expertUsedCount,
                      final int expertFeedForwardLength) {
    if (slidingWindow <= 0 || Integer.bitCount(slidingWindow) != 1) {
      throw new IllegalArgumentException("slidingWindow must be a power of 2, got " + slidingWindow);
    }
    if (feedForwardLength.length != numberOfLayers) {
      throw new IllegalArgumentException("feedForwardLength length != numberOfLayers");
    }
    if (numberOfKeyValueHeadsPerLayer.length != numberOfLayers) {
      throw new IllegalArgumentException("numberOfKeyValueHeadsPerLayer length != numberOfLayers");
    }
    if (isSWA.length != numberOfLayers) {
      throw new IllegalArgumentException("isSWA length != numberOfLayers");
    }
    if (nLayerKvFromStart <= 0 || nLayerKvFromStart > numberOfLayers) {
      throw new IllegalArgumentException("Invalid nLayerKvFromStart: " + nLayerKvFromStart);
    }
    this.embeddingLength = embeddingLength;
    this.feedForwardLength = feedForwardLength.clone();
    this.numberOfLayers = numberOfLayers;
    this.numberOfHeads = numberOfHeads;
    this.numberOfKeyValueHeadsPerLayer = numberOfKeyValueHeadsPerLayer.clone();
    this.vocabularySize = vocabularySize;
    this.contextLength = contextLength;
    this.rmsNormEps = rmsNormEps;
    this.ropeTheta = ropeTheta;
    this.ropeThetaSWA = ropeThetaSWA;
    this.headSizeFull = headSizeFull;
    this.headSizeSWA = headSizeSWA;
    this.slidingWindow = slidingWindow;
    this.logitSoftcapping = logitSoftcapping;
    this.isSWA = isSWA.clone();
    this.nLayerKvFromStart = nLayerKvFromStart;
    this.embeddingLengthPerLayer = embeddingLengthPerLayer;
    this.expertCount = expertCount;
    this.expertUsedCount = expertUsedCount;
    this.expertFeedForwardLength = expertFeedForwardLength;
  }

  public boolean isMoE() {
    return expertCount > 0;
  }

  public boolean hasPerLayerEmbeddings() {
    return embeddingLengthPerLayer > 0;
  }

  public boolean hasKv(final int layer) {
    return layer < nLayerKvFromStart;
  }

  public int kvSourceLayer(final int layer) {
    if (layer < nLayerKvFromStart) {
      return layer;
    }
    boolean targetSwa = isSWA[layer];
    for (int k = nLayerKvFromStart - 1; k >= 0; k--) {
      if (isSWA[k] == targetSwa) {
        return k;
      }
    }
    return nLayerKvFromStart - (targetSwa ? 2 : 1);
  }

  public int headSize(final int layer) {
    return isSWA[layer] ? headSizeSWA : headSizeFull;
  }

  public int numberOfKeyValueHeads(final int layer) {
    return numberOfKeyValueHeadsPerLayer[layer];
  }

  public int kvCachePositions(final int layer) {
    return isSWA[layer] ? Math.min(contextLength, slidingWindow) : contextLength;
  }

  public int kvCacheIndex(final int layer, final int position) {
    return isSWA[layer] ? (position & (slidingWindow - 1)) : position;
  }

  public int kvDim(final int layer) {
    return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer);
  }

  public int queryDim(final int layer) {
    return numberOfHeads * headSize(layer);
  }

  public int maxHiddenDim() {
    return Arrays.stream(feedForwardLength).max().orElse(0);
  }

  @Nonnull
  public Gemma4Config withContextLength(final int newContextLength) {
    return new Gemma4Config(
        embeddingLength, feedForwardLength, numberOfLayers,
        numberOfHeads, numberOfKeyValueHeadsPerLayer, vocabularySize,
        newContextLength, rmsNormEps, ropeTheta, ropeThetaSWA,
        headSizeFull, headSizeSWA, slidingWindow,
        logitSoftcapping, isSWA, nLayerKvFromStart,
        embeddingLengthPerLayer,
        expertCount, expertUsedCount, expertFeedForwardLength
    );
  }

  @Nonnull
  public static Gemma4Config fromGGUF(@Nonnull final GGUF gguf, final int requestedContextLength) {
    Map<String, Object> metadata = gguf.getMetadata();
    Map<String, GGUF.GGUFTensorInfo> tensorInfos = gguf.getTensorInfos();

    int modelContextLength = getInt(metadata, "gemma4.context_length");
    int contextLength = (requestedContextLength < 0 || modelContextLength < requestedContextLength)
        ? modelContextLength
        : requestedContextLength;

    int embeddingLength = getInt(metadata, "gemma4.embedding_length");
    int numberOfHeads = getInt(metadata, "gemma4.attention.head_count");
    int numberOfLayers = getInt(metadata, "gemma4.block_count");
    int headSizeFull = getInt(metadata, "gemma4.attention.key_length");
    int headSizeSWA = getInt(metadata, "gemma4.attention.key_length_swa");
    int slidingWindow = getInt(metadata, "gemma4.attention.sliding_window");
    float logitSoftcapping = getFloatOrDefault(metadata, "gemma4.final_logit_softcapping", 0f);
    float rmsNormEps = getFloatOrDefault(metadata, "gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
    float ropeTheta = getFloatOrDefault(metadata, "gemma4.rope.freq_base", 1000000f);
    float ropeThetaSWA = getFloatOrDefault(metadata, "gemma4.rope.freq_base_swa", 10000f);

    int expertCount = getIntOrDefault(metadata, "gemma4.expert_count", 0);
    int expertUsedCount = getIntOrDefault(metadata, "gemma4.expert_used_count", 0);
    int expertFeedForwardLength = getIntOrDefault(metadata, "gemma4.expert_feed_forward_length", 0);

    int[] feedForwardLength;
    Object ffnRaw = metadata.get("gemma4.feed_forward_length");
    if (ffnRaw instanceof int[]) {
      feedForwardLength = ((int[]) ffnRaw).clone();
    } else if (ffnRaw instanceof Number) {
      feedForwardLength = new int[numberOfLayers];
      Arrays.fill(feedForwardLength, ((Number) ffnRaw).intValue());
    } else {
      throw new IllegalArgumentException("Missing gemma4.feed_forward_length metadata");
    }

    boolean[] isSWA;
    Object swaRaw = metadata.get("gemma4.attention.sliding_window_pattern");
    if (swaRaw instanceof boolean[]) {
      isSWA = ((boolean[]) swaRaw).clone();
    } else {
      isSWA = new boolean[numberOfLayers];
      for (int i = 0; i < numberOfLayers; i++) {
        GGUF.GGUFTensorInfo qNorm = tensorInfos.get("blk." + i + ".attn_q_norm.weight");
        if (qNorm != null) {
          long qNormSize = GGUFTensor.numberOfElementsLong(qNorm.dimensions());
          isSWA[i] = (qNormSize == headSizeSWA);
        } else {
          isSWA[i] = (i % 5 != 4);
        }
      }
    }

    int[] kvHeadsPerLayer = new int[numberOfLayers];
    for (int i = 0; i < numberOfLayers; i++) {
      GGUF.GGUFTensorInfo kWeight = tensorInfos.get("blk." + i + ".attn_k.weight");
      int headSize = isSWA[i] ? headSizeSWA : headSizeFull;
      if (kWeight != null) {
        kvHeadsPerLayer[i] = kWeight.dimensions()[1] / headSize;
      } else {
        int src = i - (isSWA[i] ? 2 : 1);
        kvHeadsPerLayer[i] = (src >= 0 && kvHeadsPerLayer[src] > 0) ? kvHeadsPerLayer[src] : numberOfHeads;
      }
    }
    int sharedKvLayers = getIntOrDefault(metadata, "gemma4.attention.shared_kv_layers", 0);
    int nLayerKvFromStart = numberOfLayers - sharedKvLayers;
    for (int i = nLayerKvFromStart; i < numberOfLayers; i++) {
      boolean targetSwa = isSWA[i];
      for (int k = nLayerKvFromStart - 1; k >= 0; k--) {
        if (isSWA[k] == targetSwa) {
          kvHeadsPerLayer[i] = kvHeadsPerLayer[k];
          break;
        }
      }
    }

    int embeddingLengthPerLayer = getIntOrDefault(metadata, "gemma4.embedding_length_per_layer_input", 0);
    String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
    int vocabSize;
    if (tokens != null) {
      vocabSize = tokens.length;
    } else {
      GGUF.GGUFTensorInfo embInfo = tensorInfos.get("token_embd.weight");
      if (embInfo == null) {
        throw new IllegalArgumentException("Cannot determine vocabulary size from GGUF");
      }
      vocabSize = embInfo.dimensions()[1];
    }

    return new Gemma4Config(
        embeddingLength, feedForwardLength, numberOfLayers, numberOfHeads, kvHeadsPerLayer,
        vocabSize, contextLength, rmsNormEps, ropeTheta, ropeThetaSWA,
        headSizeFull, headSizeSWA, slidingWindow, logitSoftcapping,
        isSWA, nLayerKvFromStart, embeddingLengthPerLayer,
        expertCount, expertUsedCount, expertFeedForwardLength
    );
  }

  private static int getInt(final Map<String, Object> metadata, final String key) {
    Object val = metadata.get(key);
    if (!(val instanceof Number)) {
      throw new IllegalArgumentException("Missing or non-numeric GGUF metadata key: " + key);
    }
    return ((Number) val).intValue();
  }

  private static int getIntOrDefault(final Map<String, Object> metadata, final String key, final int def) {
    Object val = metadata.get(key);
    return val instanceof Number ? ((Number) val).intValue() : def;
  }

  private static float getFloatOrDefault(final Map<String, Object> metadata, final String key, final float def) {
    Object val = metadata.get(key);
    return val instanceof Number ? ((Number) val).floatValue() : def;
  }

  /**
   * Fluent builder for creating custom or test {@link Gemma4Config} instances.
   */
  @RefIgnore
  public static final class Builder {
    private int embeddingLength = 64;
    private int[] feedForwardLength = null;
    private int defaultFeedForwardLength = 128;
    private int numberOfLayers = 2;
    private int numberOfHeads = 4;
    private int[] kvHeadsPerLayer = null;
    private int defaultKvHeads = 2;
    private int vocabularySize = 128;
    private int contextLength = 64;
    private float rmsNormEps = 1e-6f;
    private float ropeTheta = 1000000f;
    private float ropeThetaSWA = 10000f;
    private int headSizeFull = 16;
    private int headSizeSWA = 8;
    private int slidingWindow = 8;
    private float logitSoftcapping = 30.0f;
    private boolean[] isSWA = null;
    private int sharedKvLayers = 0;
    private int embeddingLengthPerLayer = 0;
    private int expertCount = 0;
    private int expertUsedCount = 0;
    private int expertFeedForwardLength = 0;

    @Nonnull
    public Builder setEmbeddingLength(final int embeddingLength) {
      this.embeddingLength = embeddingLength;
      return this;
    }

    @Nonnull
    public Builder setDefaultFeedForwardLength(final int ffnLen) {
      this.defaultFeedForwardLength = ffnLen;
      return this;
    }

    @Nonnull
    public Builder setFeedForwardLength(@Nonnull final int[] feedForwardLength) {
      this.feedForwardLength = feedForwardLength.clone();
      return this;
    }

    @Nonnull
    public Builder setNumberOfLayers(final int numberOfLayers) {
      this.numberOfLayers = numberOfLayers;
      return this;
    }

    @Nonnull
    public Builder setNumberOfHeads(final int numberOfHeads) {
      this.numberOfHeads = numberOfHeads;
      return this;
    }

    @Nonnull
    public Builder setDefaultKvHeads(final int defaultKvHeads) {
      this.defaultKvHeads = defaultKvHeads;
      return this;
    }

    @Nonnull
    public Builder setKvHeadsPerLayer(@Nonnull final int[] kvHeadsPerLayer) {
      this.kvHeadsPerLayer = kvHeadsPerLayer.clone();
      return this;
    }

    @Nonnull
    public Builder setVocabularySize(final int vocabularySize) {
      this.vocabularySize = vocabularySize;
      return this;
    }

    @Nonnull
    public Builder setContextLength(final int contextLength) {
      this.contextLength = contextLength;
      return this;
    }

    @Nonnull
    public Builder setRmsNormEps(final float rmsNormEps) {
      this.rmsNormEps = rmsNormEps;
      return this;
    }

    @Nonnull
    public Builder setRopeTheta(final float ropeTheta, final float ropeThetaSWA) {
      this.ropeTheta = ropeTheta;
      this.ropeThetaSWA = ropeThetaSWA;
      return this;
    }

    @Nonnull
    public Builder setHeadSizes(final int headSizeFull, final int headSizeSWA) {
      this.headSizeFull = headSizeFull;
      this.headSizeSWA = headSizeSWA;
      return this;
    }

    @Nonnull
    public Builder setSlidingWindow(final int slidingWindow) {
      this.slidingWindow = slidingWindow;
      return this;
    }

    @Nonnull
    public Builder setLogitSoftcapping(final float logitSoftcapping) {
      this.logitSoftcapping = logitSoftcapping;
      return this;
    }

    @Nonnull
    public Builder setIsSWA(@Nonnull final boolean[] isSWA) {
      this.isSWA = isSWA.clone();
      return this;
    }

    @Nonnull
    public Builder setSharedKvLayers(final int sharedKvLayers) {
      this.sharedKvLayers = sharedKvLayers;
      return this;
    }

    @Nonnull
    public Builder setEmbeddingLengthPerLayer(final int embeddingLengthPerLayer) {
      this.embeddingLengthPerLayer = embeddingLengthPerLayer;
      return this;
    }

    @Nonnull
    public Builder setMoE(final int expertCount, final int expertUsedCount, final int expertFeedForwardLength) {
      this.expertCount = expertCount;
      this.expertUsedCount = expertUsedCount;
      this.expertFeedForwardLength = expertFeedForwardLength;
      return this;
    }

    @Nonnull
    public Gemma4Config build() {
      int[] ffn = feedForwardLength != null ? feedForwardLength : new int[numberOfLayers];
      if (feedForwardLength == null) {
        Arrays.fill(ffn, defaultFeedForwardLength);
      }
      int[] kvh = kvHeadsPerLayer != null ? kvHeadsPerLayer : new int[numberOfLayers];
      if (kvHeadsPerLayer == null) {
        Arrays.fill(kvh, defaultKvHeads);
      }
      boolean[] swa = isSWA != null ? isSWA : new boolean[numberOfLayers];
      if (isSWA == null) {
        for (int i = 0; i < numberOfLayers; i++) {
          swa[i] = (i % 2 == 0);
        }
      }
      return new Gemma4Config(
          embeddingLength, ffn, numberOfLayers, numberOfHeads, kvh,
          vocabularySize, contextLength, rmsNormEps, ropeTheta, ropeThetaSWA,
          headSizeFull, headSizeSWA, slidingWindow, logitSoftcapping,
          swa, numberOfLayers - sharedKvLayers, embeddingLengthPerLayer,
          expertCount, expertUsedCount, expertFeedForwardLength
      );
    }
  }
}
