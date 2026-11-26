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

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.PointSample;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.ref.lang.RefIgnore;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * End-to-end equivalence, GGUF loading, tokenizer/chat formatting, generation, serialization,
 * and MindsEye optimization tests for Gemma 4.
 */
@RefIgnore
public class Gemma4ModelAndNetworkTest {

  private static void assertNetworkMatchesStatefulForward(final Gemma4Model model,
                                                          final int[] tokenSeq,
                                                          final boolean fp32KvCache,
                                                          final double tol) {
    Gemma4Config cfg = model.configuration();
    Gemma4Model.State state = model.createNewState(fp32KvCache);

    double[] seqDoubles = new double[tokenSeq.length];
    for (int t = 0; t < tokenSeq.length; t++) {
      seqDoubles[t] = tokenSeq[t];
    }

    PipelineNetwork network = model.toNetwork();
    try {
      Result res = network.eval(new Tensor(seqDoubles, tokenSeq.length));
      Tensor seqLogits = Result.getData0(res);
      try {
        Assertions.assertArrayEquals(new int[]{cfg.vocabularySize, tokenSeq.length}, seqLogits.getDimensions());
        double[] dagLogits = seqLogits.getData();

        for (int pos = 0; pos < tokenSeq.length; pos++) {
          GGUFTensor stepLogits = model.forward(state, tokenSeq[pos], pos);
          int base = pos * cfg.vocabularySize;
          for (int v = 0; v < cfg.vocabularySize; v++) {
            Assertions.assertEquals(stepLogits.getFloat(v), dagLogits[base + v], tol,
                "Logit mismatch at pos=" + pos + ", vocab=" + v);
          }
        }
      } finally {
        seqLogits.freeRef();
      }
    } finally {
      network.freeRef();
    }
  }

  @Test
  @Tag("UnitTest")
  public void testDensePleSwaSharedKvAndVEqualsKEquivalence() {
    // 4-layer config exercising ALL Gemma 4 dense architectural features simultaneously:
    // - SWA layers (0, 2) + Full attention layers (1, 3)
    // - slidingWindow = 4 with sequence length 6 (tests ring-buffer KV wrap-around & causal SWA window!)
    // - sharedKvLayers = 2 (layers 2 & 3 reuse KV from layers 0 & 1)
    // - V=K on odd layer 1 (includeAllVWeights = false)
    // - Per-layer input embeddings (embeddingLengthPerLayer = 6)
    // - Custom rope_freqs.weight factors + logitSoftcapping = 30.0 + tied embeddings
    Gemma4Config config = new Gemma4Config.Builder()
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

    Gemma4Weights weights = Gemma4Weights.randomInit(config, 12345L, false, true);
    Gemma4Model model = new Gemma4Model(config, null, weights);

    int[] tokens = new int[]{2, 7, 13, 4, 19, 11};
    assertNetworkMatchesStatefulForward(model, tokens, true, 1e-4);
    assertNetworkMatchesStatefulForward(model, tokens, false, 5e-3);

    // Verify PipelineNetwork copy/serialization preserves forward output
    PipelineNetwork net = model.toNetwork();
    Layer copied = net.copy(SerialPrecision.Double);
    Tensor input = new Tensor(new double[]{2, 7, 13}, 3);
    Tensor outOrig = Result.getData0(net.eval(input.copy()));
    Tensor outCopy = Result.getData0(copied.eval(input));
    for (int i = 0; i < outOrig.length(); i++) {
      Assertions.assertEquals(outOrig.getData()[i], outCopy.getData()[i], 1e-6);
    }
    outOrig.freeRef();
    outCopy.freeRef();
    copied.freeRef();
    net.freeRef();
  }

  @Test
  @Tag("UnitTest")
  public void testMixtureOfExpertsEquivalence() {
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

    int[] tokens = new int[]{1, 9, 5, 14, 3};
    assertNetworkMatchesStatefulForward(moeModel, tokens, true, 1e-4);
  }

  @Test
  @Tag("UnitTest")
  public void testMindsEyeFineTuningReducesLoss() {
    Gemma4Config tinyConfig = new Gemma4Config.Builder()
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

    Gemma4Weights weights = Gemma4Weights.randomInit(tinyConfig, 42L, false, true);
    Gemma4Model model = new Gemma4Model(tinyConfig, null, weights);
    PipelineNetwork network = model.toNetwork();

    Tensor inputTokens = new Tensor(new double[]{2, 4}, 2);
    Tensor targetLogits = new Tensor(10, 2);
    targetLogits.set(new int[]{5, 0}, 4.0);
    targetLogits.set(new int[]{7, 1}, 4.0);

    Tensor[][] trainingData = new Tensor[][]{{inputTokens, targetLogits}};
    SimpleLossNetwork supervised = new SimpleLossNetwork(network, new MeanSqLossLayer());
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
          "Expected training to reduce loss: initial=" + initialLoss + ", final=" + finalLoss);
    } finally {
      trainer.freeRef();
    }
  }

  private static byte[] quantizeToQ8_0(final float[] src) {
    if (src.length % 32 != 0) {
      throw new IllegalArgumentException("Length must be multiple of 32");
    }
    int blocks = src.length / 32;
    ByteBuffer buf = ByteBuffer.allocate(blocks * 34).order(ByteOrder.LITTLE_ENDIAN);
    for (int b = 0; b < blocks; b++) {
      float maxAbs = 0f;
      for (int i = 0; i < 32; i++) {
        maxAbs = Math.max(maxAbs, Math.abs(src[b * 32 + i]));
      }
      float d = maxAbs == 0f ? 0f : (maxAbs / 127f);
      buf.putShort(Float.floatToFloat16(d));
      float invD = d == 0f ? 0f : (1f / d);
      for (int i = 0; i < 32; i++) {
        int q = Math.max(-127, Math.min(127, Math.round(src[b * 32 + i] * invD)));
        buf.put((byte) q);
      }
    }
    return buf.array();
  }

  @Test
  @Tag("UnitTest")
  public void testFullGGUFModelLoadTokenizerChatAndGeneration(@TempDir final Path tempDir) throws IOException {
    // Build realistic vocabulary with specials + byte fallback <0x00>..<0xFF> + BPE merges
    List<String> vocabList = new ArrayList<>();
    List<Float> scoreList = new ArrayList<>();
    List<Integer> typeList = new ArrayList<>();

    String[] specials = {
        "<pad>", "<eos>", "<bos>", "<unk>",
        "<|turn>", "<|think|>", "<|channel>", "<channel|>",
        "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_separator|>",
        "<turn|>"
    };
    for (String sp : specials) {
      vocabList.add(sp);
      scoreList.add(0f);
      typeList.add(3);
    }
    for (int b = 0; b < 256; b++) {
      vocabList.add(String.format("<0x%02X>", b));
      scoreList.add(0f);
      typeList.add(6);
    }
    String[] pieces = {
        "\u2581", "\n", "s", "y", "t", "e", "m", "u", "r", "o", "d", "l", "h", "i",
        "sy", "st", "em", "syst", "system",
        "us", "er", "user",
        "mo", "de", "model",
        "\u2581hi", "hi"
    };
    float score = 1.0f;
    for (String p : pieces) {
      vocabList.add(p);
      scoreList.add(score++);
      typeList.add(1);
    }
    while (vocabList.size() % 32 != 0) {
      vocabList.add("tok_" + vocabList.size());
      scoreList.add(0.1f);
      typeList.add(1);
    }

    int vocabSize = vocabList.size();
    String[] tokensArr = vocabList.toArray(new String[0]);
    float[] scoresArr = new float[vocabSize];
    int[] typesArr = new int[vocabSize];
    for (int i = 0; i < vocabSize; i++) {
      scoresArr[i] = scoreList.get(i);
      typesArr[i] = typeList.get(i);
    }

    // Dim=32 so Q8_0 blockSize=32 divides every matrix cleanly!
    Gemma4Config cfg = new Gemma4Config.Builder()
        .setEmbeddingLength(32)
        .setDefaultFeedForwardLength(64)
        .setNumberOfLayers(2)
        .setNumberOfHeads(4)
        .setDefaultKvHeads(2)
        .setHeadSizes(8, 8)
        .setSlidingWindow(8)
        .setIsSWA(new boolean[]{true, false})
        .setVocabularySize(vocabSize)
        .setContextLength(64)
        .setLogitSoftcapping(30.0f)
        .build();
    Gemma4Weights w = Gemma4Weights.randomInit(cfg, 777L, true, true);

    GGUF.Builder builder = new GGUF.Builder()
        .putString("general.architecture", "gemma4")
        .putInt("gemma4.context_length", cfg.contextLength)
        .putInt("gemma4.embedding_length", cfg.embeddingLength)
        .putInt("gemma4.block_count", cfg.numberOfLayers)
        .putInt("gemma4.attention.head_count", cfg.numberOfHeads)
        .putInt("gemma4.attention.key_length", cfg.headSizeFull)
        .putInt("gemma4.attention.key_length_swa", cfg.headSizeSWA)
        .putInt("gemma4.attention.sliding_window", cfg.slidingWindow)
        .putFloat("gemma4.final_logit_softcapping", cfg.logitSoftcapping)
        .putFloat("gemma4.attention.layer_norm_rms_epsilon", cfg.rmsNormEps)
        .putFloat("gemma4.rope.freq_base", cfg.ropeTheta)
        .putFloat("gemma4.rope.freq_base_swa", cfg.ropeThetaSWA)
        .putInt("gemma4.feed_forward_length", 64)
        .putBooleanArray("gemma4.attention.sliding_window_pattern", cfg.isSWA)
        .putStringArray("tokenizer.ggml.tokens", tokensArr)
        .putFloatArray("tokenizer.ggml.scores", scoresArr)
        .putIntArray("tokenizer.ggml.token_type", typesArr)
        .addRawTensor("token_embd.weight", new int[]{32, vocabSize}, GGMLType.Q8_0, quantizeToQ8_0(w.tokenEmbeddingTable.toFloatArray()))
        .addFloatTensor("output_norm.weight", new int[]{32}, w.rmsFinalWeight)
        .addFloatTensor("rope_freqs.weight", new int[]{4}, w.ropeFreqFactors);

    for (int l = 0; l < 2; l++) {
      builder
          .addFloatTensor("blk." + l + ".attn_norm.weight", new int[]{32}, w.rmsAttWeight[l])
          .addRawTensor("blk." + l + ".attn_q.weight", new int[]{32, 32}, GGMLType.Q8_0, quantizeToQ8_0(w.wq[l].toFloatArray()))
          .addRawTensor("blk." + l + ".attn_k.weight", new int[]{32, 16}, GGMLType.Q8_0, quantizeToQ8_0(w.wk[l].toFloatArray()))
          .addRawTensor("blk." + l + ".attn_v.weight", new int[]{32, 16}, GGMLType.Q8_0, quantizeToQ8_0(w.wv[l].toFloatArray()))
          .addRawTensor("blk." + l + ".attn_output.weight", new int[]{32, 32}, GGMLType.Q8_0, quantizeToQ8_0(w.wo[l].toFloatArray()))
          .addFloatTensor("blk." + l + ".attn_q_norm.weight", new int[]{8}, w.attnQNorm[l])
          .addFloatTensor("blk." + l + ".attn_k_norm.weight", new int[]{8}, w.attnKNorm[l])
          .addFloatTensor("blk." + l + ".post_attention_norm.weight", new int[]{32}, w.postAttentionNorm[l])
          .addFloatTensor("blk." + l + ".ffn_norm.weight", new int[]{32}, w.rmsFfnWeight[l])
          .addRawTensor("blk." + l + ".ffn_gate.weight", new int[]{32, 64}, GGMLType.Q8_0, quantizeToQ8_0(w.w1[l].toFloatArray()))
          .addRawTensor("blk." + l + ".ffn_down.weight", new int[]{64, 32}, GGMLType.Q8_0, quantizeToQ8_0(w.w2[l].toFloatArray()))
          .addRawTensor("blk." + l + ".ffn_up.weight", new int[]{32, 64}, GGMLType.Q8_0, quantizeToQ8_0(w.w3[l].toFloatArray()))
          .addFloatTensor("blk." + l + ".post_ffw_norm.weight", new int[]{32}, w.postFfwNorm[l])
          .addFloatTensor("blk." + l + ".layer_output_scale.weight", new int[]{1}, new float[]{w.layerOutputScale[l]});
    }

    Path ggufPath = tempDir.resolve("mini_gemma4_q8.gguf");
    builder.writeTo(ggufPath);

    Gemma4Model loaded = Gemma4Model.load(ggufPath, 32);
    GemmaTokenizer tok = loaded.tokenizer();
    Assertions.assertNotNull(tok);

    // Tokenizer BPE + UTF-8 byte fallback round-trip
    String text = "system user hi!";
    List<Integer> encoded = tok.encode(text);
    Assertions.assertEquals(text, tok.decode(encoded));

    // Chat format: system thinking + FIM + thought channel stripping
    GemmaChatFormat chat = new GemmaChatFormat(tok);
    List<Integer> sysThink = chat.encodeSystemThinkingTurn("hi");
    Assertions.assertTrue(sysThink.contains(tok.getSpecialTokens().get("<|think|>")));
    List<Integer> fim = chat.encodeFillInTheMiddle("hi", "user");
    Assertions.assertEquals(tok.getSpecialTokens().get("<|fim_prefix|>"), fim.get(0));
    Set<Integer> stops = chat.getStopTokens();
    Assertions.assertTrue(stops.contains(chat.getEndOfTurn()));

    int chOpen = tok.getSpecialTokens().get("<|channel>");
    int chClose = tok.getSpecialTokens().get("<channel|>");
    List<Integer> withThoughts = List.of(chOpen, encoded.get(0), chClose, encoded.get(encoded.size() - 1));
    List<Integer> stripped = GemmaChatFormat.stripThoughtChannelTokens(tok, withThoughts);
    Assertions.assertEquals(List.of(encoded.get(encoded.size() - 1)), stripped);

    // Generation (argmax and top-p) + DAG equivalence on Q8_0 GGUF model
    String responseArgmax = loaded.generate("hi", null, 4, 0.0f, 0.95f, 1L, false);
    Assertions.assertNotNull(responseArgmax);
    String responseTopP = loaded.generate("hi", "system", 3, 0.8f, 0.9f, 42L, true);
    Assertions.assertNotNull(responseTopP);

    assertNetworkMatchesStatefulForward(loaded, new int[]{2, 15, 20}, true, 1e-4);
  }
}
