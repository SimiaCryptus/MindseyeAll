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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.ChunkSelectLayer;
import com.simiacryptus.mindseye.layers.java.GeluActivationLayer;
import com.simiacryptus.mindseye.layers.java.GroupedQueryAttentionLayer;
import com.simiacryptus.mindseye.layers.java.LinearProjectionLayer;
import com.simiacryptus.mindseye.layers.java.ProductInputsLayer;
import com.simiacryptus.mindseye.layers.java.RmsNormLayer;
import com.simiacryptus.mindseye.layers.java.RopeLayer;
import com.simiacryptus.mindseye.layers.java.ScalarScaleLayer;
import com.simiacryptus.mindseye.layers.java.SoftcapLayer;
import com.simiacryptus.mindseye.layers.java.SumInputsLayer;
import com.simiacryptus.mindseye.layers.java.TokenEmbeddingLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;

/**
 * Constructs a differentiable MindsEye {@link PipelineNetwork} ({@code DAGNetwork}) implementing the full
 * Gemma 4 transformer architecture (dense GeGLU, MoE, SWA + full attention, dual RoPE, Q/K/V RMSNorm,
 * V=K fallback, shared KV layers, per-layer embeddings, per-layer output scale, tied/untied LM head, and softcap).
 */
@RefIgnore
public final class Gemma4Network {

  private Gemma4Network() {
  }

  /**
   * Builds a 1-input differentiable {@link PipelineNetwork} mapping token IDs {@code Tensor(seqLen)}
   * to logits {@code Tensor(vocabSize, seqLen)}.
   */
  @Nonnull
  public static PipelineNetwork buildNetwork(@Nonnull final Gemma4Config config,
                                             @Nonnull final Gemma4Weights weights) {
    PipelineNetwork network = new PipelineNetwork(1);
    int dim = config.embeddingLength;
    int vocabSize = config.vocabularySize;
    int nLayers = config.numberOfLayers;

    Tensor tokenEmbdTensor = weights.tokenEmbeddingTable.toMindsEyeTensor(dim, vocabSize);
    Tensor wclsTensor = weights.hasTiedEmbeddings()
        ? tokenEmbdTensor.addRef()
        : weights.wcls.toMindsEyeTensor(dim, vocabSize);

    DAGNode tokenIdsNode = network.getInput(0);
    TokenEmbeddingLayer tokEmbdLayer = new TokenEmbeddingLayer(vocabSize, dim, Math.sqrt(dim), tokenEmbdTensor);
    tokEmbdLayer.setName("token_embd");
    DAGNode xNode = network.add("token_embd", tokEmbdLayer, tokenIdsNode.addRef());

    // Optional Per-Layer Embeddings (PLE)
    DAGNode perLayerInputsNode = null;
    int plDim = config.embeddingLengthPerLayer;
    if (plDim > 0 && weights.perLayerTokenEmbd != null && weights.perLayerModelProj != null && weights.perLayerProjNorm != null) {
      int plTotal = plDim * nLayers;
      LinearProjectionLayer plModelProjLayer = new LinearProjectionLayer(
          dim, plTotal, weights.perLayerModelProj.toMindsEyeTensor(dim, plTotal));
      plModelProjLayer.setName("per_layer_model_proj");
      DAGNode projNode = network.add(plModelProjLayer, xNode.addRef());

      ScalarScaleLayer projScaleLayer = new ScalarScaleLayer(1.0 / Math.sqrt(dim), true);
      projScaleLayer.setName("per_layer_proj_scale");
      projNode = network.add(projScaleLayer, projNode);

      RmsNormLayer projNormLayer = new RmsNormLayer(plDim, config.rmsNormEps, true);
      projNormLayer.setWeights(weights.perLayerProjNorm);
      projNormLayer.setName("per_layer_proj_norm");
      projNode = network.add(projNormLayer, projNode);

      TokenEmbeddingLayer plTokLayer = new TokenEmbeddingLayer(
          vocabSize, plTotal, Math.sqrt(plDim), weights.perLayerTokenEmbd.toMindsEyeTensor(plTotal, vocabSize));
      plTokLayer.setName("per_layer_token_embd");
      DAGNode plTokNode = network.add(plTokLayer, tokenIdsNode.addRef());

      DAGNode combinedPl = network.add(new SumInputsLayer(), projNode, plTokNode);
      ScalarScaleLayer inputScaleLayer = new ScalarScaleLayer(1.0 / Math.sqrt(2.0), true);
      inputScaleLayer.setName("per_layer_input_scale");
      perLayerInputsNode = network.add("per_layer_inputs", inputScaleLayer, combinedPl);
    }
    tokenIdsNode.freeRef();

    DAGNode finalLogits = buildTransformerBackbone(network, xNode, perLayerInputsNode, wclsTensor, config, weights);
    network.setHead(finalLogits);
    disableNodeParallelism(network);
    return network;
  }

  /**
   * Builds a differentiable {@link PipelineNetwork} starting from continuous embeddings {@code Tensor(dim, seqLen)}
   * (useful for testing continuous input derivatives across the full transformer stack).
   */
  @Nonnull
  public static PipelineNetwork buildFromEmbeddingsNetwork(@Nonnull final Gemma4Config config,
                                                           @Nonnull final Gemma4Weights weights) {
    boolean hasPle = config.embeddingLengthPerLayer > 0
        && weights.perLayerTokenEmbd != null
        && weights.perLayerModelProj != null
        && weights.perLayerProjNorm != null;
    PipelineNetwork network = new PipelineNetwork(hasPle ? 2 : 1);
    int dim = config.embeddingLength;
    int nLayers = config.numberOfLayers;

    DAGNode xNode = network.getInput(0);
    DAGNode perLayerInputsNode = null;
    if (hasPle) {
      int plDim = config.embeddingLengthPerLayer;
      int plTotal = plDim * nLayers;
      LinearProjectionLayer plModelProjLayer = new LinearProjectionLayer(
          dim, plTotal, weights.perLayerModelProj.toMindsEyeTensor(dim, plTotal));
      DAGNode projNode = network.add(plModelProjLayer, xNode.addRef());
      projNode = network.add(new ScalarScaleLayer(1.0 / Math.sqrt(dim), true), projNode);
      RmsNormLayer projNormLayer = new RmsNormLayer(plDim, config.rmsNormEps, true);
      projNormLayer.setWeights(weights.perLayerProjNorm);
      projNode = network.add(projNormLayer, projNode);

      DAGNode plTokEmbdIn = network.getInput(1);
      DAGNode combinedPl = network.add(new SumInputsLayer(), projNode, plTokEmbdIn);
      perLayerInputsNode = network.add(new ScalarScaleLayer(1.0 / Math.sqrt(2.0), true), combinedPl);
    }

    Tensor wclsTensor = weights.wcls.toMindsEyeTensor(dim, config.vocabularySize);
    DAGNode finalLogits = buildTransformerBackbone(network, xNode, perLayerInputsNode, wclsTensor, config, weights);
    network.setHead(finalLogits);
    disableNodeParallelism(network);
    return network;
  }

  /**
   * Builds a CUDA/cuDNN-accelerated 1-input differentiable {@link PipelineNetwork} mapping token IDs
   * {@code Tensor(seqLen)} to logits {@code Tensor(vocabSize, seqLen)} using {@link CudaSettings#getDefaultPrecision()}.
   */
  @Nonnull
  public static PipelineNetwork buildCudaNetwork(@Nonnull final Gemma4Config config,
                                                 @Nonnull final Gemma4Weights weights) {
    return buildCudaNetwork(config, weights, CudaSettings.INSTANCE().getDefaultPrecision());
  }

  /**
   * Builds a CUDA/cuDNN-accelerated 1-input differentiable {@link PipelineNetwork} mapping token IDs
   * {@code Tensor(seqLen)} to logits {@code Tensor(vocabSize, seqLen)} at the requested GPU {@link Precision}.
   * Automatically falls back to CPU compatibility layers if {@code CudaSystem.isEnabled()} is {@code false}.
   */
  @Nonnull
  public static PipelineNetwork buildCudaNetwork(@Nonnull final Gemma4Config config,
                                                 @Nonnull final Gemma4Weights weights,
                                                 @Nonnull final Precision precision) {
    PipelineNetwork network = new PipelineNetwork(1);
    int dim = config.embeddingLength;
    int vocabSize = config.vocabularySize;
    int nLayers = config.numberOfLayers;

    Tensor tokenEmbdTensor = weights.tokenEmbeddingTable.toMindsEyeTensor(dim, vocabSize);
    Tensor wclsTensor = weights.hasTiedEmbeddings()
        ? tokenEmbdTensor.addRef()
        : weights.wcls.toMindsEyeTensor(dim, vocabSize);

    DAGNode tokenIdsNode = network.getInput(0);
    com.simiacryptus.mindseye.layers.cudnn.TokenEmbeddingLayer tokEmbdLayer =
        new com.simiacryptus.mindseye.layers.cudnn.TokenEmbeddingLayer(vocabSize, dim, Math.sqrt(dim), tokenEmbdTensor);
    tokEmbdLayer.setPrecision(precision);
    tokEmbdLayer.setName("token_embd");
    DAGNode xNode = network.add("token_embd", tokEmbdLayer, tokenIdsNode.addRef());

    DAGNode perLayerInputsNode = null;
    int plDim = config.embeddingLengthPerLayer;
    if (plDim > 0 && weights.perLayerTokenEmbd != null && weights.perLayerModelProj != null && weights.perLayerProjNorm != null) {
      int plTotal = plDim * nLayers;
      com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer plModelProjLayer =
          new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
              dim, plTotal, weights.perLayerModelProj.toMindsEyeTensor(dim, plTotal));
      plModelProjLayer.setPrecision(precision);
      plModelProjLayer.setName("per_layer_model_proj");
      DAGNode projNode = network.add(plModelProjLayer, xNode.addRef());

      com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer projScaleLayer =
          new com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer(1.0 / Math.sqrt(dim), true);
      projScaleLayer.setPrecision(precision);
      projScaleLayer.setName("per_layer_proj_scale");
      projNode = network.add(projScaleLayer, projNode);

      com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer projNormLayer =
          new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(plDim, config.rmsNormEps, true);
      projNormLayer.setWeights(weights.perLayerProjNorm);
      projNormLayer.setPrecision(precision);
      projNormLayer.setName("per_layer_proj_norm");
      projNode = network.add(projNormLayer, projNode);

      com.simiacryptus.mindseye.layers.cudnn.TokenEmbeddingLayer plTokLayer =
          new com.simiacryptus.mindseye.layers.cudnn.TokenEmbeddingLayer(
              vocabSize, plTotal, Math.sqrt(plDim), weights.perLayerTokenEmbd.toMindsEyeTensor(plTotal, vocabSize));
      plTokLayer.setPrecision(precision);
      plTokLayer.setName("per_layer_token_embd");
      DAGNode plTokNode = network.add(plTokLayer, tokenIdsNode.addRef());

      DAGNode combinedPl = network.add(newCudaSum(precision), projNode, plTokNode);
      com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer inputScaleLayer =
          new com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer(1.0 / Math.sqrt(2.0), true);
      inputScaleLayer.setPrecision(precision);
      inputScaleLayer.setName("per_layer_input_scale");
      perLayerInputsNode = network.add("per_layer_inputs", inputScaleLayer, combinedPl);
    }
    tokenIdsNode.freeRef();

    DAGNode finalLogits = buildCudaTransformerBackbone(
        network, xNode, perLayerInputsNode, wclsTensor, config, weights, precision);
    network.setHead(finalLogits);
    disableNodeParallelism(network);
    return network;
  }

  /**
   * Builds a CUDA/cuDNN-accelerated differentiable {@link PipelineNetwork} starting from continuous embeddings.
   */
  @Nonnull
  public static PipelineNetwork buildCudaFromEmbeddingsNetwork(@Nonnull final Gemma4Config config,
                                                               @Nonnull final Gemma4Weights weights,
                                                               @Nonnull final Precision precision) {
    boolean hasPle = config.embeddingLengthPerLayer > 0
        && weights.perLayerTokenEmbd != null
        && weights.perLayerModelProj != null
        && weights.perLayerProjNorm != null;
    PipelineNetwork network = new PipelineNetwork(hasPle ? 2 : 1);
    int dim = config.embeddingLength;
    int nLayers = config.numberOfLayers;

    DAGNode xNode = network.getInput(0);
    DAGNode perLayerInputsNode = null;
    if (hasPle) {
      int plDim = config.embeddingLengthPerLayer;
      int plTotal = plDim * nLayers;
      com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer plModelProjLayer =
          new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
              dim, plTotal, weights.perLayerModelProj.toMindsEyeTensor(dim, plTotal));
      plModelProjLayer.setPrecision(precision);
      DAGNode projNode = network.add(plModelProjLayer, xNode.addRef());

      com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer projScaleLayer =
          new com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer(1.0 / Math.sqrt(dim), true);
      projScaleLayer.setPrecision(precision);
      projNode = network.add(projScaleLayer, projNode);

      com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer projNormLayer =
          new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(plDim, config.rmsNormEps, true);
      projNormLayer.setWeights(weights.perLayerProjNorm);
      projNormLayer.setPrecision(precision);
      projNode = network.add(projNormLayer, projNode);

      DAGNode plTokEmbdIn = network.getInput(1);
      DAGNode combinedPl = network.add(newCudaSum(precision), projNode, plTokEmbdIn);
      com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer inputScaleLayer =
          new com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer(1.0 / Math.sqrt(2.0), true);
      inputScaleLayer.setPrecision(precision);
      perLayerInputsNode = network.add(inputScaleLayer, combinedPl);
    }

    Tensor wclsTensor = weights.wcls.toMindsEyeTensor(dim, config.vocabularySize);
    DAGNode finalLogits = buildCudaTransformerBackbone(
        network, xNode, perLayerInputsNode, wclsTensor, config, weights, precision);
    network.setHead(finalLogits);
    disableNodeParallelism(network);
    return network;
  }

  private static com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer newCudaSum(@Nonnull final Precision precision) {
    com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer sum =
        new com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer();
    sum.setPrecision(precision);
    sum.setParallel(false);
    return sum;
  }

  private static com.simiacryptus.mindseye.layers.cudnn.ProductLayer newCudaProd(@Nonnull final Precision precision) {
    com.simiacryptus.mindseye.layers.cudnn.ProductLayer prod =
        new com.simiacryptus.mindseye.layers.cudnn.ProductLayer();
    prod.setPrecision(precision);
    return prod;
  }

  private static com.simiacryptus.mindseye.layers.cudnn.GeluActivationLayer newCudaGelu(@Nonnull final Precision precision) {
    com.simiacryptus.mindseye.layers.cudnn.GeluActivationLayer gelu =
        new com.simiacryptus.mindseye.layers.cudnn.GeluActivationLayer();
    gelu.setPrecision(precision);
    return gelu;
  }

  private static void disableNodeParallelism(@Nonnull final PipelineNetwork network) {
    com.simiacryptus.ref.wrappers.RefList<DAGNode> nodes = network.getNodes();
    nodes.forEach(node -> {
      if (node instanceof InnerNode) {
        ((InnerNode) node).setParallel(false);
      }
      node.freeRef();
    });
    nodes.freeRef();
  }

  @Nonnull
  private static DAGNode buildTransformerBackbone(@Nonnull final PipelineNetwork network,
                                                  @Nonnull final DAGNode initialX,
                                                  final DAGNode perLayerInputsNode,
                                                  @Nonnull final Tensor wclsTensor,
                                                  @Nonnull final Gemma4Config config,
                                                  @Nonnull final Gemma4Weights weights) {
    int dim = config.embeddingLength;
    int nLayers = config.numberOfLayers;
    double eps = config.rmsNormEps;
    double[] fullRopeFactors = weights.getRopeFreqFactorsAsDouble();
    int plDim = config.embeddingLengthPerLayer;

    DAGNode x = initialX;
    DAGNode[] savedK = new DAGNode[config.nLayerKvFromStart];
    DAGNode[] savedV = new DAGNode[config.nLayerKvFromStart];

    for (int l = 0; l < nLayers; l++) {
      boolean layerIsSwa = config.isSWA[l];
      int headSize = config.headSize(l);
      int queryDim = config.queryDim(l);
      int kvDim = config.kvDim(l);
      int nKvHeads = config.numberOfKeyValueHeads(l);
      int hiddenDim = config.feedForwardLength[l];
      double ropeTheta = layerIsSwa ? config.ropeThetaSWA : config.ropeTheta;
      double[] layerRopeFactors = layerIsSwa ? null : fullRopeFactors;

      // 1. Pre-attention RMSNorm
      RmsNormLayer attNorm = new RmsNormLayer(dim, eps, true);
      attNorm.setWeights(weights.rmsAttWeight[l]);
      attNorm.setName("blk." + l + ".attn_norm");
      DAGNode xb = network.add(attNorm, x.addRef());

      // 2. Q projection + per-head Q norm + RoPE
      LinearProjectionLayer wqLayer = new LinearProjectionLayer(
          dim, queryDim, weights.wq[l].toMindsEyeTensor(dim, queryDim));
      wqLayer.setName("blk." + l + ".attn_q");
      DAGNode qNode = network.add(wqLayer, xb.addRef());

      RmsNormLayer qNormLayer = new RmsNormLayer(headSize, eps, true);
      qNormLayer.setWeights(weights.attnQNorm[l]);
      qNormLayer.setName("blk." + l + ".attn_q_norm");
      qNode = network.add(qNormLayer, qNode);

      RopeLayer qRopeLayer = new RopeLayer(config.numberOfHeads, headSize, ropeTheta, layerRopeFactors, 0);
      qRopeLayer.setName("blk." + l + ".attn_q_rope");
      qNode = network.add(qRopeLayer, qNode);

      // 3. K and V projections (or shared KV reuse)
      DAGNode kForAttn;
      DAGNode vForAttn;
      if (config.hasKv(l)) {
        LinearProjectionLayer wkLayer = new LinearProjectionLayer(
            dim, kvDim, weights.wk[l].toMindsEyeTensor(dim, kvDim));
        wkLayer.setName("blk." + l + ".attn_k");
        DAGNode kRaw = network.add(wkLayer, xb.addRef());

        DAGNode vRaw;
        if (weights.wv[l] != null) {
          LinearProjectionLayer wvLayer = new LinearProjectionLayer(
              dim, kvDim, weights.wv[l].toMindsEyeTensor(dim, kvDim));
          wvLayer.setName("blk." + l + ".attn_v");
          vRaw = network.add(wvLayer, xb.addRef());
        } else {
          vRaw = kRaw.addRef();
        }

        RmsNormLayer kNormLayer = new RmsNormLayer(headSize, eps, true);
        kNormLayer.setWeights(weights.attnKNorm[l]);
        kNormLayer.setName("blk." + l + ".attn_k_norm");
        DAGNode kNormed = network.add(kNormLayer, kRaw);

        RopeLayer kRopeLayer = new RopeLayer(nKvHeads, headSize, ropeTheta, layerRopeFactors, 0);
        kRopeLayer.setName("blk." + l + ".attn_k_rope");
        DAGNode kFinal = network.add(kRopeLayer, kNormed);

        RmsNormLayer vNormLayer = new RmsNormLayer(headSize, eps, false);
        vNormLayer.setName("blk." + l + ".attn_v_norm");
        DAGNode vFinal = network.add(vNormLayer, vRaw);

        savedK[l] = kFinal.addRef();
        savedV[l] = vFinal.addRef();
        kForAttn = kFinal;
        vForAttn = vFinal;
      } else {
        int srcLayer = config.kvSourceLayer(l);
        kForAttn = savedK[srcLayer].addRef();
        vForAttn = savedV[srcLayer].addRef();
      }
      xb.freeRef();

      // 4. Grouped-Query Attention (scale=1.0, causal=true, slidingWindow on SWA layers)
      int swaWin = layerIsSwa ? config.slidingWindow : 0;
      GroupedQueryAttentionLayer gqaLayer = new GroupedQueryAttentionLayer(
          config.numberOfHeads, nKvHeads, headSize, 1.0, true, swaWin);
      gqaLayer.setName("blk." + l + ".attn");
      DAGNode attnOut = network.add(gqaLayer, qNode, kForAttn, vForAttn);

      // 5. Output projection + post-attention RMSNorm + residual
      LinearProjectionLayer woLayer = new LinearProjectionLayer(
          queryDim, dim, weights.wo[l].toMindsEyeTensor(queryDim, dim));
      woLayer.setName("blk." + l + ".attn_output");
      DAGNode attnProj = network.add(woLayer, attnOut);

      RmsNormLayer postAttNorm = new RmsNormLayer(dim, eps, true);
      postAttNorm.setWeights(weights.postAttentionNorm[l]);
      postAttNorm.setName("blk." + l + ".post_attention_norm");
      attnProj = network.add(postAttNorm, attnProj);

      x = network.add("blk." + l + ".attn_res", new SumInputsLayer(), x, attnProj);

      // 6. Feed-Forward Network (MoE or Dense GeGLU)
      boolean isMoELayer = config.isMoE() && weights.ffnGateInp != null && weights.ffnGateInp[l] != null;
      if (isMoELayer) {
        RmsNormLayer ffnNorm = new RmsNormLayer(dim, eps, true);
        ffnNorm.setWeights(weights.rmsFfnWeight[l]);
        ffnNorm.setName("blk." + l + ".ffn_norm");
        DAGNode xbFfn = network.add(ffnNorm, x.addRef());

        LinearProjectionLayer w1Layer = new LinearProjectionLayer(
            dim, hiddenDim, weights.w1[l].toMindsEyeTensor(dim, hiddenDim));
        DAGNode gate = network.add(w1Layer, xbFfn.addRef());
        gate = network.add(new GeluActivationLayer(), gate);

        LinearProjectionLayer w3Layer = new LinearProjectionLayer(
            dim, hiddenDim, weights.w3[l].toMindsEyeTensor(dim, hiddenDim));
        DAGNode up = network.add(w3Layer, xbFfn);

        DAGNode gated = network.add(new ProductInputsLayer(), gate, up);
        LinearProjectionLayer w2Layer = new LinearProjectionLayer(
            hiddenDim, dim, weights.w2[l].toMindsEyeTensor(hiddenDim, dim));
        DAGNode sharedMlp = network.add(w2Layer, gated);

        RmsNormLayer postNorm1 = new RmsNormLayer(dim, eps, true);
        postNorm1.setWeights(weights.ffnPostNorm1[l]);
        sharedMlp = network.add(postNorm1, sharedMlp);

        int nExp = config.expertCount;
        int expFF = config.expertFeedForwardLength;
        GemmaMoELayer moeLayer = new GemmaMoELayer(
            dim, nExp, config.expertUsedCount, expFF, eps,
            to1DTensor(weights.preFfwNorm2[l]),
            to1DTensor(weights.ffnGateInpScale[l]),
            weights.ffnGateInp[l].toMindsEyeTensor(dim, nExp),
            weights.ffnGateUpExps[l].toMindsEyeTensor(dim, 2 * expFF * nExp),
            weights.ffnDownExps[l].toMindsEyeTensor(expFF, dim * nExp),
            to1DTensor(weights.ffnDownExpsScale[l]),
            to1DTensor(weights.ffnPostNorm2[l])
        );
        moeLayer.setName("blk." + l + ".moe");
        DAGNode moeOut = network.add(moeLayer, x.addRef());

        DAGNode combinedFfw = network.add(new SumInputsLayer(), sharedMlp, moeOut);
        RmsNormLayer postFfwNorm = new RmsNormLayer(dim, eps, true);
        postFfwNorm.setWeights(weights.postFfwNorm[l]);
        combinedFfw = network.add(postFfwNorm, combinedFfw);
        x = network.add("blk." + l + ".ffn_res", new SumInputsLayer(), x, combinedFfw);
      } else {
        RmsNormLayer ffnNorm = new RmsNormLayer(dim, eps, true);
        ffnNorm.setWeights(weights.rmsFfnWeight[l]);
        ffnNorm.setName("blk." + l + ".ffn_norm");
        DAGNode xbFfn = network.add(ffnNorm, x.addRef());

        LinearProjectionLayer w1Layer = new LinearProjectionLayer(
            dim, hiddenDim, weights.w1[l].toMindsEyeTensor(dim, hiddenDim));
        w1Layer.setName("blk." + l + ".ffn_gate");
        DAGNode gate = network.add(w1Layer, xbFfn.addRef());
        gate = network.add(new GeluActivationLayer(), gate);

        LinearProjectionLayer w3Layer = new LinearProjectionLayer(
            dim, hiddenDim, weights.w3[l].toMindsEyeTensor(dim, hiddenDim));
        w3Layer.setName("blk." + l + ".ffn_up");
        DAGNode up = network.add(w3Layer, xbFfn);

        DAGNode gated = network.add(new ProductInputsLayer(), gate, up);
        LinearProjectionLayer w2Layer = new LinearProjectionLayer(
            hiddenDim, dim, weights.w2[l].toMindsEyeTensor(hiddenDim, dim));
        w2Layer.setName("blk." + l + ".ffn_down");
        DAGNode ffnOut = network.add(w2Layer, gated);

        RmsNormLayer postFfwNorm = new RmsNormLayer(dim, eps, true);
        postFfwNorm.setWeights(weights.postFfwNorm[l]);
        postFfwNorm.setName("blk." + l + ".post_ffw_norm");
        ffnOut = network.add(postFfwNorm, ffnOut);

        x = network.add("blk." + l + ".ffn_res", new SumInputsLayer(), x, ffnOut);
      }

      // 7. Optional Per-Layer Embedding GELU-gated projection
      if (perLayerInputsNode != null && plDim > 0 && weights.perLayerInpGate != null && weights.perLayerInpGate[l] != null) {
        ChunkSelectLayer sliceLayer = new ChunkSelectLayer(plDim, nLayers, l);
        DAGNode plSlice = network.add(sliceLayer, perLayerInputsNode.addRef());

        LinearProjectionLayer inpGateLayer = new LinearProjectionLayer(
            dim, plDim, weights.perLayerInpGate[l].toMindsEyeTensor(dim, plDim));
        inpGateLayer.setName("blk." + l + ".inp_gate");
        DAGNode plGate = network.add(inpGateLayer, x.addRef());
        plGate = network.add(new GeluActivationLayer(), plGate);

        DAGNode plMod = network.add(new ProductInputsLayer(), plGate, plSlice);
        LinearProjectionLayer projLayer = new LinearProjectionLayer(
            plDim, dim, weights.perLayerProj[l].toMindsEyeTensor(plDim, dim));
        projLayer.setName("blk." + l + ".proj");
        DAGNode plProj = network.add(projLayer, plMod);

        RmsNormLayer postPlNorm = new RmsNormLayer(dim, eps, true);
        postPlNorm.setWeights(weights.perLayerPostNorm[l]);
        postPlNorm.setName("blk." + l + ".post_norm");
        plProj = network.add(postPlNorm, plProj);

        x = network.add("blk." + l + ".ple_res", new SumInputsLayer(), x, plProj);
      }

      // 8. Layer output scale
      float scale = weights.layerOutputScale[l];
      if (scale != 1.0f) {
        ScalarScaleLayer scaleLayer = new ScalarScaleLayer(scale, false);
        scaleLayer.setName("blk." + l + ".layer_output_scale");
        x = network.add(scaleLayer, x);
      }
    }

    RefUtil.freeRef(savedK);
    RefUtil.freeRef(savedV);
    if (perLayerInputsNode != null) {
      perLayerInputsNode.freeRef();
    }

    // 9. Final RMSNorm + LM Head + Logit Softcapping
    RmsNormLayer finalNorm = new RmsNormLayer(dim, eps, true);
    finalNorm.setWeights(weights.rmsFinalWeight);
    finalNorm.setName("output_norm");
    DAGNode normedOut = network.add("output_norm", finalNorm, x);

    LinearProjectionLayer lmHead = new LinearProjectionLayer(dim, config.vocabularySize, wclsTensor);
    lmHead.setName("output");
    DAGNode logits = network.add("output", lmHead, normedOut);

    if (config.logitSoftcapping > 0f) {
      SoftcapLayer softcap = new SoftcapLayer(config.logitSoftcapping);
      softcap.setName("final_logit_softcapping");
      logits = network.add("final_logit_softcapping", softcap, logits);
    }
    return logits;
  }

  @Nonnull
  private static DAGNode buildCudaTransformerBackbone(@Nonnull final PipelineNetwork network,
                                                      @Nonnull final DAGNode initialX,
                                                      final DAGNode perLayerInputsNode,
                                                      @Nonnull final Tensor wclsTensor,
                                                      @Nonnull final Gemma4Config config,
                                                      @Nonnull final Gemma4Weights weights,
                                                      @Nonnull final Precision precision) {
    int dim = config.embeddingLength;
    int nLayers = config.numberOfLayers;
    double eps = config.rmsNormEps;
    double[] fullRopeFactors = weights.getRopeFreqFactorsAsDouble();
    int plDim = config.embeddingLengthPerLayer;

    DAGNode x = initialX;
    DAGNode[] savedK = new DAGNode[config.nLayerKvFromStart];
    DAGNode[] savedV = new DAGNode[config.nLayerKvFromStart];

    for (int l = 0; l < nLayers; l++) {
      boolean layerIsSwa = config.isSWA[l];
      int headSize = config.headSize(l);
      int queryDim = config.queryDim(l);
      int kvDim = config.kvDim(l);
      int nKvHeads = config.numberOfKeyValueHeads(l);
      int hiddenDim = config.feedForwardLength[l];
      double ropeTheta = layerIsSwa ? config.ropeThetaSWA : config.ropeTheta;
      double[] layerRopeFactors = layerIsSwa ? null : fullRopeFactors;

      // 1. Pre-attention RMSNorm
      com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer attNorm =
          new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
      attNorm.setWeights(weights.rmsAttWeight[l]);
      attNorm.setPrecision(precision);
      attNorm.setName("blk." + l + ".attn_norm");
      DAGNode xb = network.add(attNorm, x.addRef());

      // 2. Q projection + per-head Q norm + RoPE
      com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer wqLayer =
          new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
              dim, queryDim, weights.wq[l].toMindsEyeTensor(dim, queryDim));
      wqLayer.setPrecision(precision);
      wqLayer.setName("blk." + l + ".attn_q");
      DAGNode qNode = network.add(wqLayer, xb.addRef());

      com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer qNormLayer =
          new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(headSize, eps, true);
      qNormLayer.setWeights(weights.attnQNorm[l]);
      qNormLayer.setPrecision(precision);
      qNormLayer.setName("blk." + l + ".attn_q_norm");
      qNode = network.add(qNormLayer, qNode);

      com.simiacryptus.mindseye.layers.cudnn.RopeLayer qRopeLayer =
          new com.simiacryptus.mindseye.layers.cudnn.RopeLayer(config.numberOfHeads, headSize, ropeTheta, layerRopeFactors, 0);
      qRopeLayer.setPrecision(precision);
      qRopeLayer.setName("blk." + l + ".attn_q_rope");
      qNode = network.add(qRopeLayer, qNode);

      // 3. K and V projections (or shared KV reuse)
      DAGNode kForAttn;
      DAGNode vForAttn;
      if (config.hasKv(l)) {
        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer wkLayer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                dim, kvDim, weights.wk[l].toMindsEyeTensor(dim, kvDim));
        wkLayer.setPrecision(precision);
        wkLayer.setName("blk." + l + ".attn_k");
        DAGNode kRaw = network.add(wkLayer, xb.addRef());

        DAGNode vRaw;
        if (weights.wv[l] != null) {
          com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer wvLayer =
              new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                  dim, kvDim, weights.wv[l].toMindsEyeTensor(dim, kvDim));
          wvLayer.setPrecision(precision);
          wvLayer.setName("blk." + l + ".attn_v");
          vRaw = network.add(wvLayer, xb.addRef());
        } else {
          vRaw = kRaw.addRef();
        }

        com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer kNormLayer =
            new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(headSize, eps, true);
        kNormLayer.setWeights(weights.attnKNorm[l]);
        kNormLayer.setPrecision(precision);
        kNormLayer.setName("blk." + l + ".attn_k_norm");
        DAGNode kNormed = network.add(kNormLayer, kRaw);

        com.simiacryptus.mindseye.layers.cudnn.RopeLayer kRopeLayer =
            new com.simiacryptus.mindseye.layers.cudnn.RopeLayer(nKvHeads, headSize, ropeTheta, layerRopeFactors, 0);
        kRopeLayer.setPrecision(precision);
        kRopeLayer.setName("blk." + l + ".attn_k_rope");
        DAGNode kFinal = network.add(kRopeLayer, kNormed);

        com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer vNormLayer =
            new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(headSize, eps, false);
        vNormLayer.setPrecision(precision);
        vNormLayer.setName("blk." + l + ".attn_v_norm");
        DAGNode vFinal = network.add(vNormLayer, vRaw);

        savedK[l] = kFinal.addRef();
        savedV[l] = vFinal.addRef();
        kForAttn = kFinal;
        vForAttn = vFinal;
      } else {
        int srcLayer = config.kvSourceLayer(l);
        kForAttn = savedK[srcLayer].addRef();
        vForAttn = savedV[srcLayer].addRef();
      }
      xb.freeRef();

      // 4. Grouped-Query Attention
      int swaWin = layerIsSwa ? config.slidingWindow : 0;
      com.simiacryptus.mindseye.layers.cudnn.GroupedQueryAttentionLayer gqaLayer =
          new com.simiacryptus.mindseye.layers.cudnn.GroupedQueryAttentionLayer(
              config.numberOfHeads, nKvHeads, headSize, 1.0, true, swaWin);
      gqaLayer.setPrecision(precision);
      gqaLayer.setName("blk." + l + ".attn");
      DAGNode attnOut = network.add(gqaLayer, qNode, kForAttn, vForAttn);

      // 5. Output projection + post-attention RMSNorm + residual
      com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer woLayer =
          new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
              queryDim, dim, weights.wo[l].toMindsEyeTensor(queryDim, dim));
      woLayer.setPrecision(precision);
      woLayer.setName("blk." + l + ".attn_output");
      DAGNode attnProj = network.add(woLayer, attnOut);

      com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer postAttNorm =
          new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
      postAttNorm.setWeights(weights.postAttentionNorm[l]);
      postAttNorm.setPrecision(precision);
      postAttNorm.setName("blk." + l + ".post_attention_norm");
      attnProj = network.add(postAttNorm, attnProj);

      x = network.add("blk." + l + ".attn_res", newCudaSum(precision), x, attnProj);

      // 6. Feed-Forward Network (MoE or Dense GeGLU)
      boolean isMoELayer = config.isMoE() && weights.ffnGateInp != null && weights.ffnGateInp[l] != null;
      if (isMoELayer) {
        com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer ffnNorm =
            new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
        ffnNorm.setWeights(weights.rmsFfnWeight[l]);
        ffnNorm.setPrecision(precision);
        ffnNorm.setName("blk." + l + ".ffn_norm");
        DAGNode xbFfn = network.add(ffnNorm, x.addRef());

        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer w1Layer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                dim, hiddenDim, weights.w1[l].toMindsEyeTensor(dim, hiddenDim));
        w1Layer.setPrecision(precision);
        DAGNode gate = network.add(w1Layer, xbFfn.addRef());
        gate = network.add(newCudaGelu(precision), gate);

        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer w3Layer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                dim, hiddenDim, weights.w3[l].toMindsEyeTensor(dim, hiddenDim));
        w3Layer.setPrecision(precision);
        DAGNode up = network.add(w3Layer, xbFfn);

        DAGNode gated = network.add(newCudaProd(precision), gate, up);
        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer w2Layer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                hiddenDim, dim, weights.w2[l].toMindsEyeTensor(hiddenDim, dim));
        w2Layer.setPrecision(precision);
        DAGNode sharedMlp = network.add(w2Layer, gated);

        com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer postNorm1 =
            new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
        postNorm1.setWeights(weights.ffnPostNorm1[l]);
        postNorm1.setPrecision(precision);
        sharedMlp = network.add(postNorm1, sharedMlp);

        int nExp = config.expertCount;
        int expFF = config.expertFeedForwardLength;
        CudaGemmaMoELayer moeLayer = new CudaGemmaMoELayer(
            dim, nExp, config.expertUsedCount, expFF, eps,
            to1DTensor(weights.preFfwNorm2[l]),
            to1DTensor(weights.ffnGateInpScale[l]),
            weights.ffnGateInp[l].toMindsEyeTensor(dim, nExp),
            weights.ffnGateUpExps[l].toMindsEyeTensor(dim, 2 * expFF * nExp),
            weights.ffnDownExps[l].toMindsEyeTensor(expFF, dim * nExp),
            to1DTensor(weights.ffnDownExpsScale[l]),
            to1DTensor(weights.ffnPostNorm2[l])
        );
        moeLayer.setPrecision(precision);
        moeLayer.setName("blk." + l + ".moe");
        DAGNode moeOut = network.add(moeLayer, x.addRef());

        DAGNode combinedFfw = network.add(newCudaSum(precision), sharedMlp, moeOut);
        com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer postFfwNorm =
            new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
        postFfwNorm.setWeights(weights.postFfwNorm[l]);
        postFfwNorm.setPrecision(precision);
        combinedFfw = network.add(postFfwNorm, combinedFfw);
        x = network.add("blk." + l + ".ffn_res", newCudaSum(precision), x, combinedFfw);
      } else {
        com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer ffnNorm =
            new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
        ffnNorm.setWeights(weights.rmsFfnWeight[l]);
        ffnNorm.setPrecision(precision);
        ffnNorm.setName("blk." + l + ".ffn_norm");
        DAGNode xbFfn = network.add(ffnNorm, x.addRef());

        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer w1Layer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                dim, hiddenDim, weights.w1[l].toMindsEyeTensor(dim, hiddenDim));
        w1Layer.setPrecision(precision);
        w1Layer.setName("blk." + l + ".ffn_gate");
        DAGNode gate = network.add(w1Layer, xbFfn.addRef());
        gate = network.add(newCudaGelu(precision), gate);

        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer w3Layer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                dim, hiddenDim, weights.w3[l].toMindsEyeTensor(dim, hiddenDim));
        w3Layer.setPrecision(precision);
        w3Layer.setName("blk." + l + ".ffn_up");
        DAGNode up = network.add(w3Layer, xbFfn);

        DAGNode gated = network.add(newCudaProd(precision), gate, up);
        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer w2Layer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                hiddenDim, dim, weights.w2[l].toMindsEyeTensor(hiddenDim, dim));
        w2Layer.setPrecision(precision);
        w2Layer.setName("blk." + l + ".ffn_down");
        DAGNode ffnOut = network.add(w2Layer, gated);

        com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer postFfwNorm =
            new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
        postFfwNorm.setWeights(weights.postFfwNorm[l]);
        postFfwNorm.setPrecision(precision);
        postFfwNorm.setName("blk." + l + ".post_ffw_norm");
        ffnOut = network.add(postFfwNorm, ffnOut);

        x = network.add("blk." + l + ".ffn_res", newCudaSum(precision), x, ffnOut);
      }

      // 7. Optional Per-Layer Embedding GELU-gated projection
      if (perLayerInputsNode != null && plDim > 0 && weights.perLayerInpGate != null && weights.perLayerInpGate[l] != null) {
        com.simiacryptus.mindseye.layers.cudnn.ChunkSelectLayer sliceLayer =
            new com.simiacryptus.mindseye.layers.cudnn.ChunkSelectLayer(plDim, nLayers, l);
        sliceLayer.setPrecision(precision);
        DAGNode plSlice = network.add(sliceLayer, perLayerInputsNode.addRef());

        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer inpGateLayer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                dim, plDim, weights.perLayerInpGate[l].toMindsEyeTensor(dim, plDim));
        inpGateLayer.setPrecision(precision);
        inpGateLayer.setName("blk." + l + ".inp_gate");
        DAGNode plGate = network.add(inpGateLayer, x.addRef());
        plGate = network.add(newCudaGelu(precision), plGate);

        DAGNode plMod = network.add(newCudaProd(precision), plGate, plSlice);
        com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer projLayer =
            new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(
                plDim, dim, weights.perLayerProj[l].toMindsEyeTensor(plDim, dim));
        projLayer.setPrecision(precision);
        projLayer.setName("blk." + l + ".proj");
        DAGNode plProj = network.add(projLayer, plMod);

        com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer postPlNorm =
            new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
        postPlNorm.setWeights(weights.perLayerPostNorm[l]);
        postPlNorm.setPrecision(precision);
        postPlNorm.setName("blk." + l + ".post_norm");
        plProj = network.add(postPlNorm, plProj);

        x = network.add("blk." + l + ".ple_res", newCudaSum(precision), x, plProj);
      }

      // 8. Layer output scale
      float scale = weights.layerOutputScale[l];
      if (scale != 1.0f) {
        com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer scaleLayer =
            new com.simiacryptus.mindseye.layers.cudnn.ScalarScaleLayer(scale, false);
        scaleLayer.setPrecision(precision);
        scaleLayer.setName("blk." + l + ".layer_output_scale");
        x = network.add(scaleLayer, x);
      }
    }

    RefUtil.freeRef(savedK);
    RefUtil.freeRef(savedV);
    if (perLayerInputsNode != null) {
      perLayerInputsNode.freeRef();
    }

    // 9. Final RMSNorm + LM Head + Logit Softcapping
    com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer finalNorm =
        new com.simiacryptus.mindseye.layers.cudnn.RmsNormLayer(dim, eps, true);
    finalNorm.setWeights(weights.rmsFinalWeight);
    finalNorm.setPrecision(precision);
    finalNorm.setName("output_norm");
    DAGNode normedOut = network.add("output_norm", finalNorm, x);

    com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer lmHead =
        new com.simiacryptus.mindseye.layers.cudnn.LinearProjectionLayer(dim, config.vocabularySize, wclsTensor);
    lmHead.setPrecision(precision);
    lmHead.setName("output");
    DAGNode logits = network.add("output", lmHead, normedOut);

    if (config.logitSoftcapping > 0f) {
      com.simiacryptus.mindseye.layers.cudnn.SoftcapLayer softcap =
          new com.simiacryptus.mindseye.layers.cudnn.SoftcapLayer(config.logitSoftcapping);
      softcap.setPrecision(precision);
      softcap.setName("final_logit_softcapping");
      logits = network.add("final_logit_softcapping", softcap, logits);
    }
    return logits;
  }

  private static Tensor to1DTensor(@Nonnull final float[] values) {
    double[] d = new double[values.length];
    for (int i = 0; i < values.length; i++) {
      d[i] = values[i];
    }
    return new Tensor(d, values.length);
  }
}
