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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.lang.RefIgnore;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

/**
 * Differentiable Grouped-Query Attention (GQA) layer taking 3 inputs {@code (Q, K, V)}:
 * <ul>
 *   <li>{@code Q}: {@code [numHeads * headSize, qSeqLen]} (or 1D {@code [numHeads * headSize]})</li>
 *   <li>{@code K}: {@code [numKvHeads * headSize, kvSeqLen]}</li>
 *   <li>{@code V}: {@code [numKvHeads * headSize, kvSeqLen]}</li>
 * </ul>
 * Supports causal masking, Gemma 4 sliding-window attention (SWA), shared KV inputs, V=K graphs,
 * and configurable attention score scale ({@code 1.0} in Gemma 4 with Q/K RMSNorm).
 */
@SuppressWarnings("serial")
public class GroupedQueryAttentionLayer extends LayerBase {

  private final int numHeads;
  private final int numKvHeads;
  private final int headSize;
  private final double scale;
  private final boolean causal;
  private final int slidingWindow;

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
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static GroupedQueryAttentionLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new GroupedQueryAttentionLayer(json);
  }

  @RefIgnore
  static void forwardAttention(@Nonnull final double[] q, @Nonnull final double[] k, @Nonnull final double[] v,
                               @Nonnull final double[] out, final int numHeads, final int numKvHeads,
                               final int headSize, final double scale, final boolean causal, final int slidingWindow) {
    int qDim = numHeads * headSize;
    int kvDim = numKvHeads * headSize;
    if (q.length % qDim != 0 || k.length % kvDim != 0 || v.length != k.length) {
      throw new IllegalArgumentException("Incompatible Q/K/V lengths: " + q.length + ", " + k.length + ", " + v.length);
    }
    int qSeqLen = q.length / qDim;
    int kvSeqLen = k.length / kvDim;
    int kvMul = numHeads / numKvHeads;
    int posOffset = kvSeqLen - qSeqLen;
    double[] scores = new double[kvSeqLen];

    for (int t = 0; t < qSeqLen; t++) {
      int globalPos = causal ? Math.min(kvSeqLen - 1, Math.max(0, posOffset + t)) : (kvSeqLen - 1);
      int endS = causal ? globalPos : (kvSeqLen - 1);
      int startS = slidingWindow > 0 ? Math.max(0, endS - slidingWindow + 1) : 0;

      for (int h = 0; h < numHeads; h++) {
        int kvHead = h / kvMul;
        int qBase = t * qDim + h * headSize;
        double maxScore = Double.NEGATIVE_INFINITY;
        for (int s = startS; s <= endS; s++) {
          int kBase = s * kvDim + kvHead * headSize;
          double dot = 0.0;
          for (int d = 0; d < headSize; d++) {
            dot += q[qBase + d] * k[kBase + d];
          }
          double sc = dot * scale;
          scores[s] = sc;
          if (sc > maxScore) {
            maxScore = sc;
          }
        }
        double sumExp = 0.0;
        for (int s = startS; s <= endS; s++) {
          double e = Math.exp(scores[s] - maxScore);
          scores[s] = e;
          sumExp += e;
        }
        for (int s = startS; s <= endS; s++) {
          scores[s] /= sumExp;
        }
        int outBase = t * qDim + h * headSize;
        for (int d = 0; d < headSize; d++) {
          double acc = 0.0;
          for (int s = startS; s <= endS; s++) {
            acc += scores[s] * v[s * kvDim + kvHead * headSize + d];
          }
          out[outBase + d] = acc;
        }
      }
    }
  }

  @RefIgnore
  static void backwardAttention(@Nonnull final double[] q, @Nonnull final double[] k, @Nonnull final double[] v,
                                @Nonnull final double[] dOut, @Nonnull final double[] dQ,
                                @Nonnull final double[] dK, @Nonnull final double[] dV,
                                final int numHeads, final int numKvHeads, final int headSize,
                                final double scale, final boolean causal, final int slidingWindow) {
    int qDim = numHeads * headSize;
    int kvDim = numKvHeads * headSize;
    int qSeqLen = q.length / qDim;
    int kvSeqLen = k.length / kvDim;
    int kvMul = numHeads / numKvHeads;
    int posOffset = kvSeqLen - qSeqLen;
    double[] attn = new double[kvSeqLen];
    double[] dAttn = new double[kvSeqLen];

    for (int t = 0; t < qSeqLen; t++) {
      int globalPos = causal ? Math.min(kvSeqLen - 1, Math.max(0, posOffset + t)) : (kvSeqLen - 1);
      int endS = causal ? globalPos : (kvSeqLen - 1);
      int startS = slidingWindow > 0 ? Math.max(0, endS - slidingWindow + 1) : 0;

      for (int h = 0; h < numHeads; h++) {
        int kvHead = h / kvMul;
        int qBase = t * qDim + h * headSize;
        double maxScore = Double.NEGATIVE_INFINITY;
        for (int s = startS; s <= endS; s++) {
          int kBase = s * kvDim + kvHead * headSize;
          double dot = 0.0;
          for (int d = 0; d < headSize; d++) {
            dot += q[qBase + d] * k[kBase + d];
          }
          double sc = dot * scale;
          attn[s] = sc;
          if (sc > maxScore) {
            maxScore = sc;
          }
        }
        double sumExp = 0.0;
        for (int s = startS; s <= endS; s++) {
          double e = Math.exp(attn[s] - maxScore);
          attn[s] = e;
          sumExp += e;
        }
        for (int s = startS; s <= endS; s++) {
          attn[s] /= sumExp;
        }

        int outBase = t * qDim + h * headSize;
        double attnDot = 0.0;
        for (int s = startS; s <= endS; s++) {
          int vBase = s * kvDim + kvHead * headSize;
          double da = 0.0;
          double a = attn[s];
          for (int d = 0; d < headSize; d++) {
            double doi = dOut[outBase + d];
            da += doi * v[vBase + d];
            dV[vBase + d] += a * doi;
          }
          dAttn[s] = da;
          attnDot += a * da;
        }

        for (int s = startS; s <= endS; s++) {
          double dScore = scale * attn[s] * (dAttn[s] - attnDot);
          int kBase = s * kvDim + kvHead * headSize;
          for (int d = 0; d < headSize; d++) {
            dQ[qBase + d] += dScore * k[kBase + d];
            dK[kBase + d] += dScore * q[qBase + d];
          }
        }
      }
    }
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
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
    int batchSize = qList.length();

    TensorArray outputData = new TensorArray(RefIntStream.range(0, batchSize)
        .mapToObj(RefUtil.wrapInterface((IntFunction<Tensor>) b -> {
          Tensor qt = qList.get(b);
          Tensor kt = kList.get(kList.length() == 1 ? 0 : b);
          Tensor vt = vList.get(vList.length() == 1 ? 0 : b);
          Tensor out = new Tensor(qt.getDimensions());
          forwardAttention(qt.getData(), kt.getData(), vt.getData(), out.getData(),
              numHeads, numKvHeads, headSize, scale, causal, slidingWindow);
          qt.freeRef();
          kt.freeRef();
          vt.freeRef();
          return out;
        }, qList.addRef(), kList.addRef(), vList.addRef()))
        .toArray(Tensor[]::new));

    boolean qAlive = qRes.isAlive();
    boolean kAlive = kRes.isAlive();
    boolean vAlive = vRes.isAlive();
    boolean anyAlive = qAlive || kAlive || vAlive;

    Accumulator accumulator = new Accumulator(
        qList, kList, vList,
        qRes.getAccumulator(), kRes.getAccumulator(), vRes.getAccumulator(),
        qAlive, kAlive, vAlive,
        numHeads, numKvHeads, headSize, scale, causal, slidingWindow
    );
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

    Accumulator(final TensorList qList, final TensorList kList, final TensorList vList,
                final Result.Accumulator qAcc, final Result.Accumulator kAcc, final Result.Accumulator vAcc,
                final boolean qAlive, final boolean kAlive, final boolean vAlive,
                final int numHeads, final int numKvHeads, final int headSize,
                final double scale, final boolean causal, final int slidingWindow) {
      this.qList = qList;
      this.kList = kList;
      this.vList = vList;
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
    }

    @Override
    public void accept(@Nullable final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) {
      int batchSize = qList.length();
      int[] qDims = qList.getDimensions();
      int[] kDims = kList.getDimensions();
      int[] vDims = vList.getDimensions();
      int qLen = Tensor.length(qDims);
      int kLen = Tensor.length(kDims);

      double[][] dQBatch = new double[batchSize][qLen];
      double[][] dKBatch = new double[batchSize][kLen];
      double[][] dVBatch = new double[batchSize][kLen];

      for (int b = 0; b < batchSize; b++) {
        Tensor qt = qList.get(b);
        Tensor kt = kList.get(kList.length() == 1 ? 0 : b);
        Tensor vt = vList.get(vList.length() == 1 ? 0 : b);
        Tensor dt = delta.get(b);
        backwardAttention(
            qt.getData(), kt.getData(), vt.getData(), dt.getData(),
            dQBatch[b], dKBatch[b], dVBatch[b],
            numHeads, numKvHeads, headSize, scale, causal, slidingWindow
        );
        dt.freeRef();
        vt.freeRef();
        kt.freeRef();
        qt.freeRef();
      }

      if (qAlive) {
        TensorArray dqArr = new TensorArray(RefIntStream.range(0, batchSize)
            .mapToObj(b -> new Tensor(dQBatch[b], qDims))
            .toArray(Tensor[]::new));
        DeltaSet<UUID> bufCopy = buffer == null ? null : buffer.addRef();
        qAcc.accept(bufCopy, dqArr);
      }
      if (kAlive) {
        TensorArray dkArr = buildKvPassback(dKBatch, kDims, kList.length(), batchSize);
        DeltaSet<UUID> bufCopy = buffer == null ? null : buffer.addRef();
        kAcc.accept(bufCopy, dkArr);
      }
      if (vAlive) {
        TensorArray dvArr = buildKvPassback(dVBatch, vDims, vList.length(), batchSize);
        DeltaSet<UUID> bufCopy = buffer == null ? null : buffer.addRef();
        vAcc.accept(bufCopy, dvArr);
      }

      delta.freeRef();
      if (buffer != null) {
        buffer.freeRef();
      }
    }

    private static TensorArray buildKvPassback(final double[][] grads, final int[] dims,
                                               final int targetBatch, final int batchSize) {
      if (targetBatch == 1 && batchSize > 1) {
        double[] sum = new double[grads[0].length];
        for (int b = 0; b < batchSize; b++) {
          for (int i = 0; i < sum.length; i++) {
            sum[i] += grads[b][i];
          }
        }
        return new TensorArray(new Tensor(sum, dims));
      }
      return new TensorArray(RefIntStream.range(0, batchSize)
          .mapToObj(b -> new Tensor(grads[b], dims))
          .toArray(Tensor[]::new));
    }

    @Override
    public void _free() {
      super._free();
      qList.freeRef();
      kList.freeRef();
      vList.freeRef();
      qAcc.freeRef();
      kAcc.freeRef();
      vAcc.freeRef();
    }
  }
}
