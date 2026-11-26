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
import com.simiacryptus.ref.lang.RefIgnore;

import javax.annotation.Nonnull;
import java.util.Comparator;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Token sampling strategies (greedy argmax, temperature categorical, and nucleus top-p) for Gemma 4 generation.
 */
@RefIgnore
@FunctionalInterface
public interface GemmaSampler {

  GemmaSampler ARGMAX = GGUFTensor::argmax;

  int sampleToken(@Nonnull GGUFTensor logits);

  default int sampleFromLastPosition(@Nonnull final Tensor sequenceLogits, final int vocabSize) {
    try {
      double[] data = sequenceLogits.getData();
      if (data.length < vocabSize || data.length % vocabSize != 0) {
        throw new IllegalArgumentException("Logits tensor length " + data.length + " invalid for vocabSize " + vocabSize);
      }
      int offset = data.length - vocabSize;
      float[] lastLogits = new float[vocabSize];
      for (int i = 0; i < vocabSize; i++) {
        lastLogits[i] = (float) data[offset + i];
      }
      return sampleToken(new GGUFTensor.ArrayFloatTensor(lastLogits));
    } finally {
      sequenceLogits.freeRef();
    }
  }

  @Nonnull
  static GemmaSampler selectSampler(final int vocabularySize, final float temperature,
                                    final float topp, final long rngSeed) {
    if (temperature == 0.0f) {
      return ARGMAX;
    }
    RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
    final GemmaSampler innerSampler;
    if (topp <= 0f || topp >= 1f) {
      innerSampler = new CategoricalSampler(rng);
    } else {
      innerSampler = new ToppSampler(vocabularySize, topp, rng);
    }
    return logits -> {
      int logitsSize = Math.toIntExact(logits.size());
      logits.divideInPlace(0, logitsSize, temperature);
      logits.softmaxInPlace(0, logitsSize);
      return innerSampler.sampleToken(logits);
    };
  }

  @RefIgnore
  final class CategoricalSampler implements GemmaSampler {
    private final RandomGenerator rng;

    public CategoricalSampler(@Nonnull final RandomGenerator rng) {
      this.rng = rng;
    }

    @Override
    public int sampleToken(@Nonnull final GGUFTensor logits) {
      float random0to1 = rng.nextFloat(1f);
      float cdf = 0.0f;
      int n = Math.toIntExact(logits.size());
      for (int i = 0; i < n; i++) {
        cdf += logits.getFloat(i);
        if (random0to1 < cdf) {
          return i;
        }
      }
      return n - 1;
    }
  }

  @RefIgnore
  final class ToppSampler implements GemmaSampler {
    private final int[] indices;
    private final float topp;
    private final RandomGenerator rng;

    public ToppSampler(final int maxNumberOfElements, final float topp, @Nonnull final RandomGenerator rng) {
      this.indices = new int[maxNumberOfElements];
      this.topp = topp;
      this.rng = rng;
    }

    private static void swap(final int[] array, final int from, final int to) {
      int tmp = array[from];
      array[from] = array[to];
      array[to] = tmp;
    }

    private static void siftDown(final int[] array, final int from, final int n,
                                 final Comparator<Integer> comparator) {
      int prev = from;
      int next;
      while ((next = 2 * prev + 1) < n) {
        int r = 2 * prev + 2;
        if (r < n && comparator.compare(array[r], array[next]) < 0) {
          next = r;
        }
        if (comparator.compare(array[next], array[prev]) < 0) {
          swap(array, prev, next);
          prev = next;
        } else {
          break;
        }
      }
    }

    @Override
    public int sampleToken(@Nonnull final GGUFTensor logits) {
      Comparator<Integer> comparator = Comparator.comparingDouble((Integer i) -> (double) logits.getFloat(i)).reversed();
      int n = Math.toIntExact(logits.size());
      int head = 0;
      int tail = n - 1;
      float cutoff = (1.0f - topp) / Math.max(1, n - 1);
      for (int i = 0; i < n; i++) {
        if (logits.getFloat(i) >= cutoff) {
          indices[head++] = i;
        } else {
          indices[tail--] = i;
        }
      }

      int n0 = head;
      for (int i = n0 / 2 - 1; i >= 0; --i) {
        siftDown(indices, i, n0, comparator);
      }

      float cumulativeProb = 0.0f;
      int lastIndex = 0;
      for (int i = n0 - 1; i >= 0; i--) {
        swap(indices, 0, i);
        cumulativeProb += logits.getFloat(indices[i]);
        if (cumulativeProb > topp) {
          lastIndex = i;
          break;
        }
        siftDown(indices, 0, i, comparator);
      }

      float r = rng.nextFloat(1f) * cumulativeProb;
      float cdf = 0.0f;
      for (int i = n0 - 1; i >= lastIndex; i--) {
        cdf += logits.getFloat(indices[i]);
        if (r < cdf) {
          return indices[i];
        }
      }
      return indices[lastIndex];
    }
  }
}
