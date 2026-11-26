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
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * SentencePiece-compatible BPE tokenizer for Gemma 4 models loaded from GGUF tokenizer metadata.
 */
@RefIgnore
public class GemmaTokenizer {

  @RefIgnore
  public static final class Vocabulary {
    private final String[] tokens;
    private final float[] scores;
    private final Map<String, Integer> tokenToIndex;

    public Vocabulary(@Nonnull final String[] vocabulary, @Nonnull final float[] scores) {
      this.tokens = vocabulary.clone();
      this.scores = scores.clone();
      Map<String, Integer> map = new HashMap<>(vocabulary.length * 2);
      for (int i = 0; i < vocabulary.length; i++) {
        map.putIfAbsent(vocabulary[i], i);
      }
      this.tokenToIndex = Collections.unmodifiableMap(map);
    }

    @Nonnull
    public String get(final int tokenIndex) {
      return tokens[tokenIndex];
    }

    @Nonnull
    public OptionalInt getIndex(@Nonnull final String token) {
      Integer value = tokenToIndex.get(token);
      return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public int size() {
      return tokens.length;
    }

    public float getScore(final int tokenIndex) {
      return scores[tokenIndex];
    }
  }

  private final Vocabulary vocabulary;
  private final Map<String, Integer> specialTokens;
  private final int[] tokenType;
  private final int byte0;

  public GemmaTokenizer(@Nonnull final Vocabulary vocabulary, @Nonnull final int[] tokenType) {
    this.vocabulary = vocabulary;
    this.tokenType = tokenType.clone();
    OptionalInt eotOpt = vocabulary.getIndex("<turn|>");
    if (eotOpt.isPresent()) {
      int endOfTurn = eotOpt.getAsInt();
      for (int i = 0; i <= endOfTurn && i < this.tokenType.length; ++i) {
        if (this.tokenType[i] == 1) {
          this.tokenType[i] = 6;
        }
      }
    }
    this.byte0 = vocabulary.getIndex("<0x00>").orElse(0);
    Map<String, Integer> specials = new HashMap<>();
    for (int i = 0; i < this.tokenType.length; i++) {
      if (this.tokenType[i] != 1) {
        specials.put(vocabulary.get(i), i);
      }
    }
    this.specialTokens = Collections.unmodifiableMap(specials);
  }

  @Nonnull
  public static GemmaTokenizer fromGGUF(@Nonnull final Map<String, Object> metadata) {
    String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
    float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
    int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
    if (tokens == null || scores == null || tokenTypes == null) {
      throw new IllegalArgumentException("Missing tokenizer.ggml.{tokens,scores,token_type} metadata in GGUF");
    }
    return new GemmaTokenizer(new Vocabulary(tokens, scores), tokenTypes);
  }

  @Nonnull
  public Vocabulary getVocabulary() {
    return vocabulary;
  }

  @Nonnull
  public Map<String, Integer> getSpecialTokens() {
    return specialTokens;
  }

  public boolean isSpecialToken(final int tokenIndex) {
    return tokenType[tokenIndex] != 1;
  }

  @Nonnull
  public List<Integer> encode(@Nonnull final String text) {
    return encodeImpl(text.replace(' ', '\u2581'));
  }

  @Nonnull
  public Tensor encodeToTensor(@Nonnull final String text) {
    List<Integer> ids = encode(text);
    double[] data = new double[ids.size()];
    for (int i = 0; i < ids.size(); i++) {
      data[i] = ids.get(i);
    }
    return new Tensor(data, ids.size());
  }

  private List<Integer> encodeImpl(@Nonnull final String text) {
    List<Integer> tokens = new ArrayList<>();
    for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
      cpi = text.codePointAt(i);
      String singleCodepoint = new String(Character.toChars(cpi));
      int id = vocabulary.getIndex(singleCodepoint).orElse(-1);
      if (id != -1 && !isSpecialToken(id)) {
        tokens.add(id);
      } else {
        for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
          tokens.add(Byte.toUnsignedInt(b) + byte0);
        }
      }
    }

    while (tokens.size() >= 2) {
      float bestScore = -1e10f;
      int bestId = -1;
      int bestIdx = -1;

      for (int i = 0; i < tokens.size() - 1; ++i) {
        String merged = vocabulary.get(tokens.get(i)) + vocabulary.get(tokens.get(i + 1));
        int id = vocabulary.getIndex(merged).orElse(-1);
        if (id != -1 && vocabulary.getScore(id) > bestScore) {
          bestScore = vocabulary.getScore(id);
          bestId = id;
          bestIdx = i;
        }
      }

      if (bestIdx == -1) {
        break;
      }
      tokens.set(bestIdx, bestId);
      tokens.remove(bestIdx + 1);
    }
    return tokens;
  }

  @Nonnull
  public String decode(@Nonnull final List<Integer> tokens) {
    StringBuilder sb = new StringBuilder();
    for (int token : tokens) {
      String tokenString = vocabulary.get(token);
      if (isSpecialToken(token)) {
        String prefix = "<0x";
        String suffix = ">";
        if (tokenString.length() == 6 && tokenString.startsWith(prefix) && tokenString.endsWith(suffix)) {
          String code = tokenString.substring(prefix.length(), tokenString.length() - suffix.length());
          int cp = Integer.parseInt(code, 16);
          tokenString = new String(Character.toChars(cp));
        }
      } else {
        tokenString = tokenString.replace('\u2581', ' ');
      }
      sb.append(tokenString);
    }
    return sb.toString();
  }

  @Nonnull
  public static String replaceControlCharacters(@Nonnull final int[] codePoints) {
    StringBuilder chars = new StringBuilder();
    for (int cp : codePoints) {
      if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
        chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
      } else {
        chars.appendCodePoint(cp);
      }
    }
    return chars.toString();
  }

  @Nonnull
  public static String replaceControlCharacters(@Nonnull final String str) {
    return replaceControlCharacters(str.codePoints().toArray());
  }

  @Nonnull
  public static List<Integer> toTokenList(@Nonnull final int[] ids) {
    return IntStream.of(ids).boxed().collect(Collectors.toList());
  }
}
