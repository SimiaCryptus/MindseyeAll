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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Gemma 4 chat template formatter supporting system turns, thinking channels ({@code <|think|>}, {@code <|channel>}),
 * user/model turns, and fill-in-the-middle (FIM) prompts.
 */
@RefIgnore
public class GemmaChatFormat {

  @RefIgnore
  public static final class Role {
    public static final Role SYSTEM = new Role("system");
    public static final Role USER = new Role("user");
    public static final Role MODEL = new Role("model");

    private final String name;

    public Role(@Nonnull final String name) {
      this.name = name;
    }

    @Nonnull
    public String name() {
      return name;
    }

    @Nonnull
    @Override
    public String toString() {
      return name;
    }
  }

  @RefIgnore
  public static final class Message {
    private final Role role;
    private final String content;

    public Message(@Nonnull final Role role, @Nonnull final String content) {
      this.role = role;
      this.content = content;
    }

    @Nonnull
    public Role role() {
      return role;
    }

    @Nonnull
    public String content() {
      return content;
    }
  }

  protected final GemmaTokenizer tokenizer;
  protected final int beginOfSentence;
  protected final int startOfTurn;
  protected final int endOfTurn;
  protected final int endOfSentence;
  protected final int fimSuffix;
  protected final int fimPrefix;
  protected final int fimMiddle;
  protected final int fileSeparator;

  public GemmaChatFormat(@Nonnull final GemmaTokenizer tokenizer) {
    this.tokenizer = tokenizer;
    Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
    this.beginOfSentence = requireToken(specialTokens, "<bos>");
    this.startOfTurn = requireToken(specialTokens, "<|turn>");
    this.endOfTurn = requireToken(specialTokens, "<turn|>");
    this.endOfSentence = requireToken(specialTokens, "<eos>");
    this.fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
    this.fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
    this.fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
    this.fileSeparator = specialTokens.getOrDefault("<|file_separator|>", -1);
  }

  private static int requireToken(final Map<String, Integer> map, final String key) {
    Integer v = map.get(key);
    if (v == null) {
      throw new IllegalArgumentException("Missing required special token in vocabulary: " + key);
    }
    return v;
  }

  public int getBeginOfSentence() {
    return beginOfSentence;
  }

  public int getEndOfTurn() {
    return endOfTurn;
  }

  public int getEndOfSentence() {
    return endOfSentence;
  }

  @Nonnull
  public Set<Integer> getStopTokens() {
    Set<Integer> tokens = new HashSet<>();
    tokens.add(endOfSentence);
    tokens.add(endOfTurn);
    if (fimSuffix != -1) {
      tokens.add(fimSuffix);
    }
    if (fimPrefix != -1) {
      tokens.add(fimPrefix);
    }
    if (fimMiddle != -1) {
      tokens.add(fimMiddle);
    }
    if (fileSeparator != -1) {
      tokens.add(fileSeparator);
    }
    return tokens;
  }

  @Nonnull
  public List<Integer> encodeHeader(@Nonnull final Message message) {
    List<Integer> tokens = new ArrayList<>();
    tokens.add(startOfTurn);
    tokens.addAll(tokenizer.encode(message.role().toString()));
    tokens.addAll(tokenizer.encode("\n"));
    return tokens;
  }

  @Nonnull
  public List<Integer> encodeMessage(@Nonnull final Message message) {
    List<Integer> tokens = encodeHeader(message);
    tokens.addAll(tokenizer.encode(message.content().strip()));
    tokens.add(endOfTurn);
    tokens.addAll(tokenizer.encode("\n"));
    return tokens;
  }

  @Nonnull
  public List<Integer> encodeSystemThinkingTurn(@Nullable final String systemPrompt) {
    List<Integer> tokens = new ArrayList<>();
    tokens.addAll(encodeHeader(new Message(Role.SYSTEM, "")));
    Integer thinkToken = tokenizer.getSpecialTokens().get("<|think|>");
    if (thinkToken != null) {
      tokens.add(thinkToken);
      tokens.addAll(tokenizer.encode("\n"));
    }
    if (systemPrompt != null && !systemPrompt.isEmpty()) {
      tokens.addAll(tokenizer.encode(systemPrompt.trim()));
    }
    tokens.add(endOfTurn);
    tokens.addAll(tokenizer.encode("\n"));
    return tokens;
  }

  @Nonnull
  public List<Integer> encodeFillInTheMiddle(@Nonnull final String prefix, @Nonnull final String suffix) {
    if (fimPrefix == -1 || fimSuffix == -1 || fimMiddle == -1) {
      throw new IllegalStateException("Vocabulary does not define FIM special tokens");
    }
    List<Integer> tokens = new ArrayList<>();
    tokens.add(fimPrefix);
    tokens.addAll(tokenizer.encode(prefix));
    tokens.add(fimSuffix);
    tokens.addAll(tokenizer.encode(suffix));
    tokens.add(fimMiddle);
    return tokens;
  }

  @Nonnull
  public static List<Integer> stripThoughtChannelTokens(@Nonnull final GemmaTokenizer tokenizer,
                                                        @Nonnull final List<Integer> tokens) {
    Integer channelOpen = tokenizer.getSpecialTokens().get("<|channel>");
    Integer channelClose = tokenizer.getSpecialTokens().get("<channel|>");
    if (channelOpen == null || channelClose == null || tokens.isEmpty()) {
      return tokens;
    }
    List<Integer> out = new ArrayList<>(tokens.size());
    boolean inChannel = false;
    for (int tok : tokens) {
      if (tok == channelOpen) {
        inChannel = true;
        continue;
      }
      if (tok == channelClose) {
        inChannel = false;
        continue;
      }
      if (!inChannel) {
        out.add(tok);
      }
    }
    return out;
  }

  @Nonnull
  public static List<Integer> visibleTokens(@Nonnull final GemmaTokenizer tokenizer,
                                            @Nonnull final List<Integer> tokens,
                                            final boolean think) {
    return think ? stripThoughtChannelTokens(tokenizer, tokens) : tokens;
  }
}
