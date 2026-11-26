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

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * Tanh logit soft-capping activation layer: {@code cap * tanh(x / cap)} when {@code cap > 0}, or identity otherwise.
 */
@SuppressWarnings("serial")
public final class SoftcapLayer extends SimpleActivationLayer<SoftcapLayer> {

  private final double cap;

  public SoftcapLayer(final double cap) {
    super();
    this.cap = cap;
  }

  protected SoftcapLayer(@Nonnull final JsonObject json) {
    super(json);
    this.cap = json.get("cap").getAsDouble();
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static SoftcapLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new SoftcapLayer(json);
  }

  public double getCap() {
    return cap;
  }

  @Override
  protected void eval(final double x, final double[] results) {
    if (cap > 0.0) {
      double t = Math.tanh(x / cap);
      results[0] = cap * t;
      results[1] = 1.0 - t * t;
    } else {
      results[0] = x;
      results[1] = 1.0;
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, final DataSerializer dataSerializer) {
    JsonObject json = super.getJsonStub();
    json.addProperty("cap", cap);
    return json;
  }

  @Override
  public void _free() {
    super._free();
  }

  @Nonnull
  @Override
  public SoftcapLayer addRef() {
    return (SoftcapLayer) super.addRef();
  }
}
