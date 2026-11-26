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
 * Tanh-approximated Gaussian Error Linear Unit (GELU) activation layer used in Gemma 4:
 * {@code 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))}.
 */
@SuppressWarnings("serial")
public final class GeluActivationLayer extends SimpleActivationLayer<GeluActivationLayer> {

  private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);
  private static final double COEFF = 0.044715;

  public GeluActivationLayer() {
    super();
  }

  protected GeluActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static GeluActivationLayer fromJson(@Nonnull final JsonObject json, final Map<CharSequence, byte[]> rs) {
    return new GeluActivationLayer(json);
  }

  public static double gelu(final double x) {
    double x3 = x * x * x;
    double u = SQRT_2_OVER_PI * (x + COEFF * x3);
    return 0.5 * x * (1.0 + Math.tanh(u));
  }

  public static double geluGrad(final double x) {
    double x2 = x * x;
    double u = SQRT_2_OVER_PI * (x + COEFF * x * x2);
    double t = Math.tanh(u);
    double du = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x2);
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;
  }

  @Override
  protected void eval(final double x, final double[] results) {
    double x2 = x * x;
    double u = SQRT_2_OVER_PI * (x + COEFF * x * x2);
    double t = Math.tanh(u);
    double du = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x2);
    results[0] = 0.5 * x * (1.0 + t);
    results[1] = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;
  }

  @Nonnull
  @Override
  public JsonObject getJson(final Map<CharSequence, byte[]> resources, final DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Override
  public void _free() {
    super._free();
  }

  @Nonnull
  @Override
  public GeluActivationLayer addRef() {
    return (GeluActivationLayer) super.addRef();
  }
}
