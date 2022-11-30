/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const VERTEX_SHADER = `
struct VertexOutput {
  @builtin(position) Position : vec4<f32>,
  @location(0) fragUV : vec2<f32>,
}

@vertex
fn main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
  var pos = array<vec2<f32>, 6>(
    vec2<f32>( 1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(-1.0,  1.0)
  );

  var uv = array<vec2<f32>, 6>(
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 0.0)
  );

  var output : VertexOutput;
  output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = uv[VertexIndex];
  return output;
}
`;

const PIXEL_SHADER = `
struct SizeParams {
  width : i32,
  height : i32,
}

@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var myTexture: texture_external;

@group(0) @binding(3) var<storage, read> buf : array<vec4<f32>>;
@group(0) @binding(4) var<uniform> size : SizeParams;

@fragment
fn main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
  // The user-facing camera is mirrored, flip horizontally.
  let coord = vec2(1.0 - fragUV.x, fragUV.y);
  let src_color = textureSampleBaseClampToEdge(myTexture, mySampler, coord);

  let rowCol = vec2<i32>(i32(coord.y * f32(size.height)), i32(coord.x * f32(size.width)));
  let probability = (buf[rowCol.x * size.width + rowCol.y]).a;

  var out_color : vec4<f32>;
  if (probability > 0.5) {
    out_color = vec4<f32>(src_color.rgb, 1.0);
  } else {
    let purple = vec4<f32>(0.365, 0.247, 0.827, 1.0);
    out_color = 0.5 * purple + 0.5 * src_color;
  }

  return out_color;
}
`
