//
// fragment_main
//
#version 310 es
precision highp float;
precision highp int;

layout(binding = 0, std430)
buffer f_prevent_dce_block_ssbo {
  float inner;
} v;
uniform highp sampler2DShadow f_arg_0_arg_1;
float textureSampleCompareLevel_2ad2b1() {
  float res = texture(f_arg_0_arg_1, vec3(vec2(1.0f), 1.0f));
  return res;
}
void main() {
  v.inner = textureSampleCompareLevel_2ad2b1();
}
//
// compute_main
//
#version 310 es

layout(binding = 0, std430)
buffer prevent_dce_block_1_ssbo {
  float inner;
} v;
uniform highp sampler2DShadow arg_0_arg_1;
float textureSampleCompareLevel_2ad2b1() {
  float res = texture(arg_0_arg_1, vec3(vec2(1.0f), 1.0f));
  return res;
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  v.inner = textureSampleCompareLevel_2ad2b1();
}
//
// vertex_main
//
#version 310 es


struct VertexOutput {
  vec4 pos;
  float prevent_dce;
};

uniform highp sampler2DShadow v_arg_0_arg_1;
layout(location = 0) flat out float tint_interstage_location0;
float textureSampleCompareLevel_2ad2b1() {
  float res = texture(v_arg_0_arg_1, vec3(vec2(1.0f), 1.0f));
  return res;
}
VertexOutput vertex_main_inner() {
  VertexOutput v = VertexOutput(vec4(0.0f), 0.0f);
  v.pos = vec4(0.0f);
  v.prevent_dce = textureSampleCompareLevel_2ad2b1();
  return v;
}
void main() {
  VertexOutput v_1 = vertex_main_inner();
  gl_Position = vec4(v_1.pos.x, -(v_1.pos.y), ((2.0f * v_1.pos.z) - v_1.pos.w), v_1.pos.w);
  tint_interstage_location0 = v_1.prevent_dce;
  gl_PointSize = 1.0f;
}
