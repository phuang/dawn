//
// fragment_main
//
#version 310 es
precision highp float;
precision highp int;

layout(binding = 0, std430)
buffer f_prevent_dce_block_ssbo {
  vec4 inner;
} v;
uniform highp sampler2DArray f_arg_1_arg_2;
vec4 textureGather_ea8eb4() {
  vec2 arg_3 = vec2(1.0f);
  uint arg_4 = 1u;
  vec2 v_1 = arg_3;
  vec3 v_2 = vec3(v_1, float(arg_4));
  vec4 res = textureGather(f_arg_1_arg_2, v_2, int(1u));
  return res;
}
void main() {
  v.inner = textureGather_ea8eb4();
}
//
// compute_main
//
#version 310 es

layout(binding = 0, std430)
buffer prevent_dce_block_1_ssbo {
  vec4 inner;
} v;
uniform highp sampler2DArray arg_1_arg_2;
vec4 textureGather_ea8eb4() {
  vec2 arg_3 = vec2(1.0f);
  uint arg_4 = 1u;
  vec2 v_1 = arg_3;
  vec3 v_2 = vec3(v_1, float(arg_4));
  vec4 res = textureGather(arg_1_arg_2, v_2, int(1u));
  return res;
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  v.inner = textureGather_ea8eb4();
}
//
// vertex_main
//
#version 310 es


struct VertexOutput {
  vec4 pos;
  vec4 prevent_dce;
};

uniform highp sampler2DArray v_arg_1_arg_2;
layout(location = 0) flat out vec4 tint_interstage_location0;
vec4 textureGather_ea8eb4() {
  vec2 arg_3 = vec2(1.0f);
  uint arg_4 = 1u;
  vec2 v = arg_3;
  vec3 v_1 = vec3(v, float(arg_4));
  vec4 res = textureGather(v_arg_1_arg_2, v_1, int(1u));
  return res;
}
VertexOutput vertex_main_inner() {
  VertexOutput v_2 = VertexOutput(vec4(0.0f), vec4(0.0f));
  v_2.pos = vec4(0.0f);
  v_2.prevent_dce = textureGather_ea8eb4();
  return v_2;
}
void main() {
  VertexOutput v_3 = vertex_main_inner();
  gl_Position = vec4(v_3.pos.x, -(v_3.pos.y), ((2.0f * v_3.pos.z) - v_3.pos.w), v_3.pos.w);
  tint_interstage_location0 = v_3.prevent_dce;
  gl_PointSize = 1.0f;
}
