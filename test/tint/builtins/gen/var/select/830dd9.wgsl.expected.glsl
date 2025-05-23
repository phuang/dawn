//
// fragment_main
//
#version 310 es
#extension GL_AMD_gpu_shader_half_float: require
precision highp float;
precision highp int;

layout(binding = 0, std430)
buffer f_prevent_dce_block_ssbo {
  f16vec4 inner;
} v;
f16vec4 select_830dd9() {
  f16vec4 arg_0 = f16vec4(1.0hf);
  f16vec4 arg_1 = f16vec4(1.0hf);
  bool arg_2 = true;
  f16vec4 v_1 = arg_0;
  f16vec4 v_2 = arg_1;
  f16vec4 res = mix(v_1, v_2, bvec4(arg_2));
  return res;
}
void main() {
  v.inner = select_830dd9();
}
//
// compute_main
//
#version 310 es
#extension GL_AMD_gpu_shader_half_float: require

layout(binding = 0, std430)
buffer prevent_dce_block_1_ssbo {
  f16vec4 inner;
} v;
f16vec4 select_830dd9() {
  f16vec4 arg_0 = f16vec4(1.0hf);
  f16vec4 arg_1 = f16vec4(1.0hf);
  bool arg_2 = true;
  f16vec4 v_1 = arg_0;
  f16vec4 v_2 = arg_1;
  f16vec4 res = mix(v_1, v_2, bvec4(arg_2));
  return res;
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  v.inner = select_830dd9();
}
//
// vertex_main
//
#version 310 es
#extension GL_AMD_gpu_shader_half_float: require


struct VertexOutput {
  vec4 pos;
  f16vec4 prevent_dce;
};

layout(location = 0) flat out f16vec4 tint_interstage_location0;
f16vec4 select_830dd9() {
  f16vec4 arg_0 = f16vec4(1.0hf);
  f16vec4 arg_1 = f16vec4(1.0hf);
  bool arg_2 = true;
  f16vec4 v = arg_0;
  f16vec4 v_1 = arg_1;
  f16vec4 res = mix(v, v_1, bvec4(arg_2));
  return res;
}
VertexOutput vertex_main_inner() {
  VertexOutput v_2 = VertexOutput(vec4(0.0f), f16vec4(0.0hf));
  v_2.pos = vec4(0.0f);
  v_2.prevent_dce = select_830dd9();
  return v_2;
}
void main() {
  VertexOutput v_3 = vertex_main_inner();
  gl_Position = vec4(v_3.pos.x, -(v_3.pos.y), ((2.0f * v_3.pos.z) - v_3.pos.w), v_3.pos.w);
  tint_interstage_location0 = v_3.prevent_dce;
  gl_PointSize = 1.0f;
}
