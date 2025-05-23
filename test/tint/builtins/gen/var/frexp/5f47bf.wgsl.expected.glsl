//
// fragment_main
//
#version 310 es
#extension GL_AMD_gpu_shader_half_float: require
precision highp float;
precision highp int;


struct frexp_result_vec2_f16 {
  f16vec2 member_0;
  ivec2 member_1;
};

void frexp_5f47bf() {
  f16vec2 arg_0 = f16vec2(1.0hf);
  frexp_result_vec2_f16 v = frexp_result_vec2_f16(f16vec2(0.0hf), ivec2(0));
  v.member_0 = frexp(arg_0, v.member_1);
  frexp_result_vec2_f16 res = v;
}
void main() {
  frexp_5f47bf();
}
//
// compute_main
//
#version 310 es
#extension GL_AMD_gpu_shader_half_float: require


struct frexp_result_vec2_f16 {
  f16vec2 member_0;
  ivec2 member_1;
};

void frexp_5f47bf() {
  f16vec2 arg_0 = f16vec2(1.0hf);
  frexp_result_vec2_f16 v = frexp_result_vec2_f16(f16vec2(0.0hf), ivec2(0));
  v.member_0 = frexp(arg_0, v.member_1);
  frexp_result_vec2_f16 res = v;
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  frexp_5f47bf();
}
//
// vertex_main
//
#version 310 es
#extension GL_AMD_gpu_shader_half_float: require


struct frexp_result_vec2_f16 {
  f16vec2 member_0;
  ivec2 member_1;
};

struct VertexOutput {
  vec4 pos;
};

void frexp_5f47bf() {
  f16vec2 arg_0 = f16vec2(1.0hf);
  frexp_result_vec2_f16 v = frexp_result_vec2_f16(f16vec2(0.0hf), ivec2(0));
  v.member_0 = frexp(arg_0, v.member_1);
  frexp_result_vec2_f16 res = v;
}
VertexOutput vertex_main_inner() {
  VertexOutput v_1 = VertexOutput(vec4(0.0f));
  v_1.pos = vec4(0.0f);
  frexp_5f47bf();
  return v_1;
}
void main() {
  vec4 v_2 = vertex_main_inner().pos;
  gl_Position = vec4(v_2.x, -(v_2.y), ((2.0f * v_2.z) - v_2.w), v_2.w);
  gl_PointSize = 1.0f;
}
