#version 310 es
precision highp float;
precision highp int;

layout(binding = 0, std430)
buffer f_prevent_dce_block_ssbo {
  vec4 inner;
} v;
uniform highp sampler2DArray f_arg_0_arg_1;
vec4 textureSample_17e988() {
  vec4 res = textureOffset(f_arg_0_arg_1, vec3(vec2(1.0f), float(1)), ivec2(1));
  return res;
}
void main() {
  v.inner = textureSample_17e988();
}
