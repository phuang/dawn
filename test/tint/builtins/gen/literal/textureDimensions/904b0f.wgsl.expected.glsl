SKIP: FAILED

#version 310 es

layout(rgba16i) uniform highp iimage2DArray arg_0;
layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  uvec2 inner;
} prevent_dce;

void textureDimensions_904b0f() {
  uvec2 res = uvec2(imageSize(arg_0).xy);
  prevent_dce.inner = res;
}

vec4 vertex_main() {
  textureDimensions_904b0f();
  return vec4(0.0f);
}

void main() {
  gl_PointSize = 1.0;
  vec4 inner_result = vertex_main();
  gl_Position = inner_result;
  gl_Position.y = -(gl_Position.y);
  gl_Position.z = ((2.0f * gl_Position.z) - gl_Position.w);
  return;
}
Error parsing GLSL shader:
ERROR: 0:3: 'rgba16i' : format requires readonly or writeonly memory qualifier 
ERROR: 1 compilation errors.  No code generated.



#version 310 es
precision highp float;

layout(rgba16i) uniform highp iimage2DArray arg_0;
layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  uvec2 inner;
} prevent_dce;

void textureDimensions_904b0f() {
  uvec2 res = uvec2(imageSize(arg_0).xy);
  prevent_dce.inner = res;
}

void fragment_main() {
  textureDimensions_904b0f();
}

void main() {
  fragment_main();
  return;
}
Error parsing GLSL shader:
ERROR: 0:4: 'rgba16i' : format requires readonly or writeonly memory qualifier 
ERROR: 1 compilation errors.  No code generated.



#version 310 es

layout(rgba16i) uniform highp iimage2DArray arg_0;
layout(binding = 0, std430) buffer prevent_dce_block_ssbo {
  uvec2 inner;
} prevent_dce;

void textureDimensions_904b0f() {
  uvec2 res = uvec2(imageSize(arg_0).xy);
  prevent_dce.inner = res;
}

void compute_main() {
  textureDimensions_904b0f();
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  compute_main();
  return;
}
Error parsing GLSL shader:
ERROR: 0:3: 'rgba16i' : format requires readonly or writeonly memory qualifier 
ERROR: 1 compilation errors.  No code generated.


