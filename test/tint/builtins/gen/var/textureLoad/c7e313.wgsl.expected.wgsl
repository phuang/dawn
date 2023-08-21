enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_2d<rgba32uint, read_write>;

fn textureLoad_c7e313() {
  var arg_1 = vec2<u32>(1u);
  var res : vec4<u32> = textureLoad(arg_0, arg_1);
  prevent_dce = res;
}

@group(2) @binding(0) var<storage, read_write> prevent_dce : vec4<u32>;

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureLoad_c7e313();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureLoad_c7e313();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureLoad_c7e313();
}