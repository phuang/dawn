enable chromium_experimental_read_write_storage_texture;

@group(1) @binding(0) var arg_0 : texture_storage_2d_array<r32sint, read_write>;

fn textureStore_ceb832() {
  var arg_1 = vec2<u32>(1u);
  var arg_2 = 1u;
  var arg_3 = vec4<i32>(1i);
  textureStore(arg_0, arg_1, arg_2, arg_3);
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  textureStore_ceb832();
  return vec4<f32>();
}

@fragment
fn fragment_main() {
  textureStore_ceb832();
}

@compute @workgroup_size(1)
fn compute_main() {
  textureStore_ceb832();
}