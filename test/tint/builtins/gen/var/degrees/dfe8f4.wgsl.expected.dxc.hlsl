//
// fragment_main
//
vector<float16_t, 3> tint_degrees(vector<float16_t, 3> param_0) {
  return param_0 * 57.29577951308232286465;
}

RWByteAddressBuffer prevent_dce : register(u0);

vector<float16_t, 3> degrees_dfe8f4() {
  vector<float16_t, 3> arg_0 = (float16_t(1.0h)).xxx;
  vector<float16_t, 3> res = tint_degrees(arg_0);
  return res;
}

void fragment_main() {
  prevent_dce.Store<vector<float16_t, 3> >(0u, degrees_dfe8f4());
  return;
}
//
// compute_main
//
vector<float16_t, 3> tint_degrees(vector<float16_t, 3> param_0) {
  return param_0 * 57.29577951308232286465;
}

RWByteAddressBuffer prevent_dce : register(u0);

vector<float16_t, 3> degrees_dfe8f4() {
  vector<float16_t, 3> arg_0 = (float16_t(1.0h)).xxx;
  vector<float16_t, 3> res = tint_degrees(arg_0);
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store<vector<float16_t, 3> >(0u, degrees_dfe8f4());
  return;
}
//
// vertex_main
//
vector<float16_t, 3> tint_degrees(vector<float16_t, 3> param_0) {
  return param_0 * 57.29577951308232286465;
}

vector<float16_t, 3> degrees_dfe8f4() {
  vector<float16_t, 3> arg_0 = (float16_t(1.0h)).xxx;
  vector<float16_t, 3> res = tint_degrees(arg_0);
  return res;
}

struct VertexOutput {
  float4 pos;
  vector<float16_t, 3> prevent_dce;
};
struct tint_symbol_1 {
  nointerpolation vector<float16_t, 3> prevent_dce : TEXCOORD0;
  float4 pos : SV_Position;
};

VertexOutput vertex_main_inner() {
  VertexOutput tint_symbol = (VertexOutput)0;
  tint_symbol.pos = (0.0f).xxxx;
  tint_symbol.prevent_dce = degrees_dfe8f4();
  return tint_symbol;
}

tint_symbol_1 vertex_main() {
  VertexOutput inner_result = vertex_main_inner();
  tint_symbol_1 wrapper_result = (tint_symbol_1)0;
  wrapper_result.pos = inner_result.pos;
  wrapper_result.prevent_dce = inner_result.prevent_dce;
  return wrapper_result;
}
