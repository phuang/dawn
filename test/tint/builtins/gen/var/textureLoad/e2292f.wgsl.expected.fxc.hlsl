//
// fragment_main
//
RWByteAddressBuffer prevent_dce : register(u0);
Texture1D<int4> arg_0 : register(t0, space1);

int4 textureLoad_e2292f() {
  uint arg_1 = 1u;
  int4 res = arg_0.Load(uint2(arg_1, uint(0)));
  return res;
}

void fragment_main() {
  prevent_dce.Store4(0u, asuint(textureLoad_e2292f()));
  return;
}
//
// compute_main
//
RWByteAddressBuffer prevent_dce : register(u0);
Texture1D<int4> arg_0 : register(t0, space1);

int4 textureLoad_e2292f() {
  uint arg_1 = 1u;
  int4 res = arg_0.Load(uint2(arg_1, uint(0)));
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store4(0u, asuint(textureLoad_e2292f()));
  return;
}
//
// vertex_main
//
Texture1D<int4> arg_0 : register(t0, space1);

int4 textureLoad_e2292f() {
  uint arg_1 = 1u;
  int4 res = arg_0.Load(uint2(arg_1, uint(0)));
  return res;
}

struct VertexOutput {
  float4 pos;
  int4 prevent_dce;
};
struct tint_symbol_1 {
  nointerpolation int4 prevent_dce : TEXCOORD0;
  float4 pos : SV_Position;
};

VertexOutput vertex_main_inner() {
  VertexOutput tint_symbol = (VertexOutput)0;
  tint_symbol.pos = (0.0f).xxxx;
  tint_symbol.prevent_dce = textureLoad_e2292f();
  return tint_symbol;
}

tint_symbol_1 vertex_main() {
  VertexOutput inner_result = vertex_main_inner();
  tint_symbol_1 wrapper_result = (tint_symbol_1)0;
  wrapper_result.pos = inner_result.pos;
  wrapper_result.prevent_dce = inner_result.prevent_dce;
  return wrapper_result;
}
