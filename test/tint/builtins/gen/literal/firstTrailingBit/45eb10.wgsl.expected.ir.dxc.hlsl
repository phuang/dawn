//
// fragment_main
//

RWByteAddressBuffer prevent_dce : register(u0);
uint2 firstTrailingBit_45eb10() {
  uint2 res = (0u).xx;
  return res;
}

void fragment_main() {
  prevent_dce.Store2(0u, firstTrailingBit_45eb10());
}

//
// compute_main
//

RWByteAddressBuffer prevent_dce : register(u0);
uint2 firstTrailingBit_45eb10() {
  uint2 res = (0u).xx;
  return res;
}

[numthreads(1, 1, 1)]
void compute_main() {
  prevent_dce.Store2(0u, firstTrailingBit_45eb10());
}

//
// vertex_main
//
struct VertexOutput {
  float4 pos;
  uint2 prevent_dce;
};

struct vertex_main_outputs {
  nointerpolation uint2 VertexOutput_prevent_dce : TEXCOORD0;
  float4 VertexOutput_pos : SV_Position;
};


uint2 firstTrailingBit_45eb10() {
  uint2 res = (0u).xx;
  return res;
}

VertexOutput vertex_main_inner() {
  VertexOutput v = (VertexOutput)0;
  v.pos = (0.0f).xxxx;
  v.prevent_dce = firstTrailingBit_45eb10();
  VertexOutput v_1 = v;
  return v_1;
}

vertex_main_outputs vertex_main() {
  VertexOutput v_2 = vertex_main_inner();
  vertex_main_outputs v_3 = {v_2.prevent_dce, v_2.pos};
  return v_3;
}

