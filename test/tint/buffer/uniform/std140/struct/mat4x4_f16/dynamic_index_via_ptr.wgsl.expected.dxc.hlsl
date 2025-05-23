struct Inner {
  matrix<float16_t, 4, 4> m;
};
struct Outer {
  Inner a[4];
};

cbuffer cbuffer_a : register(b0) {
  uint4 a[64];
};
static int counter = 0;

int i() {
  counter = (counter + 1);
  return counter;
}

matrix<float16_t, 4, 4> a_load_4(uint offset) {
  const uint scalar_offset = ((offset + 0u)) / 4;
  uint4 ubo_load_1 = a[scalar_offset / 4];
  uint2 ubo_load = ((scalar_offset & 2) ? ubo_load_1.zw : ubo_load_1.xy);
  vector<float16_t, 2> ubo_load_xz = vector<float16_t, 2>(f16tof32(ubo_load & 0xFFFF));
  vector<float16_t, 2> ubo_load_yw = vector<float16_t, 2>(f16tof32(ubo_load >> 16));
  const uint scalar_offset_1 = ((offset + 8u)) / 4;
  uint4 ubo_load_3 = a[scalar_offset_1 / 4];
  uint2 ubo_load_2 = ((scalar_offset_1 & 2) ? ubo_load_3.zw : ubo_load_3.xy);
  vector<float16_t, 2> ubo_load_2_xz = vector<float16_t, 2>(f16tof32(ubo_load_2 & 0xFFFF));
  vector<float16_t, 2> ubo_load_2_yw = vector<float16_t, 2>(f16tof32(ubo_load_2 >> 16));
  const uint scalar_offset_2 = ((offset + 16u)) / 4;
  uint4 ubo_load_5 = a[scalar_offset_2 / 4];
  uint2 ubo_load_4 = ((scalar_offset_2 & 2) ? ubo_load_5.zw : ubo_load_5.xy);
  vector<float16_t, 2> ubo_load_4_xz = vector<float16_t, 2>(f16tof32(ubo_load_4 & 0xFFFF));
  vector<float16_t, 2> ubo_load_4_yw = vector<float16_t, 2>(f16tof32(ubo_load_4 >> 16));
  const uint scalar_offset_3 = ((offset + 24u)) / 4;
  uint4 ubo_load_7 = a[scalar_offset_3 / 4];
  uint2 ubo_load_6 = ((scalar_offset_3 & 2) ? ubo_load_7.zw : ubo_load_7.xy);
  vector<float16_t, 2> ubo_load_6_xz = vector<float16_t, 2>(f16tof32(ubo_load_6 & 0xFFFF));
  vector<float16_t, 2> ubo_load_6_yw = vector<float16_t, 2>(f16tof32(ubo_load_6 >> 16));
  return matrix<float16_t, 4, 4>(vector<float16_t, 4>(ubo_load_xz[0], ubo_load_yw[0], ubo_load_xz[1], ubo_load_yw[1]), vector<float16_t, 4>(ubo_load_2_xz[0], ubo_load_2_yw[0], ubo_load_2_xz[1], ubo_load_2_yw[1]), vector<float16_t, 4>(ubo_load_4_xz[0], ubo_load_4_yw[0], ubo_load_4_xz[1], ubo_load_4_yw[1]), vector<float16_t, 4>(ubo_load_6_xz[0], ubo_load_6_yw[0], ubo_load_6_xz[1], ubo_load_6_yw[1]));
}

Inner a_load_3(uint offset) {
  Inner tint_symbol_4 = {a_load_4((offset + 0u))};
  return tint_symbol_4;
}

typedef Inner a_load_2_ret[4];
a_load_2_ret a_load_2(uint offset) {
  Inner arr[4] = (Inner[4])0;
  {
    for(uint i_1 = 0u; (i_1 < 4u); i_1 = (i_1 + 1u)) {
      arr[i_1] = a_load_3((offset + (i_1 * 64u)));
    }
  }
  return arr;
}

Outer a_load_1(uint offset) {
  Outer tint_symbol_5 = {a_load_2((offset + 0u))};
  return tint_symbol_5;
}

typedef Outer a_load_ret[4];
a_load_ret a_load(uint offset) {
  Outer arr_1[4] = (Outer[4])0;
  {
    for(uint i_2 = 0u; (i_2 < 4u); i_2 = (i_2 + 1u)) {
      arr_1[i_2] = a_load_1((offset + (i_2 * 256u)));
    }
  }
  return arr_1;
}

[numthreads(1, 1, 1)]
void f() {
  int p_a_i_save = i();
  int p_a_i_a_i_save = i();
  int p_a_i_a_i_m_i_save = i();
  Outer l_a[4] = a_load(0u);
  Outer l_a_i = a_load_1((256u * min(uint(p_a_i_save), 3u)));
  Inner l_a_i_a[4] = a_load_2((256u * min(uint(p_a_i_save), 3u)));
  Inner l_a_i_a_i = a_load_3(((256u * min(uint(p_a_i_save), 3u)) + (64u * min(uint(p_a_i_a_i_save), 3u))));
  matrix<float16_t, 4, 4> l_a_i_a_i_m = a_load_4(((256u * min(uint(p_a_i_save), 3u)) + (64u * min(uint(p_a_i_a_i_save), 3u))));
  const uint scalar_offset_4 = ((((256u * min(uint(p_a_i_save), 3u)) + (64u * min(uint(p_a_i_a_i_save), 3u))) + (8u * min(uint(p_a_i_a_i_m_i_save), 3u)))) / 4;
  uint4 ubo_load_9 = a[scalar_offset_4 / 4];
  uint2 ubo_load_8 = ((scalar_offset_4 & 2) ? ubo_load_9.zw : ubo_load_9.xy);
  vector<float16_t, 2> ubo_load_8_xz = vector<float16_t, 2>(f16tof32(ubo_load_8 & 0xFFFF));
  vector<float16_t, 2> ubo_load_8_yw = vector<float16_t, 2>(f16tof32(ubo_load_8 >> 16));
  vector<float16_t, 4> l_a_i_a_i_m_i = vector<float16_t, 4>(ubo_load_8_xz[0], ubo_load_8_yw[0], ubo_load_8_xz[1], ubo_load_8_yw[1]);
  int tint_symbol = p_a_i_save;
  int tint_symbol_1 = p_a_i_a_i_save;
  int tint_symbol_2 = p_a_i_a_i_m_i_save;
  int tint_symbol_3 = i();
  const uint scalar_offset_bytes = (((((256u * min(uint(tint_symbol), 3u)) + (64u * min(uint(tint_symbol_1), 3u))) + (8u * min(uint(tint_symbol_2), 3u))) + (2u * min(uint(tint_symbol_3), 3u))));
  const uint scalar_offset_index = scalar_offset_bytes / 4;
  float16_t l_a_i_a_i_m_i_i = float16_t(f16tof32(((a[scalar_offset_index / 4][scalar_offset_index % 4] >> (scalar_offset_bytes % 4 == 0 ? 0 : 16)) & 0xFFFF)));
  return;
}
