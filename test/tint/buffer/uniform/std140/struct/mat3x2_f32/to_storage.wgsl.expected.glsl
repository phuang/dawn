#version 310 es


struct S_std140 {
  int before;
  uint tint_pad_0;
  vec2 m_col0;
  vec2 m_col1;
  vec2 m_col2;
  uint tint_pad_1;
  uint tint_pad_2;
  uint tint_pad_3;
  uint tint_pad_4;
  uint tint_pad_5;
  uint tint_pad_6;
  uint tint_pad_7;
  uint tint_pad_8;
  int after;
  uint tint_pad_9;
  uint tint_pad_10;
  uint tint_pad_11;
  uint tint_pad_12;
  uint tint_pad_13;
  uint tint_pad_14;
  uint tint_pad_15;
  uint tint_pad_16;
  uint tint_pad_17;
  uint tint_pad_18;
  uint tint_pad_19;
  uint tint_pad_20;
  uint tint_pad_21;
  uint tint_pad_22;
  uint tint_pad_23;
};

struct S {
  int before;
  uint tint_pad_0;
  mat3x2 m;
  uint tint_pad_1;
  uint tint_pad_2;
  uint tint_pad_3;
  uint tint_pad_4;
  uint tint_pad_5;
  uint tint_pad_6;
  uint tint_pad_7;
  uint tint_pad_8;
  int after;
  uint tint_pad_9;
  uint tint_pad_10;
  uint tint_pad_11;
  uint tint_pad_12;
  uint tint_pad_13;
  uint tint_pad_14;
  uint tint_pad_15;
  uint tint_pad_16;
  uint tint_pad_17;
  uint tint_pad_18;
  uint tint_pad_19;
  uint tint_pad_20;
  uint tint_pad_21;
  uint tint_pad_22;
  uint tint_pad_23;
};

layout(binding = 0, std140)
uniform u_block_std140_1_ubo {
  S_std140 inner[4];
} v;
layout(binding = 1, std430)
buffer s_block_1_ssbo {
  S inner[4];
} v_1;
S tint_convert_S(S_std140 tint_input) {
  return S(tint_input.before, 0u, mat3x2(tint_input.m_col0, tint_input.m_col1, tint_input.m_col2), 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, tint_input.after, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}
void tint_store_and_preserve_padding_1(uint target_indices[1], S value_param) {
  v_1.inner[target_indices[0u]].before = value_param.before;
  v_1.inner[target_indices[0u]].m = value_param.m;
  v_1.inner[target_indices[0u]].after = value_param.after;
}
void tint_store_and_preserve_padding(S value_param[4]) {
  {
    uint v_2 = 0u;
    v_2 = 0u;
    while(true) {
      uint v_3 = v_2;
      if ((v_3 >= 4u)) {
        break;
      }
      tint_store_and_preserve_padding_1(uint[1](v_3), value_param[v_3]);
      {
        v_2 = (v_3 + 1u);
      }
      continue;
    }
  }
}
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
  S_std140 v_4[4] = v.inner;
  S v_5[4] = S[4](S(0, 0u, mat3x2(vec2(0.0f), vec2(0.0f), vec2(0.0f)), 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u), S(0, 0u, mat3x2(vec2(0.0f), vec2(0.0f), vec2(0.0f)), 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u), S(0, 0u, mat3x2(vec2(0.0f), vec2(0.0f), vec2(0.0f)), 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u), S(0, 0u, mat3x2(vec2(0.0f), vec2(0.0f), vec2(0.0f)), 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
  {
    uint v_6 = 0u;
    v_6 = 0u;
    while(true) {
      uint v_7 = v_6;
      if ((v_7 >= 4u)) {
        break;
      }
      v_5[v_7] = tint_convert_S(v_4[v_7]);
      {
        v_6 = (v_7 + 1u);
      }
      continue;
    }
  }
  tint_store_and_preserve_padding(v_5);
  S v_8 = tint_convert_S(v.inner[2u]);
  tint_store_and_preserve_padding_1(uint[1](1u), v_8);
  v_1.inner[3u].m = mat3x2(v.inner[2u].m_col0, v.inner[2u].m_col1, v.inner[2u].m_col2);
  v_1.inner[1u].m[0u] = v.inner[0u].m_col1.yx;
}
