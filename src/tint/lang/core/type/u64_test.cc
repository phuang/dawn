// Copyright 2025 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/tint/lang/core/type/u64.h"

#include "src/tint/lang/core/type/helper_test.h"
#include "src/tint/lang/core/type/manager.h"
#include "src/tint/lang/core/type/u32.h"

namespace tint::core::type {
namespace {

using U64Test = TestHelper;

TEST_F(U64Test, Creation) {
    Manager ty;
    auto* a = ty.u64();
    EXPECT_TRUE(a->Is<U64>());
}

TEST_F(U64Test, SizeAndAlign) {
    Manager ty;
    auto* a = ty.u64();
    EXPECT_EQ(a->Size(), 8u);
    EXPECT_EQ(a->Align(), 8u);
}

TEST_F(U64Test, Hash) {
    Manager ty;
    auto* a = ty.u64();
    auto* b = ty.u64();
    EXPECT_EQ(a->unique_hash, b->unique_hash);
}

TEST_F(U64Test, Equals) {
    Manager ty;
    auto* a = ty.u64();
    auto* b = ty.u64();
    auto* c = ty.u32();
    EXPECT_TRUE(a->Equals(*b));
    EXPECT_FALSE(a->Equals(*c));
}

TEST_F(U64Test, FriendlyName) {
    U64 i;
    EXPECT_EQ(i.FriendlyName(), "u64");
}

TEST_F(U64Test, Clone) {
    Manager ty;
    auto* a = ty.u64();

    core::type::Manager mgr;
    core::type::CloneContext ctx{{nullptr}, {nullptr, &mgr}};

    auto* b = a->Clone(ctx);
    ASSERT_TRUE(b->Is<U64>());
}

}  // namespace
}  // namespace tint::core::type
