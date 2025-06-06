// Copyright 2022 The Dawn & Tint Authors
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

#include "src/tint/lang/core/type/bool.h"
#include "src/tint/lang/core/type/helper_test.h"
#include "src/tint/lang/core/type/manager.h"
#include "src/tint/lang/core/type/void.h"

namespace tint::core::type {
namespace {

using BoolTest = TestHelper;

TEST_F(BoolTest, Creation) {
    Manager ty;
    auto* a = ty.bool_();
    auto* b = ty.bool_();
    EXPECT_EQ(a, b);
}

TEST_F(BoolTest, Hash) {
    Manager ty;
    auto* a = ty.bool_();
    auto* b = ty.bool_();
    EXPECT_EQ(a->unique_hash, b->unique_hash);
}

TEST_F(BoolTest, Equals) {
    Manager ty;
    auto* a = ty.bool_();
    auto* b = ty.bool_();
    EXPECT_TRUE(a->Equals(*b));
    EXPECT_FALSE(a->Equals(Void{}));
}

TEST_F(BoolTest, FriendlyName) {
    Manager ty;
    Bool b;
    EXPECT_EQ(b.FriendlyName(), "bool");
}

TEST_F(BoolTest, Clone) {
    Manager ty;
    auto* a = ty.bool_();
    core::type::Manager mgr;
    core::type::CloneContext ctx{{nullptr}, {nullptr, &mgr}};

    auto* b = a->Clone(ctx);
    ASSERT_TRUE(b->Is<Bool>());
}

}  // namespace
}  // namespace tint::core::type
