// Copyright 2023 The Tint Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/tint/ir/control_instruction.h"

TINT_INSTANTIATE_TYPEINFO(tint::ir::ControlInstruction);

namespace tint::ir {

ControlInstruction::~ControlInstruction() = default;

void ControlInstruction::AddExit(Exit* exit) {
    exits_.Add(exit);
}

void ControlInstruction::RemoveExit(Exit* exit) {
    exits_.Remove(exit);
}

}  // namespace tint::ir