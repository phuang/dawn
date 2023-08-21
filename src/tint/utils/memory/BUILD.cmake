# Copyright 2023 The Tint Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# File generated by tools/src/cmd/gen
# using the template:
#   tools/src/cmd/gen/build/BUILD.cmake.tmpl
#
# Do not modify this file directly
################################################################################

################################################################################
# Target:    tint_utils_memory
# Kind:      lib
################################################################################
tint_add_target(tint_utils_memory lib
  utils/memory/bitcast.h
  utils/memory/block_allocator.h
  utils/memory/bump_allocator.h
  utils/memory/memory.cc
)

tint_target_add_dependencies(tint_utils_memory lib
  tint_utils_math
)

################################################################################
# Target:    tint_utils_memory_test
# Kind:      test
################################################################################
tint_add_target(tint_utils_memory_test test
  utils/memory/bitcast_test.cc
  utils/memory/block_allocator_test.cc
  utils/memory/bump_allocator_test.cc
)

tint_target_add_dependencies(tint_utils_memory_test test
  tint_utils_math
  tint_utils_memory
)

tint_target_add_external_dependencies(tint_utils_memory_test test
  "gtest"
)