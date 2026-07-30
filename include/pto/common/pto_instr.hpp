/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_INSTR_HPP
#define PTO_INSTR_HPP

#include "pto/common/debug.h"
#include "pto/common/pto_instr_impl.hpp"
#include <common/generated/pto_isa_v0571.hpp>

#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__)

namespace pto {
namespace helper {

template <typename T, typename AddrType>
PTO_INST void TASSIGN(T &obj, AddrType addr) {
  MAP_INSTR_IMPL(TASSIGN, obj, addr);
}

#ifndef __CPU_SIM
template <Op OpCode>
PTO_INST void TSYNC() {
  TSYNC_IMPL<OpCode>();
}
#endif

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &...events) {
  WaitAllEvents(events...);
}

} // namespace helper

// The sole direct public operation surface is generated from the locked
// pto-spec 0.57.1 catalog. Hand-written convenience overloads are forbidden.
#include <pto/common/generated/pto_isa_v0571_api.inc>

} // namespace pto

#endif // PTO_INSTR_HPP
