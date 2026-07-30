#include <pto/pto-inst.hpp>

static_assert(pto::isa_v0571::kOperationCount == 120u,
              "PTO 0.57.1 public header must expose the 120-op registry");
static_assert(pto::isa_v0571::tepl::TPRELU_MODE == 0u,
              "TPRELU Mode parity");
static_assert(pto::isa_v0571::tepl::TPRELU_FUNCTION == 14u,
              "TPRELU Function parity");
static_assert(pto::isa_v0571::tepl::TPRELU == 0x00eu,
              "TPRELU raw selector parity");
static_assert(pto::isa_v0571::tma::MGATHER_CAS_FUNCTION == 8u,
              "typed TMA selector parity");
static_assert(pto::isa_v0571::cube::ACCCVT_FUNCTION == 8u,
              "typed CUBE selector parity");

struct Operand {};

int main() {
  Operand operand;
#include "generated/pto_isa_v0571_calls.inc"
  return 0;
}
