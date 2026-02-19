#include <pto/domains/ai.hpp>
#include <pto/domains/hpc.hpp>
#include <pto/domains/rendering.hpp>
#include <pto/domains/math.hpp>

int main() {
  auto gemm = pto::domains::ai::default_gemm_profile();
  auto stencil = pto::domains::hpc::default_stencil_profile();
  auto shade = pto::domains::rendering::default_shading_profile();
  auto reduce = pto::domains::math::default_reduction_profile();

  if (gemm.m <= 0 || gemm.n <= 0 || gemm.k <= 0)
    return 1;
  if (stencil.rows <= 0 || stencil.cols <= 0)
    return 2;
  if (shade.width <= 0 || shade.height <= 0)
    return 3;
  if (reduce.lanes <= 0)
    return 4;

  return 0;
}
