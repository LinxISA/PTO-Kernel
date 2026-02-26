cmake_minimum_required(VERSION 3.20)
project(PTOKernel LANGUAGES CXX)

option(PTO_ENABLE_TESTS "Enable PTO-Kernel local tests" ON)
option(PTO_ENABLE_LINX_CROSS "Enable Linx cross asm contract checks" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(pto_headers INTERFACE)
target_include_directories(pto_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(PTO_ENABLE_TESTS)
  include(CTest)
  if(BUILD_TESTING)
    add_executable(pto_domain_smoke tests/domain_smoke.cpp)
    target_link_libraries(pto_domain_smoke PRIVATE pto_headers)
    add_test(NAME pto_domain_smoke COMMAND pto_domain_smoke)
  endif()
endif()

if(PTO_ENABLE_LINX_CROSS)
  find_program(PTO_CLANGXX NAMES clang++)
  if(NOT PTO_CLANGXX)
    message(FATAL_ERROR "clang++ not found for PTO_ENABLE_LINX_CROSS")
  endif()

  set(PTO_KERNELS
    tload_store
    mamulb
    tmatmul_acc
    gemm
    gemm_basic
    gemm_demo
    gemm_performance
    add_custom
    flash_attention
    flash_attention_demo
    flash_attention_masked
    fa_performance
    mla_attention_demo
    flash_attention_cube_fp16
    flash_attention_cube_fp8_e4m3
    flash_attention_cube_fp4_e2m1
    flash_attention_vec_fp32
    flash_attention_vec_fp16
    mha_fp16
    gqa_fp16
    rope_apply_fp16
    attention_dropout_fp16
    flash_mla_deepseekv3_fp16
    flash_mla_deepseekv3_fp8_e4m3
    ifa_mla_seq1_fp16
    ifa_gqa_seq1_fp16
    paged_attention_mha_fp16
    paged_attention_gqa_fp16
    moe_sort_fp32
    moe_topk_fp32
    moe_gate_route_fp16
    moe_mlp_fp16
    gemm_reuse_a_fp16
    gemm_reuse_b_fp16
    gemm_reuse_ab_fp16
    flash_attention_backward_fp16
    flash_attention_backward_fp32
    sparse_attention_local_fp16
    sparse_attention_block_fp16
    rmsnorm_fp16
    relu_fp32
    sigmoid_fp32
    softmax_fp32
    tanh_fp32
    silu_fp32
    gelu_fp32
    swiglu_fp16
    layernorm_fp16
    batchnorm_fp16
    argmax_fp32
    matmul_a8w8
    matmul_a8w4
    matmul_a16w8
    matmul_a16w4
    decode_greedy_search_fp32
    decode_beam_search_fp32
    decode_temperature_scaling_fp32
    decode_topk_filter_fp32
    decode_topp_nucleus_filter_fp32
    transpose_large_fp32
    permute_nhwc_nchw_fp32
    slice_fp32
    gather_fp32
    scatter_fp32
    where_fp32
    concat_fp32
    split_fp32
    stack_fp32
    reshape_fp32
    flatten_fp32
    squeeze_fp32
    unsqueeze_fp32
    hash_table_lookup_fp32
    hash_table_insert_fp32
    unsorted_segment_sum_fp32
    unique_i32
  )

  set(PTO_ASM_OUTPUTS)
  foreach(k IN LISTS PTO_KERNELS)
    set(out_asm "${CMAKE_BINARY_DIR}/linx_asm/${k}.s")
    add_custom_command(
      OUTPUT "${out_asm}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/linx_asm"
      COMMAND ${PTO_CLANGXX}
              -target linx64-linx-none-elf
              -O2
              -S
              -ffreestanding
              -fno-builtin
              -fno-stack-protector
              -fno-exceptions
              -fno-rtti
              -nostdlib
              -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
              "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${k}.cpp"
              -o "${out_asm}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${k}.cpp"
      VERBATIM)
    list(APPEND PTO_ASM_OUTPUTS "${out_asm}")
  endforeach()

  add_custom_target(pto_linx_asm DEPENDS ${PTO_ASM_OUTPUTS})
  add_custom_target(
    pto_linx_contracts
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_tile_asm_contracts.py ${PTO_ASM_OUTPUTS}
    DEPENDS pto_linx_asm
    VERBATIM)
endif()
