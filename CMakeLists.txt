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
