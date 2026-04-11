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
    find_program(PTO_PYTHON3 NAMES python3)
    if(NOT PTO_PYTHON3)
      message(FATAL_ERROR "python3 not found for PTO manifest validation tests")
    endif()

    add_executable(pto_domain_smoke tests/domain_smoke.cpp)
    target_link_libraries(pto_domain_smoke PRIVATE pto_headers)
    target_compile_definitions(pto_domain_smoke PRIVATE PTO_HOST_SIM=1)
    add_test(NAME pto_domain_smoke COMMAND pto_domain_smoke)

    add_executable(pto_block_vector_smoke tests/block_vector_smoke.cpp)
    target_link_libraries(pto_block_vector_smoke PRIVATE pto_headers)
    target_compile_definitions(pto_block_vector_smoke PRIVATE PTO_HOST_SIM=1)
    add_test(NAME pto_block_vector_smoke COMMAND pto_block_vector_smoke)

    add_test(
      NAME pto_benchmark_manifest
      COMMAND ${PTO_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_benchmark_manifest.py)
  endif()
endif()

if(PTO_ENABLE_LINX_CROSS)
  set(PTO_CLANGXX_DEFAULT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../compiler/llvm/build-linxisa-clang/bin/clang++")
  if(EXISTS "${PTO_CLANGXX_DEFAULT}")
    set(PTO_CLANGXX "${PTO_CLANGXX_DEFAULT}" CACHE FILEPATH
      "clang++ used for Linx cross asm checks" FORCE)
  else()
    set(PTO_CLANGXX "${PTO_CLANGXX_DEFAULT}" CACHE FILEPATH
      "clang++ used for Linx cross asm checks")
  endif()
  if(NOT EXISTS "${PTO_CLANGXX}")
    find_program(PTO_CLANGXX NAMES clang++)
  endif()
  if(NOT PTO_CLANGXX)
    message(FATAL_ERROR "clang++ not found for PTO_ENABLE_LINX_CROSS")
  endif()

  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/catalog.txt" PTO_KERNEL_PATHS)

  set(PTO_ASM_OUTPUTS)
  foreach(rel_path IN LISTS PTO_KERNEL_PATHS)
    if(rel_path STREQUAL "" OR rel_path MATCHES "^#")
      continue()
    endif()
    get_filename_component(k "${rel_path}" NAME_WE)
    set(src_path "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${rel_path}")
    set(out_asm "${CMAKE_BINARY_DIR}/linx_asm/${k}.s")
    add_custom_command(
      OUTPUT "${out_asm}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/linx_asm"
      COMMAND ${PTO_CLANGXX}
              -target linx64-linx-none-elf
              -O2
              -S
              -DPTO_USE_MIXED_TILE_SIMT=1
              -ffreestanding
              -fno-builtin
              -fno-stack-protector
              -fno-exceptions
              -fno-rtti
              -nostdlib
              -I${CMAKE_CURRENT_SOURCE_DIR}/include
              "${src_path}"
              -o "${out_asm}"
      DEPENDS "${src_path}"
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
