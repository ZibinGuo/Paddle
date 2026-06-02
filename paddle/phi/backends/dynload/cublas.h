/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cublasXt.h>
#include <cublas_v2.h>
#include <cuda.h>
#if 1 || (CUDA_VERSION >= 12030 && defined(__linux__))
#include <cublas_api.h>
#endif

#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag cublas_dso_flag;
extern void *cublas_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)                            \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using cublas_func =                                                   \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);         \
      std::call_once(cublas_dso_flag, []() {                                \
        cublas_dso_handle = phi::dynload::GetCublasDsoHandle();             \
      });                                                                   \
      static void *p_##__name = dlsym(cublas_dso_handle, #__name);          \
      return reinterpret_cast<cublas_func>(p_##__name)(args...);            \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name

  #define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSaxpy);                   \
  __macro(cublasDaxpy);                   \
  __macro(cublasCaxpy);                   \
  __macro(cublasZaxpy);                   \
  __macro(cublasSscal);                   \
  __macro(cublasDscal);                   \
  __macro(cublasScopy);                   \
  __macro(cublasDcopy);                   \
  __macro(cublasSgemv_v2);                \
  __macro(cublasDgemv_v2);                \
  __macro(cublasCgemv_v2);                \
  __macro(cublasZgemv_v2);                \
  __macro(cublasSgemm_v2);                \
  __macro(cublasDgemm_v2);                \
  __macro(cublasCgemm_v2);                \
  __macro(cublasZgemm_v2);                \
  __macro(cublasSgeam);                   \
  __macro(cublasDgeam);                   \
  __macro(cublasStrsm_v2);                \
  __macro(cublasDtrsm_v2);                \
  __macro(cublasCtrsm_v2);                \
  __macro(cublasZtrsm_v2);                \
  __macro(cublasCreate_v2);               \
  __macro(cublasDestroy_v2);              \
  __macro(cublasSetStream_v2);            \
  __macro(cublasSetPointerMode_v2);       \
  __macro(cublasGetPointerMode_v2);       \
  __macro(cublasSgemmBatched);            \
  __macro(cublasDgemmBatched);            \
  __macro(cublasCgemmBatched);            \
  __macro(cublasZgemmBatched);            \
  __macro(cublasStrsmBatched);            \
  __macro(cublasDtrsmBatched);            \
  __macro(cublasCtrsmBatched);            \
  __macro(cublasZtrsmBatched);            \
  __macro(cublasSgetrfBatched);           \
  __macro(cublasSgetriBatched);           \
  __macro(cublasDgetrfBatched);           \
  __macro(cublasDgetriBatched);           \
  __macro(cublasCgetrfBatched);           \
  __macro(cublasCgetriBatched);           \
  __macro(cublasZgetrfBatched);           \
  __macro(cublasZgetriBatched);           \
  __macro(cublasSgetrsBatched);           \
  __macro(cublasDgetrsBatched);           \
  __macro(cublasSdot_v2);                 \
  __macro(cublasDdot_v2);                 \
  __macro(cublasCdotc_v2);                \
  __macro(cublasZdotc_v2);                \
  __macro(cublasCdotu_v2);                \
  __macro(cublasZdotu_v2);                \
  __macro(cublasDotEx);                   \
  __macro(cublasSgemmStridedBatched);     \
  __macro(cublasDgemmStridedBatched);     \
  __macro(cublasCgemmStridedBatched);     \
  __macro(cublasZgemmStridedBatched);     \
  __macro(cublasCgeam);                   \
  __macro(cublasZgeam);                   \
  /* Verification-failed entries stay commented out below. */

// Current xtrans libcublas does not export these symbols.
// #define CUBLAS_UNSUPPORTED_ROUTINE_EACH(__macro) \
//   __macro(cublasSaxpy_v2);                      \
//   __macro(cublasDaxpy_v2);                      \
//   __macro(cublasCaxpy_v2);                      \
//   __macro(cublasZaxpy_v2);                      \
//   __macro(cublasSscal_v2);                      \
//   __macro(cublasDscal_v2);                      \
//   __macro(cublasScopy_v2);                      \
//   __macro(cublasDcopy_v2);                      \
//   __macro(cublasSmatinvBatched);                \
//   __macro(cublasDmatinvBatched);                \
//   __macro(cublasCmatinvBatched);                \
//   __macro(cublasZmatinvBatched);

CUBLAS_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)

#if 1 || (CUDA_VERSION >= 12030 && defined(__linux__))
// Keep the R5 macro defined but empty because all xtrans R5 additions below
// fail API verification and must not generate active dynload wrappers.
#define CUBLAS_BLAS_ROUTINE_EACH_R5(__macro)

CUBLAS_BLAS_ROUTINE_EACH_R5(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)

// Current xtrans libcublas exports these symbols, but they fail API
// verification, so keep the deleted dynload entries as comments instead of
// generating active phi::dynload wrappers for them.
// #define CUBLAS_VERIFICATION_FAILED_ROUTINE_EACH(__macro) \
//   __macro(cublasHgemm);                   \
// xblas path returns CUBLAS_STATUS_INVALID_VALUE; ENABLE_XBLAS=false can route
// to a working xcnblas backend.
//   __macro(cublasSgemmEx);                 \
// half Ex path fails in xblas; ENABLE_XBLAS=false returns
// CUBLAS_STATUS_NOT_SUPPORTED because there is no fallback.
//   __macro(cublasGemmEx);                  \
// xcnblas_gemm_ex returns success but does not write output.
//   __macro(cublasHgemmStridedBatched);     \
// xblas path returns CUBLAS_STATUS_INVALID_VALUE; ENABLE_XBLAS=false can route
// to a working xcnblas backend.
//   __macro(cublasSetMathMode);             \
// Set only forwards mode to a companion xblas handle; cuBLAS has no readable
// state loop.
//   __macro(cublasGetMathMode);             \
// Get returns success without writing the mode output pointer.
//   __macro(cublasGemmBatchedEx);           \
// xcnblas_gemm_batched_ex returns success but does not write output.
//   __macro(cublasGemmStridedBatchedEx);
// xcnblas_gemm_strided_batched_ex returns success but does not write output.
// #define CUBLAS_BLAS_ROUTINE_EACH_R5(__macro) \
//   __macro(cublasGemmStridedBatchedEx_64);    \
// Xpumath/libcublas implements this as a success-only empty stub.
//   __macro(cublasGemmEx_64);                  \
// Xpumath/libcublas implements this as a success-only empty stub.
//   __macro(cublasSgemmEx_64);
// Xpumath/libcublas implements this as a success-only empty stub.
#endif

#undef DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP
}  // namespace dynload
}  // namespace phi
