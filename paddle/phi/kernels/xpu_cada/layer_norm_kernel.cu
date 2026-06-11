// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// xpu_cada override of paddle/phi/kernels/gpu/layer_norm_kernel.cu
//
// This TU replaces the gpu/ version at file granularity (see
// paddle/phi/kernels/CMakeLists.txt :: _xpu_cada_override). Registration
// stays on Backend::GPU; no new backend is introduced.
//
// Current state: stub — body throws Unimplemented, mirroring the adam_kernel
// xpu_cada stub. Fill in the m100-tuned implementation (or copy the gpu/
// body verbatim) before enabling WITH_XPU_CADA in a path that actually
// dispatches to layer_norm.
//
// Registration must mirror gpu/layer_norm_kernel.cu :766-795 exactly,
// including the dtype set selected by the host-toolchain macros
// (PADDLE_WITH_HIP / CUDNN_VERSION_MIN(8,1,0) / else) and the
// SetDataType(UNDEFINED) metadata on outputs 1 and 2.

#include "paddle/phi/kernels/layer_norm_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
#include "paddle/phi/kernels/funcs/fast_ln_v2.h"
#endif
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#include "paddle/phi/kernels/funcs/layer_norm_util.h"

namespace phi {

template <typename T, typename Context>
void LayerNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const optional<DenseTensor>& scale,
                     const optional<DenseTensor>& bias,
                     float epsilon,
                     int begin_norm_axis,
                     DenseTensor* out,
                     DenseTensor* mean,
                     DenseTensor* variance) {
  // TODO(xpu_cada): paste / re-implement the body of phi::LayerNormKernel
  // from paddle/phi/kernels/gpu/layer_norm_kernel.cu here, then optimize
  // for m100 SIMT.
  PADDLE_THROW(phi::errors::Unimplemented(
      "xpu_cada LayerNormKernel is a stub. Fill in the m100 optimized "
      "implementation or copy the gpu/ body before enabling WITH_XPU_CADA."));
}

// LayerNormDirectCUDAFunctor is exported as PADDLE_API from
// gpu/layer_norm_kernel.cu (see L461-495 there). Because xpu_cada replaces
// that TU at file granularity, the explicit instantiations below MUST be
// produced here as well — otherwise phi_core.so links break with
// `undefined reference to LayerNormDirectCUDAFunctor<...>::operator()`.
//
// The implementation is a verbatim copy of the gpu/ version. Do not change
// it unless you also intend to optimize the functor for m100.
template <typename T, typename U>
void LayerNormDirectCUDAFunctor<T, U>::operator()(
    gpuStream_t stream,
    const T* input,
    std::vector<int64_t> input_shape,
    const U* bias,
    const U* scale,
    T* output,
    U* mean,
    U* variance,
    int begin_norm_axis,
    float eps) {
  const auto x_dims = make_ddim(input_shape);
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = matrix_dim[0];
  int64_t feature_size = matrix_dim[1];
  // TODO(large-tensor): generic kernel launch uses int32 grid dim
  PADDLE_ENFORCE_LE_INT_MAX(batch_size, "batch_size");
  switch (funcs::GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        funcs::LayerNormForward<T, U, kBlockDim>
        <<<batch_size, kBlockDim, 0, stream>>>(
            input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
}

template class PADDLE_API LayerNormDirectCUDAFunctor<float, float>;
template class PADDLE_API LayerNormDirectCUDAFunctor<double, double>;
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
template class PADDLE_API LayerNormDirectCUDAFunctor<half, float>;
#endif

}  // namespace phi

// -------- registration: keep Backend::GPU, dtype set, and metadata --------
// Mirror gpu/layer_norm_kernel.cu :766-795 byte-for-byte.

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(
    layer_norm, GPU, ALL_LAYOUT, phi::LayerNormKernel, float, phi::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
#elif CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
#else
PD_REGISTER_KERNEL(layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   double,
                   phi::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
#endif
