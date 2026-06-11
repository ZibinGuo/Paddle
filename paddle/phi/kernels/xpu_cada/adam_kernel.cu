// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// xpu_cada override of paddle/phi/kernels/gpu/adam_kernel.cu
//
// Granularity note:
//   gpu/adam_kernel.cu is a single TU that registers TWO kernels:
//     - PD_REGISTER_KERNEL(adam,        GPU, ALL_LAYOUT, phi::AdamDenseKernel,  ...)
//     - PD_REGISTER_KERNEL(merged_adam, GPU, ALL_LAYOUT, phi::MergedAdamKernel, ...)
//
//   The xpu_cada mechanism replaces at TU granularity, so this file MUST
//   provide BOTH kernels. Optimize the one(s) you care about; for the rest,
//   keep behavior identical to the gpu/ version (recommended way: extract a
//   shared header `kernels/impl/adam_kernel_impl.h` upstream, then both
//   gpu/adam_kernel.cu and this file just instantiate from it).
//
//   Until that refactor lands, the simplest path is to copy the unchanged
//   kernel body verbatim from gpu/adam_kernel.cu so this TU is self-contained.
//
// Registration MUST stay on Backend::GPU (no new backend introduced).

#include "paddle/phi/kernels/adam_kernel.h"

#include <math.h>
#include <vector>

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

// =============================================================
// 1) AdamDenseKernel  -- m100 / xpu_cada optimized implementation
// =============================================================
//
// TODO(xpu_cada): replace this body with the m100-tuned version.
// Suggested optimization angles for SIMT on m100:
//   - vectorized load/store of (param, grad, moment1, moment2)
//   - block-level reduction for beta_pow update
//   - amp/master-param path specialization
//
// For now this is a placeholder that simply forwards to a copy of the
// original gpu kernel logic. Replace incrementally; correctness must
// match gpu/adam_kernel.cu exactly for every dtype in the register list.
template <typename T, typename Context>
void AdamDenseKernel(const Context& dev_ctx,
                     const DenseTensor& param,
                     const DenseTensor& grad,
                     const DenseTensor& learning_rate,
                     const DenseTensor& moment1,
                     const DenseTensor& moment2,
                     const paddle::optional<DenseTensor>& moment2_max,
                     const DenseTensor& beta1_pow,
                     const DenseTensor& beta2_pow,
                     const paddle::optional<DenseTensor>& master_param,
                     const paddle::optional<DenseTensor>& skip_update,
                     const Scalar& beta1,
                     const Scalar& beta2,
                     const Scalar& epsilon,
                     bool lazy_mode,
                     int64_t min_row_size_to_use_multithread,
                     bool multi_precision,
                     bool use_global_beta_pow,
                     bool amsgrad,
                     DenseTensor* param_out,
                     DenseTensor* moment1_out,
                     DenseTensor* moment2_out,
                     DenseTensor* moment2_max_out,
                     DenseTensor* beta1_pow_out,
                     DenseTensor* beta2_pow_out,
                     DenseTensor* master_param_outs) {
  // TODO(xpu_cada): paste / re-implement the body of phi::AdamDenseKernel
  // from paddle/phi/kernels/gpu/adam_kernel.cu here, then optimize.
  PADDLE_THROW(phi::errors::Unimplemented(
      "xpu_cada AdamDenseKernel is a stub. Fill in the m100 optimized "
      "implementation or copy the gpu/ body before enabling WITH_XPU_CADA."));
}

// =============================================================
// 2) MergedAdamKernel -- keep gpu/ semantics unchanged
// =============================================================
//
// If you do NOT want to optimize merged_adam, paste the original body
// from gpu/adam_kernel.cu (phi::MergedAdamKernel) verbatim below.
// It must remain bit-identical to the gpu version, otherwise users who
// only enable WITH_XPU_CADA for adam will silently get a different
// merged_adam implementation.
template <typename T, typename Context>
void MergedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& grad,
    const std::vector<const DenseTensor*>& learning_rate,
    const std::vector<const DenseTensor*>& moment1,
    const std::vector<const DenseTensor*>& moment2,
    const paddle::optional<std::vector<const DenseTensor*>>& moment2_max,
    const std::vector<const DenseTensor*>& beta1_pow,
    const std::vector<const DenseTensor*>& beta2_pow,
    const paddle::optional<std::vector<const DenseTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad,
    std::vector<DenseTensor*> param_out,
    std::vector<DenseTensor*> moment1_out,
    std::vector<DenseTensor*> moment2_out,
    std::vector<DenseTensor*> moment2_max_out,
    std::vector<DenseTensor*> beta1_pow_out,
    std::vector<DenseTensor*> beta2_pow_out,
    std::vector<DenseTensor*> master_param_out) {
  // TODO(xpu_cada): paste body of phi::MergedAdamKernel from
  // paddle/phi/kernels/gpu/adam_kernel.cu here. Do NOT change semantics
  // unless you are intentionally optimizing merged_adam for m100.
  PADDLE_THROW(phi::errors::Unimplemented(
      "xpu_cada MergedAdamKernel is a stub. Copy the gpu/ body before "
      "enabling WITH_XPU_CADA."));
}

}  // namespace phi

// -------- registration: keep Backend::GPU and the same dtype set --------

PD_REGISTER_KERNEL(adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdamDenseKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  // Mirror gpu/adam_kernel.cu metadata exactly.
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);

  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}

PD_REGISTER_KERNEL(merged_adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::MergedAdamKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);

  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  }
}
