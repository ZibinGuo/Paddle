// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/selected_rows.h"

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace imperative {

const phi::DenseTensor* GetTensorFromVar(const framework::Variable& var);
const phi::DenseTensor* GetDebugTensorFromVar(const framework::Variable& var);
template <typename VarType>
static void SetForwardDataTypeOfGradVar(const std::shared_ptr<VarType>& var);

template <>
void SetForwardDataTypeOfGradVar<VariableWrapper>(
    const std::shared_ptr<VariableWrapper>& var) {
  if (var->HasGradVar()) {
    auto grad_var = var->GetGradVar();
    VLOG(6) << "Set grad var (" << grad_var->Name() << ")'s forward dtype to ("
            << framework::DataTypeToString(var->DataType()) << ").";
    grad_var->SetForwardDataType(var->DataType());
  }
}

template <>
void SetForwardDataTypeOfGradVar<VarBase>(const std::shared_ptr<VarBase>& var) {
  if (var->HasGradVar()) {
    auto& shared_var = var->SharedVar();
    SetForwardDataTypeOfGradVar<VariableWrapper>(shared_var);
  }
}

template <>
void SetForwardDataTypeOfGradVar<egr::EagerVariable>(
    const std::shared_ptr<egr::EagerVariable>& var) {
  VLOG(10) << "Var in Eager dose not support SetForwardDataTypeOfGradVar: "
           << var->name();
  // TODO(jiabin): SetForwardDataType of Grad var is not supported yet in
  // EagerMode.
}

template <typename VarType>
std::shared_ptr<NameVarMap<VarType>> TemporaryData(
    const NameVarMap<VarType>& ins, const platform::Place& place) {
  std::shared_ptr<NameVarMap<VarType>> tmp_ins_ptr = nullptr;
  if (ins.empty()) {
    tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
    return tmp_ins_ptr;
  }

  for (const auto& name_pair : ins) {
    for (size_t i = 0; i < name_pair.second.size(); ++i) {
      auto& template_var = name_pair.second[i];
      if (template_var == nullptr) {
        if (tmp_ins_ptr == nullptr) {
          tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
          continue;
        } else {
          (*tmp_ins_ptr)[name_pair.first][i] = template_var;
          continue;
        }
      }
      const auto* tensor = GetTensorFromVar(template_var->Var());
      const auto* debug_tensor = GetDebugTensorFromVar(template_var->Var());
      if (tensor && tensor->IsInitialized() && (tensor->memory_size() != 0)) {
        if (debug_tensor && debug_tensor->IsInitialized() &&
            (debug_tensor->memory_size() != 0)) {
          VLOG(10) << name_pair.first << "-"
                   << GetNameFromVar(name_pair.second[i])
                   << "  tensor->memory_size() = " << tensor->memory_size()
                   << ", tensor->meta() = " << tensor->meta();
          VLOG(10) << "  debug_tensor->memory_size() = "
                   << debug_tensor->memory_size()
                   << ", debug_tensor->meta() = " << debug_tensor->meta();
          if (std::getenv("XPU_PADDLE_DEBUG_OP") != nullptr) {
            paddle::framework::TransformData(
                *tensor, place, const_cast<framework::Tensor*>(debug_tensor));
          }
          // tensor = debug_tensor;
        } else {
          paddle::framework::SetVoidVariableDebug(template_var->MutableVar());
          debug_tensor = GetDebugTensorFromVar(template_var->Var());
          paddle::framework::TransformData(
              *tensor, place, const_cast<framework::Tensor*>(debug_tensor));
          // tensor = debug_tensor;
        }
        VLOG(3) << "Transform Variable " << GetNameFromVar(template_var)
                << " from "
                << "{data_type[" << tensor->dtype() << "]; data_layout["
                << tensor->layout() << "]; place[" << tensor->place() << "]"
                << " to "
                << "place[" << place << "]";
        VLOG(3) << GetNameFromVar(template_var)
                << " memory size is: " << tensor->memory_size();
        if (tmp_ins_ptr == nullptr) {
          tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
        }
        auto tmp_var = std::make_shared<VarType>(GetNameFromVar(template_var));
        SetType(tmp_var, GetType(template_var));
        SetTensorToVariable(
            template_var->Var(), *debug_tensor, tmp_var->MutableVar());
        (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
      } else {
        if (tmp_ins_ptr == nullptr) {
          tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
        }
        auto tmp_var = std::make_shared<VarType>(GetNameFromVar(template_var));
        if (template_var->Var().IsInitialized()) {
          SetType(tmp_var, GetType(template_var));
          paddle::framework::SetVoidVariableDebug(template_var->MutableVar());
          CopyVoidVariable(template_var->Var(), tmp_var->MutableVar());
        }
        (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
      }

      std::stringstream ss;
      ss << name_pair.first << "-" << GetNameFromVar(name_pair.second[i])
         << "   dev1: ";
      const auto* test_tensor = GetTensorFromVar(template_var->Var());
      if (template_var->Var().IsInitialized()) {
        ss << "var_addr:" << &template_var->Var() << " "
           << "var_holder_:" << test_tensor << " ";
        if (test_tensor && test_tensor->IsInitialized() &&
            (test_tensor->memory_size() != 0)) {
          ss << "tensor_addr:" << test_tensor << " "
             << "tensor_place:" << test_tensor->place() << " "
             << "tensor_meta = " << test_tensor->meta()
             << "tensor_data_addr:" << test_tensor->data() << std::endl;
        } else {
          ss << "NOT_INITED_TENSOR or NonTensor" << std::endl;
        }
      } else {
        ss << "NOT_INITED_VAR " << std::endl;
      }

      test_tensor = GetDebugTensorFromVar(template_var->Var());
      ss << name_pair.first << "-" << GetNameFromVar(name_pair.second[i])
         << "  debug: ";
      if (template_var->Var().IsInitialized()) {
        ss << "var_addr:" << &template_var->Var() << " "
           << "var_holder_:" << test_tensor << " ";
        // ss << "var_place_before:" << GetPlace(template_var) << " ";
        if (test_tensor && test_tensor->IsInitialized() &&
            (test_tensor->memory_size() != 0)) {
          ss << "tensor_addr:" << test_tensor << " "
             << "tensor_place:" << test_tensor->place() << " "
             << "tensor_meta = " << test_tensor->meta()
             << "tensor_data_addr:" << test_tensor->data() << std::endl;
        } else {
          ss << "NOT_INITED_TENSOR or NonTensor" << std::endl;
        }
      } else {
        ss << "NOT_INITED_VAR " << std::endl;
      }

      auto& test_var = (*tmp_ins_ptr)[name_pair.first][i];
      test_tensor = GetTensorFromVar(test_var->Var());
      ss << name_pair.first << "-" << GetNameFromVar(test_var) << "   dev2: ";
      if (test_var->Var().IsInitialized()) {
        ss << "var_addr:" << &test_var << " "
           << "var_holder_:" << test_tensor << " ";
        if (test_tensor && test_tensor->IsInitialized() &&
            (test_tensor->memory_size() != 0)) {
          ss << "tensor_addr:" << test_tensor << " "
             << "tensor_place:" << test_tensor->place() << " "
             << "tensor_meta = " << test_tensor->meta()
             << "tensor_data_addr:" << test_tensor->data() << std::endl;
        } else {
          ss << "NOT_INITED_TENSOR or NonTensor" << std::endl;
        }
      } else {
        ss << "NOT_INITED_VAR " << std::endl;
      }
      VLOG(10) << ss.str();
    }
  }
  return tmp_ins_ptr;
}

template <typename VarType>
void CopyOutputData(const std::string& op_type,
                    const NameVarMap<VarType>& ins,
                    NameVarMap<VarType>* tmp_ins) {
  if (ins.empty()) {
    return;
  }
  for (const auto& name_pair : ins) {
    for (size_t i = 0; i < name_pair.second.size(); ++i) {
      auto& template_var = name_pair.second[i];
      if (template_var == nullptr) {
        continue;
      }
      // const auto* tensor = GetTensorFromVar(template_var->Var());
      const auto* debug_tensor = GetDebugTensorFromVar(template_var->Var());
      auto& template_tmp_ins_var = (*tmp_ins)[name_pair.first][i];
      const auto* tmp_ins_tensor =
          GetTensorFromVar(template_tmp_ins_var->Var());
      if (paddle::platform::is_in_xpu_debug_run_dev2_black_list(op_type)) {
        tmp_ins_tensor = GetTensorFromVar(template_var->Var());
      }
      if (tmp_ins_tensor && tmp_ins_tensor->IsInitialized() &&
          (tmp_ins_tensor->memory_size() != 0)) {
        if (debug_tensor && debug_tensor->IsInitialized() &&
            (debug_tensor->memory_size() != 0)) {
          VLOG(10) << name_pair.first << "-"
                   << GetNameFromVar(name_pair.second[i])
                   << "  tmp_ins_tensor->memory_size() = "
                   << tmp_ins_tensor->memory_size()
                   << ", tmp_ins_tensor->meta() = " << tmp_ins_tensor->meta();
          VLOG(10) << "  debug_tensor->memory_size() = "
                   << debug_tensor->memory_size()
                   << ", debug_tensor->meta() = " << debug_tensor->meta();
          // const_cast<framework::Tensor*>(debug_tensor)->set_meta(tmp_ins_tensor->meta());
          if (paddle::platform::is_in_xpu_debug_run_dev2_black_list(op_type)) {
            paddle::framework::TransformData(
                *tmp_ins_tensor,
                debug_tensor->place(),
                const_cast<framework::Tensor*>(debug_tensor));
            const auto* tmp_ins_tensor_ =
                GetTensorFromVar(template_tmp_ins_var->Var());
            const_cast<framework::Tensor*>(tmp_ins_tensor_)
                ->ShareDataWith(*debug_tensor);
          } else {
            const_cast<framework::Tensor*>(debug_tensor)
                ->ShareDataWith(*tmp_ins_tensor);
          }
        } else {
          paddle::framework::SetVoidVariableDebug(template_var->MutableVar());
          debug_tensor = GetDebugTensorFromVar(template_var->Var());
          paddle::framework::TransformData(
              *tmp_ins_tensor,
              tmp_ins_tensor->place(),
              const_cast<framework::Tensor*>(debug_tensor));
          if (paddle::platform::is_in_xpu_debug_run_dev2_black_list(op_type)) {
            const auto* tmp_ins_tensor_ =
                GetTensorFromVar(template_tmp_ins_var->Var());
            const_cast<framework::Tensor*>(tmp_ins_tensor_)
                ->ShareDataWith(*debug_tensor);
          }
        }
        VLOG(3) << "Transform Variable " << GetNameFromVar(template_var)
                << " from "
                << "{data_type[" << tmp_ins_tensor->dtype() << "]; data_layout["
                << tmp_ins_tensor->layout() << "]; place["
                << tmp_ins_tensor->place() << "]"
                << " to "
                << "]; place[" << debug_tensor->place() << "]";
        VLOG(3) << GetNameFromVar(template_var)
                << " memory size is: " << debug_tensor->memory_size();
      } else {
        if (template_var->Var().IsInitialized()) {
          paddle::framework::SetVoidVariableDebug(template_var->MutableVar());
        }
      }
      std::stringstream ss;
      ss << name_pair.first << "-" << GetNameFromVar(name_pair.second[i])
         << "  dev1: ";
      const auto* test_tensor = GetTensorFromVar(template_var->Var());
      if (template_var->Var().IsInitialized()) {
        ss << "var_addr:" << &template_var->Var() << " "
           << "var_holder_:" << test_tensor << " ";
        if (test_tensor && test_tensor->IsInitialized() &&
            (test_tensor->memory_size() != 0)) {
          ss << "tensor_addr:" << test_tensor << " "
             << "tensor_place:" << test_tensor->place() << " "
             << "tensor_meta = " << test_tensor->meta()
             << "tensor_data_addr:" << test_tensor->data() << std::endl;
        } else {
          ss << "NOT_INITED_TENSOR or NonTensor" << std::endl;
        }
      } else {
        ss << "NOT_INITED_VAR " << std::endl;
      }

      auto& test_var = (*tmp_ins)[name_pair.first][i];
      test_tensor = GetTensorFromVar(test_var->Var());
      ss << name_pair.first << "-" << GetNameFromVar(test_var) << std::endl;
      ss << "    tmp: ";
      if (test_var->Var().IsInitialized()) {
        ss << "var_addr:" << &test_var << " "
           << "var_holder_:" << test_tensor << " ";
        if (test_tensor && test_tensor->IsInitialized() &&
            (test_tensor->memory_size() != 0)) {
          ss << "tensor_addr:" << test_tensor << " "
             << "tensor_place:" << test_tensor->place() << " "
             << "tensor_meta = " << test_tensor->meta()
             << "tensor_data_addr:" << test_tensor->data() << std::endl;
        } else {
          ss << "NOT_INITED_TENSOR or NonTensor" << std::endl;
        }
      } else {
        ss << "NOT_INITED_VAR " << std::endl;
      }

      test_tensor = GetDebugTensorFromVar(template_var->Var());
      ss << name_pair.first << "-" << GetNameFromVar(name_pair.second[i])
         << "    dev2: ";
      if (template_var->Var().IsInitialized()) {
        ss << "var_addr:" << &template_var->Var() << " "
           << "var_holder_:" << test_tensor << " ";
        if (test_tensor && test_tensor->IsInitialized() &&
            (test_tensor->memory_size() != 0)) {
          ss << "tensor_addr:" << test_tensor << " "
             << "tensor_place:" << test_tensor->place() << " "
             << "tensor_meta = " << test_tensor->meta()
             << "tensor_data_addr:" << test_tensor->data() << std::endl;
        } else {
          ss << "NOT_INITED_TENSOR or NonTensor" << std::endl;
        }
      } else {
        ss << "NOT_INITED_VAR " << std::endl;
      }
      VLOG(10) << ss.str();
    }
  }
  return;
}

template <typename VarType>
std::shared_ptr<NameVarMap<VarType>> DebugPrepareData(
    const framework::OperatorWithKernel& op,
    const NameVarMap<VarType>& ins,
    const framework::OpKernelType& expected_kernel_key) {
  std::shared_ptr<NameVarMap<VarType>> tmp_ins_ptr = nullptr;
  for (const auto& name_pair : ins) {
    for (size_t i = 0; i < name_pair.second.size(); ++i) {
      auto& template_var = name_pair.second[i];
      SetForwardDataTypeOfGradVar(template_var);
      const auto* tensor = GetTensorFromVar(template_var->Var());
      if (tensor && tensor->IsInitialized() && (tensor->memory_size() != 0)) {
        auto kernel_type_for_var = op.GetKernelTypeForVar(
            name_pair.first, *tensor, expected_kernel_key);
        if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
          continue;
        } else {
          VLOG(3) << "Transform Variable " << GetNameFromVar(template_var)
                  << " from " << kernel_type_for_var << " to "
                  << expected_kernel_key;
          VLOG(3) << GetNameFromVar(template_var)
                  << " memory size is: " << tensor->memory_size();
          if (CheckCachedKey(template_var, expected_kernel_key)) {
            VLOG(3) << "Hit variable_wrapper cache: key="
                    << expected_kernel_key;
            std::shared_ptr<VariableWrapper> cache_var =
                GetCachedValue(template_var, expected_kernel_key);
            if (tmp_ins_ptr == nullptr) {
              tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
            }

            const auto* tensor = GetTensorFromVar(cache_var->Var());
            const auto* debug_tensor = GetDebugTensorFromVar(cache_var->Var());
            if (debug_tensor && debug_tensor->IsInitialized() &&
                (debug_tensor->memory_size() != 0)) {
              // tensor = debug_tensor;
            } else {
              paddle::framework::SetVoidVariableDebug(cache_var->MutableVar());
              debug_tensor = GetDebugTensorFromVar(cache_var->Var());
              paddle::framework::TransformData(
                  *tensor,
                  expected_kernel_key.place_,
                  const_cast<framework::Tensor*>(debug_tensor));
              // tensor = debug_tensor;
            }
            auto tmp_var =
                std::make_shared<VarType>(GetNameFromVar(template_var));
            SetType(tmp_var, GetType(template_var));
            SetTensorToVariable(
                cache_var->Var(), *debug_tensor, tmp_var->MutableVar());
            (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
          } else {
            framework::Tensor out;
            TransformData(
                expected_kernel_key, kernel_type_for_var, *tensor, &out);
            if (NeedTransformDataType(kernel_type_for_var,
                                      expected_kernel_key)) {
              if (tmp_ins_ptr == nullptr) {
                tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
              }
              auto tmp_var =
                  std::make_shared<VarType>(GetNameFromVar(template_var));
              SetType(tmp_var, GetType(template_var));
              SetTensorToVariable(
                  template_var->Var(), out, tmp_var->MutableVar());
              (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
              SetCachedValue(template_var, expected_kernel_key, tmp_var);
              VLOG(3) << "Set cache to variable_wrapper: key="
                      << expected_kernel_key;
            } else {
              SetTensorToVariable(
                  template_var->Var(), out, template_var->MutableVar());
            }
          }
        }
      }
    }
  }
  return tmp_ins_ptr;
}

template <typename VarType>
std::shared_ptr<NameVarMap<VarType>> PrepareData(
    const framework::OperatorWithKernel& op,
    const NameVarMap<VarType>& ins,
    const phi::KernelKey& expected_kernel_key,
    const phi::Place& place) {
  std::shared_ptr<NameVarMap<VarType>> tmp_ins_ptr = nullptr;
  for (const auto& name_pair : ins) {
    for (size_t i = 0; i < name_pair.second.size(); ++i) {
      auto& template_var = name_pair.second[i];
      SetForwardDataTypeOfGradVar(template_var);
      const auto* tensor = GetTensorFromVar(template_var->Var());
      if (tensor && tensor->IsInitialized() && (tensor->memory_size() != 0)) {
        auto kernel_type_for_var = op.GetKernelTypeForVar(
            name_pair.first, *tensor, expected_kernel_key);
        if (!framework::NeedTransform(kernel_type_for_var,
                                      expected_kernel_key)) {
          continue;
        } else {
          VLOG(3) << "Transform Variable " << GetNameFromVar(template_var)
                  << " from " << kernel_type_for_var << " to "
                  << expected_kernel_key;
          VLOG(3) << GetNameFromVar(template_var)
                  << " memory size is: " << tensor->memory_size();
          if (CheckCachedKey(template_var, expected_kernel_key)) {
            VLOG(3) << "Hit variable_wrapper cache: key="
                    << expected_kernel_key;
            std::shared_ptr<VariableWrapper> cache_var =
                GetCachedValue(template_var, expected_kernel_key);
            if (tmp_ins_ptr == nullptr) {
              tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
            }

            const auto* tensor = GetTensorFromVar(cache_var->Var());
            auto tmp_var =
                std::make_shared<VarType>(GetNameFromVar(template_var));
            SetType(tmp_var, GetType(template_var));
            SetTensorToVariable(
                cache_var->Var(), *tensor, tmp_var->MutableVar());
            (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
          } else {
            phi::DenseTensor out;
            framework::TransformData(
                expected_kernel_key, kernel_type_for_var, *tensor, &out, place);
            if (framework::NeedTransformDataType(kernel_type_for_var,
                                                 expected_kernel_key)) {
              // To avoid NameVarMap copy construction overhead in general
              // scenarios, if inplace transformed, return original input
              // directly
              if (tmp_ins_ptr == nullptr) {
                tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
              }
              auto tmp_var =
                  std::make_shared<VarType>(GetNameFromVar(template_var));
              SetType(tmp_var, GetType(template_var));
              SetTensorToVariable(
                  template_var->Var(), out, tmp_var->MutableVar());
              (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
              SetCachedValue(template_var, expected_kernel_key, tmp_var);
              VLOG(3) << "Set cache to variable_wrapper: key="
                      << expected_kernel_key;
            } else {
              // if dtype is same, transform inplace will not change the
              // original
              // value, transform inplace to avoid multiple copy
              SetTensorToVariable(
                  template_var->Var(), out, template_var->MutableVar());
            }
          }
        }
      }
    }
  }
  return tmp_ins_ptr;
}

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const phi::KernelKey& kernel_key,
             const framework::OperatorWithKernel::OpKernelFunc& func,
             const phi::ArgumentMappingFn* arg_map_fn,
             const phi::KernelSignature* default_kernel_signature,
             platform::DeviceContext* dev_ctx);

  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const phi::KernelKey& kernel_key,
             const phi::ArgumentMappingFn* arg_map_fn,
             const phi::KernelSignature* default_kernel_signature,
             phi::KernelSignature&& kernel_signature,
             const phi::Kernel& phi_kernel,
             platform::DeviceContext* dev_ctx);

  static PreparedOp Prepare(const NameVarMap<VarBase>& ins,
                            const NameVarMap<VarBase>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs,
                            const framework::AttributeMap& default_attrs);

  static PreparedOp Prepare(const NameVarMap<VariableWrapper>& ins,
                            const NameVarMap<VariableWrapper>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs,
                            const framework::AttributeMap& default_attrs);

  static PreparedOp Prepare(const NameVarMap<egr::EagerVariable>& ins,
                            const NameVarMap<egr::EagerVariable>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs,
                            const framework::AttributeMap& default_attrs);

  void Run(const NameVarMap<VarBase>& in,
           const NameVarMap<VarBase>& out,
           const framework::AttributeMap& attrs,
           const framework::AttributeMap& default_attrs);

  void Run(const NameVarMap<VariableWrapper>& ins,
           const NameVarMap<VariableWrapper>& outs,
           const framework::AttributeMap& attrs,
           const framework::AttributeMap& default_attrs);

  void Run(const NameVarMap<egr::EagerVariable>& ins,
           const NameVarMap<egr::EagerVariable>& outs,
           const framework::AttributeMap& attrs,
           const framework::AttributeMap& default_attrs);

  const phi::KernelKey& kernel_key() const { return kernel_key_; }

  const phi::Place& place() const { return dev_ctx_->GetPlace(); }

 private:
  const framework::OperatorBase& op_;
  const framework::RuntimeContext& ctx_;
  phi::KernelKey kernel_key_;
  framework::OperatorWithKernel::OpKernelFunc func_;
  platform::DeviceContext* dev_ctx_;
  // NOTE(chenweihang): Similar op members are used to adapt to
  // new phi kernel, if there is a better design in the future,
  // we may polish the implementation here
  bool run_phi_kernel_{false};
  bool run_kp_kernel_{false};
  const phi::ArgumentMappingFn* arg_map_fn_;
  const phi::KernelSignature* default_kernel_signature_;
  phi::KernelSignature kernel_signature_;
  const phi::Kernel& phi_kernel_;

  static const phi::KernelFactory& phi_kernel_factory;
  static const phi::OpUtilsMap& phi_op_utils_map;
  static const phi::DefaultKernelSignatureMap& default_phi_kernel_sig_map;
};

const inline framework::Attribute* GetAttr(
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    const std::string& name) {
  auto it = attrs.find(name);
  bool found = it != attrs.end();
  if (!found) {
    it = default_attrs.find(name);
    found = it != default_attrs.end();
  }
  if (found) {
    return &it->second;
  }
  return nullptr;
}

template <typename VarType>
void BuildDygraphPhiKernelContext(const phi::KernelSignature& kernel_signature,
                                  const phi::Kernel& phi_kernel,
                                  const NameVarMap<VarType>& ins,
                                  const NameVarMap<VarType>& outs,
                                  const framework::AttributeMap& attrs,
                                  const framework::AttributeMap& default_attrs,
                                  platform::DeviceContext* dev_ctx,
                                  phi::KernelContext* kernel_ctx) {
  kernel_ctx->SetDeviceContext(dev_ctx);

  const auto& input_names = kernel_signature.input_names;
  const auto& attr_names = kernel_signature.attr_names;
  const auto& output_names = kernel_signature.output_names;

  auto& input_defs = phi_kernel.args_def().input_defs();
  auto& output_defs = phi_kernel.args_def().output_defs();
  auto& attr_defs = phi_kernel.args_def().attribute_defs();

  PADDLE_ENFORCE_EQ(
      input_names.size(),
      input_defs.size(),
      platform::errors::InvalidArgument(
          "Op %s: the size of inputs_args names (%d) must be equal to "
          "the size of kernel input_defs (%d).",
          kernel_signature.name,
          input_names.size(),
          input_defs.size()));

  PADDLE_ENFORCE_EQ(
      output_names.size(),
      output_defs.size(),
      platform::errors::InvalidArgument(
          "Op %s: the size of outputs_args names (%d) must be equal to "
          "the size of kernel output_defs (%d).",
          kernel_signature.name,
          output_names.size(),
          output_defs.size()));

  PADDLE_ENFORCE_EQ(
      attr_names.size(),
      attr_defs.size(),
      platform::errors::InvalidArgument(
          "Op %s: the size of attribute_args names (%d) must be equal "
          "to the size of kernel attribute_defs (%d).",
          kernel_signature.name,
          attr_names.size(),
          attr_defs.size()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto it = ins.find(input_names[i]);

    size_t start_idx = (i == 0 ? 0 : kernel_ctx->InputRangeAt(i - 1).second);

    if (it == ins.end()) {
      if (LIKELY(input_defs[i].type_index ==
                 std::type_index(typeid(paddle::optional<phi::DenseTensor>)))) {
        kernel_ctx->EmplaceBackInputWithoutSetRange(nullptr);
        auto end_idx = start_idx + 1;
        kernel_ctx->AssignInputRange(std::make_pair(start_idx, end_idx), i);
        continue;
      } else if (input_defs[i].type_index ==
                 std::type_index(typeid(
                     paddle::optional<std::vector<const phi::DenseTensor*>>))) {
        kernel_ctx->EmplaceBackInputWithoutSetRange(nullptr);
        auto end_idx = start_idx + 1;
        kernel_ctx->AssignInputRange(std::make_pair(start_idx, end_idx), i);
        continue;
      } else {
        PADDLE_THROW(phi::errors::NotFound(
            "Can not find input variable '%s' for %s OP, please check whether "
            "the name setting in OpArgumentMapping is consistent with that in "
            "OpMaker.",
            input_names[i],
            kernel_signature.name));
      }
    }

    auto& ins_vector = it->second;
    size_t end_idx = start_idx + ins_vector.size();

    for (size_t offset = 0; offset < ins_vector.size(); ++offset) {
      const phi::TensorBase* tensor_in = nullptr;
      auto& var = ins_vector[offset]->Var();
      if (var.template IsType<phi::DenseTensor>()) {
        tensor_in = &(var.template Get<phi::DenseTensor>());
        kernel_ctx->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var.template IsType<phi::SelectedRows>()) {
        tensor_in = &(var.template Get<phi::SelectedRows>());
        kernel_ctx->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var.template IsType<framework::LoDTensorArray>()) {
        tensor_in = &(var.template Get<framework::LoDTensorArray>());
        kernel_ctx->EmplaceBackInputWithoutSetRange(tensor_in);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported input `%s` type when call pt kernel.",
            framework::ToTypeName(var.Type())));
      }
    }
    kernel_ctx->AssignInputRange(std::make_pair(start_idx, end_idx), i);
  }
  VLOG(6) << "BuildDygraphPhiKernelContext: Inputs parsing completed.";

  for (size_t i = 0; i < output_names.size(); ++i) {
    size_t start_idx = (i == 0 ? 0 : kernel_ctx->OutputRangeAt(i - 1).second);

    auto iter = outs.find(output_names[i]);
    if (iter == outs.end()) {
      kernel_ctx->EmplaceBackOutputWithoutSetRange(nullptr);
      kernel_ctx->AssignOutputRange(std::make_pair(start_idx, start_idx + 1),
                                    i);
      continue;
    }

    auto& outs_vector = iter->second;
    size_t end_idx = start_idx + outs_vector.size();

    for (size_t offset = 0; offset < outs_vector.size(); ++offset) {
      if (outs_vector[offset] == nullptr) {
        kernel_ctx->EmplaceBackOutputWithoutSetRange(nullptr);
        continue;
      }

      phi::TensorBase* tensor_out = nullptr;
      auto* var = outs_vector[offset]->MutableVar();
      if (var) {
        if (var->template IsType<phi::DenseTensor>()) {
          tensor_out = var->template GetMutable<phi::DenseTensor>();
          kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<phi::SelectedRows>()) {
          tensor_out = var->template GetMutable<phi::SelectedRows>();
          kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<framework::LoDTensorArray>()) {
          tensor_out = var->template GetMutable<framework::LoDTensorArray>();
          kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported output `%s` type when call pt kernel.",
              framework::ToTypeName(var->Type())));
        }
      } else {
        kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
      }
    }
    kernel_ctx->AssignOutputRange(std::make_pair(start_idx, end_idx), i);
  }
  VLOG(6) << "BuildDygraphPhiKernelContext: Outputs parsing completed.";

  for (size_t i = 0; i < attr_names.size(); ++i) {
    VLOG(6) << "BuildDygraphPhiKernelContext: " << attr_names[i] << ": "
            << attr_defs[i].type_index;
    auto* attr_ptr = GetAttr(attrs, default_attrs, attr_names[i]);
    switch (attr_defs[i].type_index) {
      case phi::AttributeType::SCALAR:
        if (attr_ptr) {
          // scalar is in the attribute
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::FLOAT:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(float, attr))));
              break;
            case framework::proto::AttrType::FLOAT64:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(double, attr))));
              break;
            case framework::proto::AttrType::INT:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(int, attr))));
              break;
            case framework::proto::AttrType::LONG:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(int64_t, attr))));
              break;
            case framework::proto::AttrType::STRING:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(std::string, attr))));
              break;
            case framework::proto::AttrType::BOOLEAN:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(bool, attr))));
              break;
            default:
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to Scalar when construct "
                  "KernelContext in dygraph.",
                  attr_names[i]));
          }
        } else {  // scalar is in the input
          auto& ins_vector = ins.at(attr_names[i]);
          kernel_ctx->EmplaceBackAttr(std::move(
              experimental::MakePhiScalarFromVar(ins_vector[0]->Var())));
        }
        break;
      case phi::AttributeType::INT_ARRAY:
        if (attr_ptr) {
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::INTS:
              kernel_ctx->EmplaceBackAttr(std::move(
                  phi::IntArray(PADDLE_GET_CONST(std::vector<int32_t>, attr))));
              break;
            case framework::proto::AttrType::LONGS:
              kernel_ctx->EmplaceBackAttr(std::move(
                  phi::IntArray(PADDLE_GET_CONST(std::vector<int64_t>, attr))));
              break;
            case framework::proto::AttrType::INT:
              kernel_ctx->EmplaceBackAttr(std::move(
                  phi::IntArray(&PADDLE_GET_CONST(int32_t, attr), 1)));
              break;
            case framework::proto::AttrType::LONG:
              kernel_ctx->EmplaceBackAttr(std::move(
                  phi::IntArray(&PADDLE_GET_CONST(int64_t, attr), 1)));
              break;
            default:
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to IntArray when "
                  "construct KernelContext.",
                  attr_names[i]));
          }
        } else {  // shape is in the input
          auto& ins_vector = ins.at(attr_names[i]);
          if (ins_vector.size() == 1) {  // ShapeTensor
            kernel_ctx->EmplaceBackAttr(std::move(
                experimental::MakePhiIntArrayFromVar(ins_vector[0]->Var())));
          } else {  // ShapeTensorList
            std::vector<framework::Variable*> variables;
            variables.reserve(ins_vector.size());
            for (const auto& var_base : ins_vector) {
              variables.push_back(var_base->MutableVar());
            }
            kernel_ctx->EmplaceBackAttr(
                std::move(experimental::MakePhiIntArrayFromVarList(variables)));
          }
        }
        break;
      case phi::AttributeType::SCALARS: {
        PADDLE_ENFORCE_NOT_NULL(
            attr_ptr,
            platform::errors::NotFound("(%s) is not found in AttributeMap when "
                                       "buildind dygraph KernelContext.",
                                       attr_names[i]));
        auto& attr = *attr_ptr;
        switch (AttrTypeID(attr)) {
          case framework::proto::AttrType::INTS: {
            const auto& vec = PADDLE_GET_CONST(std::vector<int32_t>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case framework::proto::AttrType::LONGS: {
            const auto& vec = PADDLE_GET_CONST(std::vector<int64_t>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case framework::proto::AttrType::FLOATS: {
            const auto& vec = PADDLE_GET_CONST(std::vector<float>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case framework::proto::AttrType::FLOAT64S: {
            const auto& vec = PADDLE_GET_CONST(std::vector<double>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case framework::proto::AttrType::BOOLEANS: {
            const auto& vec = PADDLE_GET_CONST(std::vector<bool>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          default:
            PADDLE_THROW(platform::errors::Unimplemented(
                "Unsupported cast op attribute `%s` to vector<Scalar> when "
                "construct KernelContext.",
                attr_names[i]));
        }
      } break;
      default: {
        PADDLE_ENFORCE_NOT_NULL(
            attr_ptr,
            platform::errors::NotFound("(%s) is not found in AttributeMap when "
                                       "buildind dygraph KernelContext.",
                                       attr_names[i]));
        auto& attr = *attr_ptr;
        switch (attr_defs[i].type_index) {
          case phi::AttributeType::FLOAT32:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(float, attr));
            break;
          case phi::AttributeType::FLOAT64:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(double, attr));
            break;
          case phi::AttributeType::INT32:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(int, attr));
            break;
          case phi::AttributeType::BOOL:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(bool, attr));
            break;
          case phi::AttributeType::INT64:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(int64_t, attr));
            break;
          case phi::AttributeType::INT32S:
            kernel_ctx->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<int>, attr));
            break;
          case phi::AttributeType::DATA_TYPE: {
            auto data_type = framework::TransToPhiDataType(
                static_cast<framework::proto::VarType::Type>(
                    PADDLE_GET_CONST(int, attr)));
            kernel_ctx->EmplaceBackAttr(data_type);
          } break;
          case phi::AttributeType::STRING:
            kernel_ctx->EmplaceBackAttr(
                std::move(PADDLE_GET_CONST(std::string, attr)));
            break;
          case phi::AttributeType::INT64S: {
            switch (AttrTypeID(attr)) {
              case framework::proto::AttrType::LONGS:
                kernel_ctx->EmplaceBackAttr(
                    PADDLE_GET_CONST(std::vector<int64_t>, attr));
                break;
              case framework::proto::AttrType::INTS: {
                const auto& vector_int_attr =
                    PADDLE_GET_CONST(std::vector<int>, attr);
                const std::vector<int64_t> vector_int64_attr(
                    vector_int_attr.begin(), vector_int_attr.end());
                kernel_ctx->EmplaceBackAttr(vector_int64_attr);
              } break;
              default:
                PADDLE_THROW(platform::errors::Unimplemented(
                    "Unsupported cast op attribute `%s` to vector<int64_t> "
                    "when "
                    "construct KernelContext.",
                    attr_names[i]));
            }
          } break;
          case phi::AttributeType::FLOAT32S:
            kernel_ctx->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<float>, attr));
            break;
          case phi::AttributeType::STRINGS:
            kernel_ctx->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<std::string>, attr));
            break;
          default:
            PADDLE_THROW(platform::errors::Unimplemented(
                "Unsupported cast op attribute `%s` when construct "
                "KernelContext in dygraph.",
                attr_names[i]));
        }
      }
    }
  }
  VLOG(6) << "BuildDygraphPhiKernelContext: Attributes parsing completed.";
}

template <typename VarType>
void PreparePhiData(const phi::Kernel& phi_kernel,
                    const phi::KernelSignature& kernel_signature,
                    const NameVarMap<VarType>& ins) {
  const auto& input_names = kernel_signature.input_names;
  auto& input_defs = phi_kernel.args_def().input_defs();

  PADDLE_ENFORCE_EQ(input_names.size(),
                    input_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of inputs_args names (%d) must be equal to "
                        "the size of kernel input_defs (%d).",
                        input_names.size(),
                        input_defs.size()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto& in_def = input_defs.at(i);
    auto iter = ins.find(input_names[i]);
    if (iter == ins.end()) {
      continue;
    }
    auto& ins_vector = iter->second;

    for (size_t offset = 0; offset < ins_vector.size(); ++offset) {
      auto& var = ins_vector[offset];
      const auto* tensor_in = GetTensorFromVar(var->Var());
      if (tensor_in && tensor_in->IsInitialized() &&
          (tensor_in->memory_size() != 0)) {
        if (in_def.backend == phi::Backend::ALL_BACKEND) {
          continue;
        }
        auto tensor_backend = phi::TransToPhiBackend(tensor_in->place());
        if (in_def.backend == tensor_backend ||
            (in_def.backend == phi::Backend::GPUDNN &&
             tensor_backend == phi::Backend::GPU)) {
          continue;
        }

        auto expected_place = phi::TransToPhiPlace(in_def.backend);

        VLOG(3) << "Phi Transform Variable " << input_names[i] << " from "
                << tensor_in->place() << " to " << expected_place;

        phi::DenseTensor tmp_tensor;
        framework::TensorCopySync(*tensor_in, expected_place, &tmp_tensor);

        SetTensorToVariable(var->Var(), tmp_tensor, var->MutableVar());
      }
    }
  }
}

}  // namespace imperative
}  // namespace paddle
