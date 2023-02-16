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

#include "paddle/fluid/imperative/layer.h"

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/infer_var_type_context.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/imperative/prepared_operator.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

DECLARE_bool(use_mkldnn);
namespace paddle {
namespace imperative {

struct timeval t1;
struct timeval t2;
static int64_t op_step_id = 0;

using framework::Variable;
void ThreadSafeNameSet::Insert(const std::string& name) {
  std::lock_guard<std::mutex> guard(mtx_);
  set_.insert(name);
}

void ThreadSafeNameSet::Remove(const std::string& name) {
  std::lock_guard<std::mutex> guard(mtx_);
  auto iter = set_.find(name);
  PADDLE_ENFORCE_EQ(
      iter != set_.end(),
      true,
      platform::errors::NotFound("Variable name %s does not exist", name));
  set_.erase(iter);
}

std::vector<std::string> ThreadSafeNameSet::Names() const {
  std::lock_guard<std::mutex> guard(mtx_);
  return std::vector<std::string>(set_.begin(), set_.end());
}

ThreadSafeNameSet VarBase::name_set_;

std::vector<std::string> VarBase::AliveVarNames() { return name_set_.Names(); }

static framework::RuntimeContext PrepareRuntimeContext(
    const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  framework::VariableValueMap inputs, outputs;
  for (auto& in_pair : ins) {
    auto& in_ctx = inputs[in_pair.first];
    in_ctx.reserve(in_pair.second.size());
    for (auto& in_var : in_pair.second) {
      in_ctx.emplace_back(in_var->MutableVar());
    }
  }

  for (auto& out_pair : outs) {
    auto& out_ctx = outputs[out_pair.first];
    out_ctx.reserve(out_pair.second.size());
    for (auto& out_var : out_pair.second) {
      out_ctx.emplace_back(out_var->MutableVar());
    }
  }
  return framework::RuntimeContext(std::move(inputs), std::move(outputs));
}

phi::Place& xpu_debug_run_dev2() {
  static phi::Place dev2 = phi::CPUPlace();
  static bool inited = false;
  static std::string device = "CPU";
  if (!inited) {
    if (std::getenv("XPU_PADDLE_DEBUG_RUN_DEV2") != nullptr) {
      std::string ops(std::getenv("XPU_PADDLE_DEBUG_RUN_DEV2"));
      if (ops == "1") {
        dev2 = phi::XPUPlace();
        device = "XPU";
      }
    }
    inited = true;
    VLOG(3) << "XPU Paddle Debug Run Dev2: " << device;
  }
  return dev2;
}

bool ContinueOrNot(const std::string& op_type) {
  bool continue_or_not =
      !phi::backends::xpu::is_in_xpu_debug_black_list(op_type) &&
      !phi::backends::xpu::is_in_xpu_debug_black_id_list(
          std::to_string(op_step_id));
  continue_or_not = continue_or_not &&
                    (phi::backends::xpu::is_in_xpu_debug_white_list(op_type) ||
                     std::getenv("XPU_PADDLE_DEBUG_WHITE_LIST") == nullptr);
  continue_or_not = continue_or_not &&
                    (phi::backends::xpu::is_in_xpu_debug_white_id_list(
                         std::to_string(op_step_id)) ||
                     std::getenv("XPU_PADDLE_DEBUG_WHITE_ID_LIST") == nullptr);
  return continue_or_not;
}

bool ContinueRunDev2OrNot(const std::string& op_type) {
  bool continue_or_not =
      !phi::backends::xpu::is_in_xpu_debug_run_dev2_black_list(op_type);
  return continue_or_not;
}

bool DebugOrNot() {
  bool continue_or_not = (std::getenv("XPU_PADDLE_DEBUG_GLOBAL") != nullptr ||
                          std::getenv("XPU_PADDLE_DEBUG_OP") != nullptr);
  return continue_or_not;
}

static std::string XPUDebugStartString(const std::string& op_type,
                                       const PreparedOp& prepared_op) {
  if (ContinueOrNot(op_type)) {
    std::stringstream print_buffer;
    print_buffer << "op_name_debug " << op_type << " " << op_step_id << " "
                 << prepared_op.place() << " "
                 << prepared_op.kernel_key().dtype() << " in: ";
    return print_buffer.str();
  } else {
    return "";
  }
}

template <typename VarType>
static std::string XPUDebugStringImpl(const std::string& op_type,
                                      const std::string& debug_str,
                                      const NameVarMap<VarType>& ins,
                                      const NameVarMap<VarType>& ins_dev2,
                                      const phi::Place& place,
                                      const phi::Place& place_dev2) {
  std::stringstream print_buffer;
  print_buffer << debug_str;
  if (platform::is_xpu_place(place) || platform::is_xpu_place(place_dev2)) {
    int r = xpu_wait();
    PADDLE_ENFORCE_EQ(
        r, 0, platform::errors::InvalidArgument("not initialized.[", op_type));
  }
  for (auto& pair : ins) {
    for (size_t i = 0; i < pair.second.size(); i++) {
      if (pair.second[i] == nullptr) continue;
      VLOG(10) << pair.first << "-" << GetNameFromVar(pair.second[i]);
      print_buffer << pair.first << "-" << GetNameFromVar(pair.second[i])
                   << "-";
      const framework::Variable& var = pair.second[i]->Var();
      const framework::Variable& var_dev2 = ins_dev2.at(pair.first)[i]->Var();
      if (!var.IsInitialized()) {
        print_buffer << "NOT_INITED_VAR ";
        // } else if (var.IsType<phi::DenseTensor>()) {
        //   auto& tensor = var.Get<phi::DenseTensor>;
        //   auto& tensor_dev2 = var_dev2.Get<phi::DenseTensor>;
        //   if (tensor.IsInitialized()) {
        //     print_buffer << tensor.dtype() << "-" << tensor.place() << "-"
        //                  << tensor_dev2.place() << " "
        //                  << tensor.check_mse(tensor_dev2) << " ";
        //     //  << tensor_dev2.check_sum() << " ";
        //   } else {
        //     print_buffer << tensor.dtype() << " "
        //                  << "NOT_INITED ";
        //   }
        // } else {
        //   print_buffer << "None NonTensor ";
        // }
      } else {
        const auto* tensor = GetTensorFromVar(var);
        const auto* tensor_dev2 = GetTensorFromVar(var_dev2);
        if (tensor && tensor->IsInitialized()) {
          print_buffer << tensor->dtype() << "-" << tensor->place() << "-"
                       << tensor_dev2->place() << " "
                       << tensor->check_mse(*tensor_dev2) << " ";
          //  << tensor_dev2.check_sum() << " ";
        } else {
          print_buffer << tensor->dtype() << " "
                       << "NOT_INITED ";
        }
      }
    }
  }
  return print_buffer.str();
}

std::string XPUDebugString(const std::string& op_type,
                           const std::string& debug_str,
                           const NameVarMap<VarBase>& ins,
                           const NameVarMap<VarBase>& ins_dev2,
                           const phi::Place& place,
                           const phi::Place& place_dev2 = phi::CPUPlace()) {
  if (ContinueOrNot(op_type)) {
    return XPUDebugStringImpl<VarBase>(
        op_type, debug_str, ins, ins_dev2, place, place_dev2);
  } else {
    return "";
  }
}

std::string XPUDebugString(const std::string& op_type,
                           const std::string& debug_str,
                           const NameVarMap<VariableWrapper>& ins,
                           const NameVarMap<VariableWrapper>& ins_dev2,
                           const phi::Place& place,
                           const phi::Place& place_dev2 = phi::CPUPlace()) {
  if (ContinueOrNot(op_type)) {
    return XPUDebugStringImpl<VariableWrapper>(
        op_type, debug_str, ins, ins_dev2, place, place_dev2);
  } else {
    return "";
  }
}

std::string XPUDebugString(const std::string& op_type,
                           const std::string& debug_str,
                           const NameVarMap<egr::EagerVariable>& ins,
                           const NameVarMap<egr::EagerVariable>& ins_dev2,
                           const phi::Place& place,
                           const phi::Place& place_dev2 = phi::CPUPlace()) {
  if (ContinueOrNot(op_type)) {
    return XPUDebugStringImpl<egr::EagerVariable>(
        op_type, debug_str, ins, ins_dev2, place, place_dev2);
  } else {
    return "";
  }
}

static void XPUPaddleOpTimeTik() {
  // struct timeval t1;
  // struct timeval t2;
  gettimeofday(&t1, NULL);
}

static void XPUPaddleOpTimeTok(const framework::OperatorBase& op,
                               const PreparedOp prepared_op,
                               const platform::Place& place) {
  // 耗时统计逻辑
  if (platform::is_xpu_place(place)) {
    int r = xpu_wait();
    PADDLE_ENFORCE_EQ(
        r,
        0,
        platform::errors::InvalidArgument("not initialized.[", op.Type()));
  }
  gettimeofday(&t2, NULL);
  uint32_t diff = 1000000 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec;
  std::cout << "op_name " << op.Type() << " " << diff << " "
            << prepared_op.place() << " " << prepared_op.kernel_key().dtype()
            << std::endl;
}

static std::string XPUDebugDumpDataStart(const std::string& op_type,
                                         const PreparedOp& prepared_op,
                                         const std::string& debug_str = "") {
  if (ContinueOrNot(op_type)) {
    std::stringstream dump_file_dir_stream;
    const char* path_char = std::getenv("XPU_PADDLE_DUMP_DATA_PATH");
    std::string palce_str;
    if (platform::is_cpu_place(prepared_op.place())) {
      palce_str = "CPU";
    } else if (platform::is_xpu_place(prepared_op.place())) {
      palce_str = "XPU";
    } else {
      palce_str = "Unknow";
    }
    if (path_char != nullptr) {
      std::string path_str = path_char;
      dump_file_dir_stream << path_str << "/"
                           << "dump_data-" << op_type << "-" << op_step_id
                           << "-" << palce_str << "-"
                           << prepared_op.kernel_key().dtype() << debug_str
                           << ".txt";
    }
    return dump_file_dir_stream.str();
  } else {
    return "";
  }
}

template <typename VarType>
static void XPUDebugDumpDataImpl(const std::string& op_type,
                                 const std::string& dump_data_path,
                                 const std::string& debug_str,
                                 const NameVarMap<VarType>& ins,
                                 const platform::Place& place) {
  std::ofstream ofs(dump_data_path, std::ios::app);
  ofs << debug_str << ":\n";
  ofs.close();

  if (platform::is_xpu_place(place)) {
    int r = xpu_wait();
    PADDLE_ENFORCE_EQ(
        r, 0, platform::errors::InvalidArgument("not initialized.[", op_type));
  }
  for (auto& pair : ins) {
    for (size_t i = 0; i < pair.second.size(); i++) {
      if (pair.second[i] == nullptr) continue;
      const framework::Variable& var = pair.second[i]->Var();
      if (!var.IsInitialized()) {
        // } else if (var.IsType<phi::DenseTensor>()) {
        //   auto& tensor = var.Get<phi::DenseTensor>();
        //   if (tensor.IsInitialized()) {
        //     std::ofstream ofs(dump_data_path, std::ios::app);
        //     ofs.precision(12);
        //     ofs << pair.first << "--" << GetNameFromVar(pair.second[i]) << ":
        //     "; ofs << tensor << "\n"; ofs.close();
        //   }
      } else {
        const auto* tensor = GetTensorFromVar(var);
        if (tensor && tensor->IsInitialized()) {
          std::ofstream ofs(dump_data_path, std::ios::app);
          ofs.precision(12);
          ofs << pair.first << "--" << GetNameFromVar(pair.second[i]) << ": ";
          ofs << *tensor << "\n";
          ofs.close();
        }
      }
    }
  }
  return;
}

void XPUDebugDumpData(const std::string& op_type,
                      const std::string& dump_data_path,
                      const std::string& debug_str,
                      const NameVarMap<VarBase>& ins,
                      const platform::Place& place) {
  if (ContinueOrNot(op_type)) {
    XPUDebugDumpDataImpl<VarBase>(
        op_type, dump_data_path, debug_str, ins, place);
  }
}

void XPUDebugDumpData(const std::string& op_type,
                      const std::string& dump_data_path,
                      const std::string& debug_str,
                      const NameVarMap<VariableWrapper>& ins,
                      const platform::Place& place) {
  if (ContinueOrNot(op_type)) {
    XPUDebugDumpDataImpl<VariableWrapper>(
        op_type, dump_data_path, debug_str, ins, place);
  }
}

void XPUDebugDumpData(const std::string& op_type,
                      const std::string& dump_data_path,
                      const std::string& debug_str,
                      const NameVarMap<egr::EagerVariable>& ins,
                      const platform::Place& place) {
  if (ContinueOrNot(op_type)) {
    XPUDebugDumpDataImpl<egr::EagerVariable>(
        op_type, dump_data_path, debug_str, ins, place);
  }
}

template <typename VarType>
static std::string DebugString(
    const std::string& name,
    const std::vector<std::shared_ptr<VarType>>& vars) {
  std::stringstream ss;
  ss << name << "{";

  for (size_t i = 0; i < vars.size(); ++i) {
    if (i > 0) ss << ", ";

    if (vars[i] == nullptr) {
      ss << "NULL";
      continue;
    }
    ss << GetNameFromVar(vars[i]) << "[";
    const framework::Variable& var = vars[i]->Var();
    if (!var.IsInitialized()) {
      ss << "NOT_INITED_VAR";
    } else if (var.IsType<phi::DenseTensor>()) {
      auto& tensor = var.Get<phi::DenseTensor>();
      ss << "DenseTensor<";
      if (tensor.IsInitialized()) {
        ss << framework::DataTypeToString(
                  framework::TransToProtoVarType(tensor.dtype()))
           << ", ";
        ss << tensor.place() << ", ";
        ss << "(" << tensor.dims() << ")";
      } else {
        ss << "NOT_INITED";
      }
      ss << ">";
    } else if (var.IsType<phi::SelectedRows>()) {
      ss << "SelectedRows<";
      auto& selected_rows = var.Get<phi::SelectedRows>();
      auto& tensor = selected_rows.value();
      auto& rows = selected_rows.rows();
      if (tensor.IsInitialized()) {
        ss << framework::DataTypeToString(
                  framework::TransToProtoVarType(tensor.dtype()))
           << ", ";
        ss << tensor.place() << ", ";
        ss << "height(" << selected_rows.height() << "), rows(";
        std::for_each(rows.cbegin(), rows.cend(), [&ss](const int64_t r) {
          ss << r << " ";
        });
        ss << "), dims(" << tensor.dims() << ")";
      } else {
        ss << "NOT_INITED";
      }
      ss << ">";
    } else {
      ss << "UNRESOLVED_TYPE";
    }
    ss << "]";
  }

  ss << "}";
  return ss.str();
}

template <typename VarType>
static std::string LayerDebugStringImpl(const std::string& op_type,
                                        const NameVarMap<VarType>& ins,
                                        const NameVarMap<VarType>& outs) {
  std::stringstream ss;
  ss << "Op(" << op_type << "): ";

  ss << "Inputs: ";

  size_t i = 0;
  for (auto& pair : ins) {
    if (i > 0) ss << ", ";
    ss << DebugString<VarType>(pair.first, pair.second);
    ++i;
  }

  ss << ",   Outputs: ";
  i = 0;
  for (auto& pair : outs) {
    if (i > 0) ss << ", ";
    ss << DebugString<VarType>(pair.first, pair.second);
    ++i;
  }
  return ss.str();
}

std::string LayerDebugString(const std::string& op_type,
                             const NameVarMap<VarBase>& ins,
                             const NameVarMap<VarBase>& outs) {
  return LayerDebugStringImpl<VarBase>(op_type, ins, outs);
}

std::string LayerDebugString(const std::string& op_type,
                             const NameVarMap<VariableWrapper>& ins,
                             const NameVarMap<VariableWrapper>& outs) {
  return LayerDebugStringImpl<VariableWrapper>(op_type, ins, outs);
}

std::string LayerDebugString(const std::string& op_type,
                             const NameVarMap<egr::EagerVariable>& ins,
                             const NameVarMap<egr::EagerVariable>& outs) {
  return LayerDebugStringImpl<egr::EagerVariable>(op_type, ins, outs);
}

template <typename VarType>
static void SetForwardDataTypeOfGradVars(const NameVarMap<VarType>& outs) {
  for (auto& var_pair : outs) {
    for (auto& var : var_pair.second) {
      // NOTE(zhiqu): The output may be NULL because of pruning.
      if (var) {
        SetForwardDataTypeOfGradVar(var);
      }
    }
  }
}
template <>
void SetForwardDataTypeOfGradVars<egr::EagerVariable>(
    const NameVarMap<egr::EagerVariable>& outs) {
  // In eager mode we don't need this.
}

void TestSetForwardDataTypeOfGradVarsEager(
    const NameVarMap<egr::EagerVariable>& outs) {
  SetForwardDataTypeOfGradVars<egr::EagerVariable>(outs);
}

VarBase::VarBase(const std::shared_ptr<VariableWrapper>& var)
    : var_(var), grad_node_(var->GetGradNode()) {
  if (auto grad_var = var_->GetGradVar()) {
    grad_var_ = std::make_shared<VarBase>(grad_var);
  }

  if (IsDebugEnabled()) {
    VLOG(10) << "Construct VarBase: " << Name();
    name_set_.Insert(Name());
  }
}

size_t VarBase::GradOpNum() const {
  return grad_node_ ? grad_node_->size() : 0;
}

void VarBase::ClearGradient(bool set_to_zero) {
  VLOG(4) << "ClearGradient " << Name();
  if (grad_var_) {
    if (grad_var_->Var().IsType<phi::SelectedRows>()) {
      auto* grad_t = grad_var_->MutableVar()->GetMutable<phi::SelectedRows>();
      if (grad_t->mutable_value()->IsInitialized()) {
#ifdef PADDLE_WITH_MKLDNN
        if (FLAGS_use_mkldnn) platform::ClearMKLDNNCache(grad_t->place());
#endif
        grad_t->mutable_rows()->clear();
        grad_t->mutable_value()->clear();
      }
    } else {
      platform::RecordEvent record_event(
          "ClearGradient", platform::TracerEventType::UserDefined, 2);
      auto* grad_t = grad_var_->MutableVar()->GetMutable<phi::DenseTensor>();
      if (grad_t->IsInitialized()) {
        if (set_to_zero) {
          auto* dev_ctx =
              platform::DeviceContextPool::Instance().Get(grad_t->place());
          phi::funcs::set_constant(*dev_ctx, grad_t, 0.0);
        } else {
          grad_t->clear();
        }
#ifdef PADDLE_WITH_MKLDNN
        if (FLAGS_use_mkldnn) platform::ClearMKLDNNCache(grad_t->place());
#endif
      }
    }
    // TODO(zhouwei): It's better to free memory of grad by grad_t->claer.
    // But will have some bug on mac CPU of yolov3 model, why?
    // After fix this bug, function SetIsEmpty() isn't need
    grad_var_->SharedVar()->SetIsEmpty(true);
  }
}

void VarBase::_GradientSetEmpty(bool is_empty) {
  VLOG(4) << "Set gradient " << Name() << " is_empty:" << is_empty;
  if (grad_var_) {
    auto share_var = grad_var_->SharedVar();
    if (share_var) {
      share_var->SetIsEmpty(is_empty);
    }
  }
}

bool VarBase::_IsGradientSetEmpty() {
  bool res = true;
  if (grad_var_) {
    auto share_var = grad_var_->SharedVar();
    if (share_var) {
      res = share_var->is_empty_;
      VLOG(4) << "Check gradient " << Name() << " is empty:" << res;
    }
  }
  return res;
}

std::shared_ptr<VarBase> VarBase::NewVarBase(const platform::Place& dst_place,
                                             const bool blocking) const {
  PADDLE_ENFORCE_EQ(
      Var().IsInitialized() && (Var().IsType<phi::DenseTensor>() ||
                                Var().IsType<phi::SelectedRows>()),
      true,
      platform::errors::InvalidArgument(
          "Variable is not initialized or Variable's type is not "
          "LoDTensor or SelectedRows when getting numpy tensor"));

  if (Var().IsType<phi::DenseTensor>()) {
    auto& src_tensor = Var().Get<phi::DenseTensor>();
    // TODO(Jiabin): change this after move unique_name generator to CXX
    auto new_var = std::make_shared<VarBase>(
        true, Name() + std::to_string(copied_counter_++));

    auto* dst_tensor = new_var->MutableVar()->GetMutable<phi::DenseTensor>();
    dst_tensor->set_lod(src_tensor.lod());
    new_var->SetPersistable(Persistable());
    new_var->SetDataType(DataType());
    new_var->SetType(Type());
    framework::TensorCopy(src_tensor, dst_place, dst_tensor);
    if (blocking) {
      platform::DeviceContextPool::Instance().Get(dst_place)->Wait();
      auto src_place = src_tensor.place();
      if (!(src_place == dst_place)) {
        platform::DeviceContextPool::Instance().Get(src_place)->Wait();
      }
    }
    VLOG(4) << "copy tensor " << Name() << " from " << Place() << " to "
            << dst_place;
    return new_var;
  } else {
    auto& src_selected_rows = Var().Get<phi::SelectedRows>();
    auto new_var = std::make_shared<VarBase>(
        false, "Itmp" + std::to_string(copied_counter_++));
    new_var->SetType(framework::proto::VarType::SELECTED_ROWS);
    auto* dst_selected_rows =
        new_var->MutableVar()->GetMutable<phi::SelectedRows>();

    framework::TensorCopy(src_selected_rows.value(),
                          dst_place,
                          dst_selected_rows->mutable_value());
    if (blocking) {
      platform::DeviceContextPool::Instance().Get(dst_place)->Wait();
      auto src_place = src_selected_rows.place();
      if (!(src_place == dst_place)) {
        platform::DeviceContextPool::Instance().Get(src_place)->Wait();
      }
    }
    dst_selected_rows->set_height(src_selected_rows.height());
    dst_selected_rows->set_rows(src_selected_rows.rows());
    VLOG(4) << "copy tensor " << Name() << " from " << Place() << " to "
            << dst_place;
    return new_var;
  }
}

void VarBase::CopyFrom(const VarBase& src, const bool blocking) {
  if (src.SharedVar()->IsEmpty()) {
    return;
  }

  VLOG(3) << "Deep copy Tensor from " << src.Name() << " to " << Name();
  if (Var().IsInitialized()) {
    PADDLE_ENFORCE_EQ(DataType(),
                      src.DataType(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s, "
                          "Tensor Copy cannot be performed!",
                          Name(),
                          src.Name()));
    PADDLE_ENFORCE_EQ(Type(),
                      src.Type(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "Copy cannot be performed!",
                          Name(),
                          src.Name()));
  } else {
    SetDataType(src.DataType());
    SetType(src.Type());
    SetPersistable(src.Persistable());
    InnerSetOverridedStopGradient(src.OverridedStopGradient());
  }

  platform::Place place = src.Place();
  if (src.Var().IsType<phi::DenseTensor>()) {
    auto& src_tensor = src.Var().Get<phi::DenseTensor>();
    auto* dst_tensor = MutableVar()->GetMutable<phi::DenseTensor>();
    if (dst_tensor && dst_tensor->IsInitialized()) {
      PADDLE_ENFORCE_EQ(dst_tensor->dims(),
                        src_tensor.dims(),
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has different dims with Tensor %s, "
                            "Tensor Copy cannot be performed!",
                            Name(),
                            src.Name()));
      PADDLE_ENFORCE_EQ(dst_tensor->lod(),
                        src_tensor.lod(),
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has different dims with Tensor %s, "
                            "Tensor Copy cannot be performed!",
                            Name(),
                            src.Name()));
      place = Place();
    } else {
      dst_tensor->set_lod(src_tensor.lod());
      dst_tensor->Resize(src_tensor.dims());
    }
    framework::TensorCopy(src_tensor, place, dst_tensor);
  } else if (src.Var().IsType<phi::SelectedRows>()) {
    auto& src_selected_rows = src.Var().Get<phi::SelectedRows>();
    auto* dst_selected_rows = MutableVar()->GetMutable<phi::SelectedRows>();
    dst_selected_rows->set_height(src_selected_rows.height());
    dst_selected_rows->set_rows(src_selected_rows.rows());

    auto& src_tensor = src_selected_rows.value();
    auto* dst_tensor = dst_selected_rows->mutable_value();
    if (dst_tensor && dst_tensor->IsInitialized()) {
      PADDLE_ENFORCE_EQ(dst_tensor->dims(),
                        src_tensor.dims(),
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has different dims with Tensor %s, "
                            "Tensor Copy cannot be performed!",
                            Name(),
                            src.Name()));
      place = Place();
    } else {
      dst_tensor->Resize(src_tensor.dims());
    }
    framework::TensorCopy(src_tensor, place, dst_tensor);
  }
  if (blocking) {
    platform::DeviceContextPool::Instance().Get(place)->Wait();
  }
}

void VarBase::BumpInplaceVersion() {
  PADDLE_ENFORCE_EQ(
      Var().IsInitialized(),
      true,
      platform::errors::InvalidArgument(
          "Tensor %s has not been initialized, please check if it has no data.",
          Name()));
  MutableVar()->BumpInplaceVersion();
}

// NOTE(weilong wu):
// This function try to copy the data from target varbase,
// and fill into the grad_var_ of the current varbase.
void VarBase::_CopyGradientFrom(const VarBase& src) {
  if (Var().IsInitialized()) {
    PADDLE_ENFORCE_EQ(DataType(),
                      src.DataType(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different data type with Tensor %s",
                          Name(),
                          src.Name()));
    PADDLE_ENFORCE_EQ(Type(),
                      src.Type(),
                      platform::errors::PreconditionNotMet(
                          "Tensor %s has different type with Tensor %s, Tensor "
                          "ShareGradientDataWith cannot be performed!",
                          Name(),
                          src.Name()));
  }
  VLOG(4) << " VarBase copy gradient with " << src.Name();
  if (grad_var_) {
    auto& src_tensor = src.Var().Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(src_tensor.IsInitialized(),
                      true,
                      platform::errors::InvalidArgument(
                          "Tensor %s has not been initialized", src.Name()));
    auto* grad_t = grad_var_->MutableVar()->GetMutable<phi::DenseTensor>();
    auto* var_ = MutableVar()->GetMutable<phi::DenseTensor>();
    grad_t->ShareDataWith(src_tensor);
    grad_t->Resize(var_->dims());
  }
}

void OpBase::SetType(const std::string& type) {
  op_ = framework::OpRegistry::CreateOp(type, {}, {}, {}, false);
}

void OpBase::ClearBackwardTrace() {
  ins_.clear();
  outs_.clear();
}

template <typename VarType>
static void OpBaseRunImpl(const framework::OperatorBase& op,
                          const NameVarMap<VarType>& ins,
                          const NameVarMap<VarType>& outs,
                          const framework::AttributeMap& attrs,
                          const framework::AttributeMap& default_attrs,
                          const platform::Place& place) {
  auto* op_kernel = static_cast<const framework::OperatorWithKernel*>(&op);
  PADDLE_ENFORCE_NOT_NULL(
      op_kernel,
      platform::errors::PermissionDenied(
          "Only support operator with kernel in Dygraph mode."));
  auto& info = op.Info();
  if (info.infer_var_type_) {
    RuntimeInferVarTypeContext<VarType> infer_var_type_ctx(
        ins, outs, attrs, default_attrs);
    info.infer_var_type_(&infer_var_type_ctx);
  }

  // Initialize output var type
  for (auto& var_pair : outs) {
    for (auto& var : var_pair.second) {
      if (var) {
        InitializeVariable(var->MutableVar(), GetType(var));
      }
    }
  }

  VLOG(5) << LayerDebugString(op.Type(), ins, outs);
  op_step_id++;
  VLOG(10) << "Op ID: " << op_step_id;
  /**
   * [ Why need temporary inputs here? ]
   *
   * PrepareData should not change original input tensor inplace.
   * Suppose the user defines a tensor(int), enters an op to execute,
   * and then this op rewrites GetExpectedKernelForVar, and converts
   * this tensor to float type during execution. After the dynamic
   * graph is executed, the user-defined variable will be lost, and
   * the user cannot get the originally defined int tensor, because
   * it has been converted to float, this should be regarded as a bug
   * in certain usage scenarios
   *
   * In static graph mode, when op is executed, a temporary scope
   * `transfer_scope` is created before PrepareData, the data after
   * transform is stored in the temporary scope, and then discarded
   * after the execution of op, but the original input is directly
   * overwritten in the previous dynamic graph implementation.
   */
  VLOG(10) << "Start prepare dev1!";
  auto prepared_op =
      PreparedOp::Prepare(ins, outs, *op_kernel, place, attrs, default_attrs);
  auto tmp_ins_ptr = PrepareData<VarType>(
      *op_kernel, ins, prepared_op.kernel_key(), prepared_op.place());
  VLOG(10) << "End prepare dev1!";

  std::shared_ptr<NameVarMap<VarType>> ins_dev2_ptr = nullptr;
  std::shared_ptr<NameVarMap<VarType>> outs_dev2_ptr = nullptr;
  if (DebugOrNot()) {
    VLOG(10) << "Start copy input!";
    ins_dev2_ptr = TemporaryData<VarType>(ins, xpu_debug_run_dev2());
    VLOG(10) << "End copy input!";
    VLOG(10) << "Start copy output!";
    outs_dev2_ptr =
        TemporaryData<VarType>(outs, xpu_debug_run_dev2(), ins, ins_dev2_ptr);
    VLOG(10) << "End copy output!";
  }
  VLOG(10) << "Start prepare dev2!";
  auto prepared_op_dev2 = PreparedOp::Prepare(*ins_dev2_ptr,
                                              *outs_dev2_ptr,
                                              *op_kernel,
                                              xpu_debug_run_dev2(),
                                              attrs,
                                              default_attrs);
  auto tmp_ins_dev2_ptr =
      DebugPrepareData<VarType>(*op_kernel,
                                *ins_dev2_ptr,
                                prepared_op_dev2.kernel_key(),
                                prepared_op_dev2.place());
  VLOG(10) << "End prepare dev2!";

  std::string debug_str;
  if (DebugOrNot()) {
    VLOG(10) << "Start check mse for input!";
    debug_str = XPUDebugStartString(op.Type(), prepared_op);
    debug_str = XPUDebugString(op.Type(),
                               debug_str,
                               ins,
                               *ins_dev2_ptr,
                               prepared_op.place(),
                               prepared_op_dev2.place());
    VLOG(10) << "End check mse for input!";
  }

  std::string dump_data_path;
  std::string dump_data_dubug_path;
  if (std::getenv("XPU_PADDLE_DUMP_DATA_PATH") != nullptr) {
    dump_data_path = XPUDebugDumpDataStart(op.Type(), prepared_op);
    XPUDebugDumpData(op.Type(), dump_data_path, "input", ins, place);
    if (DebugOrNot()) {
      dump_data_dubug_path =
          XPUDebugDumpDataStart(op.Type(), prepared_op_dev2, "-debug");
      XPUDebugDumpData(op.Type(),
                       dump_data_dubug_path,
                       "input",
                       *ins_dev2_ptr,
                       xpu_debug_run_dev2());
    }
  }

  if (std::getenv("XPU_PADDLE_OP_TIME") != nullptr) {
    XPUPaddleOpTimeTik();
  }

  VLOG(10) << "Strat run dev1";
  if (tmp_ins_ptr == nullptr) {
    prepared_op.Run(ins, outs, attrs, default_attrs);
  } else {
    prepared_op.Run(*tmp_ins_ptr, outs, attrs, default_attrs);
  }
  VLOG(10) << "End run dev1";

  if (std::getenv("XPU_PADDLE_OP_TIME") != nullptr) {
    XPUPaddleOpTimeTok(op, prepared_op, place);
  }

  if (DebugOrNot() && ContinueRunDev2OrNot(op.Type())) {
    VLOG(10) << "Strat run dev2";
    if (paddle::platform::is_xpu_place(prepared_op.place())) {
      xpu_wait();
    }
    if (tmp_ins_dev2_ptr == nullptr) {
      prepared_op_dev2.Run(*ins_dev2_ptr, *outs_dev2_ptr, attrs, default_attrs);
    } else {
      prepared_op_dev2.Run(
          *tmp_ins_dev2_ptr, *outs_dev2_ptr, attrs, default_attrs);
    }
    if (paddle::platform::is_xpu_place(prepared_op_dev2.place())) {
      xpu_wait();
    }
    VLOG(10) << "End run dev2";
  }

  if (DebugOrNot()) {
    VLOG(10) << "Start copy output after dev2 run!";
    CopyOutputData<VarType>(
        op.Type(), outs, outs_dev2_ptr.get(), xpu_debug_run_dev2());
    VLOG(10) << "End copy output after dev2 run!";
    VLOG(10) << "Start check mse for output!";
    debug_str = XPUDebugString(op.Type(),
                               debug_str + " out: ",
                               outs,
                               *outs_dev2_ptr,
                               prepared_op.place(),
                               prepared_op_dev2.place());
    VLOG(10) << "End check mse for output!";
    if (debug_str != "") {
      std::cout << debug_str << std::endl;
    }
  }

  if (std::getenv("XPU_PADDLE_DUMP_DATA_PATH") != nullptr) {
    XPUDebugDumpData(op.Type(), dump_data_path, "output", outs, place);
    if (DebugOrNot()) {
      XPUDebugDumpData(op.Type(),
                       dump_data_dubug_path,
                       "output",
                       *outs_dev2_ptr,
                       xpu_debug_run_dev2());
    }
  }

  VLOG(4) << LayerDebugString(op.Type(), ins, outs);

  // set the output var
  SetForwardDataTypeOfGradVars<VarType>(outs);
}

void OpBase::Run(const framework::OperatorBase& op,
                 const NameVarMap<VarBase>& ins,
                 const NameVarMap<VarBase>& outs,
                 const framework::AttributeMap& attrs,
                 const framework::AttributeMap& default_attrs,
                 const platform::Place& place) {
  OpBaseRunImpl<VarBase>(op, ins, outs, attrs, default_attrs, place);
}

void OpBase::Run(const framework::OperatorBase& op,
                 const NameVarMap<VariableWrapper>& ins,
                 const NameVarMap<VariableWrapper>& outs,
                 const framework::AttributeMap& attrs,
                 const framework::AttributeMap& default_attrs,
                 const platform::Place& place) {
  OpBaseRunImpl<VariableWrapper>(op, ins, outs, attrs, default_attrs, place);
}

void OpBase::Run(const framework::OperatorBase& op,
                 const NameVarMap<egr::EagerVariable>& ins,
                 const NameVarMap<egr::EagerVariable>& outs,
                 const framework::AttributeMap& attrs,
                 const framework::AttributeMap& default_attrs,
                 const platform::Place& place) {
  OpBaseRunImpl<egr::EagerVariable>(op, ins, outs, attrs, default_attrs, place);
}

void ClearNoNeedBufferInputs(OpBase* op) {
  auto& inferer = op->Info().NoNeedBufferVarsInferer();
  if (!inferer) return;
  auto* ins = op->GetMutableInsMap();
  const auto& no_need_buffer_slots =
      inferer(*ins, op->GetOutsMap(), op->Attrs());
  if (no_need_buffer_slots.empty()) return;

  for (auto& slot : no_need_buffer_slots) {
    auto iter = ins->find(slot);
    if (iter == ins->end()) continue;
    VLOG(2) << "Clear data buffer of " << slot << " in " << op->Type();

    PADDLE_ENFORCE_EQ(
        iter->second.IsGrad(),
        false,
        platform::errors::InvalidArgument(
            "Only forward variable buffers can be clear, this may be a bug"));

    for (auto& each_var : *(iter->second.MutableVarList())) {
      if (!each_var) continue;

      auto& var = each_var->Var();
      PADDLE_ENFORCE_EQ(var.IsType<phi::DenseTensor>(),
                        true,
                        platform::errors::PermissionDenied(
                            "NoNeedBufferVars only support LoDTensor"));
      auto new_var = new VariableWrapper(each_var->Name());
      auto* new_tensor = new_var->MutableVar()->GetMutable<phi::DenseTensor>();
      auto& old_tensor = var.Get<phi::DenseTensor>();
      new_tensor->Resize(old_tensor.dims());
      new_tensor->set_lod(old_tensor.lod());
      new_tensor->set_type(old_tensor.dtype());
      new_tensor->set_layout(old_tensor.layout());
      each_var.reset(new_var);
    }
  }
}

std::shared_ptr<GradOpNode> CreateGradOpNode(
    const framework::OperatorBase& op,
    const NameVarBaseMap& ins,
    const NameVarBaseMap& outs,
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    const platform::Place& place,
    const std::map<std::string, std::string>& inplace_map) {
  const auto& info = op.Info();
  if (!info.dygraph_grad_op_maker_) {
    return nullptr;
  }

  auto grad_node = info.dygraph_grad_op_maker_(
      op.Type(), ins, outs, attrs, default_attrs, inplace_map);
  if (grad_node && !grad_node->empty()) {
    for (auto& grad_op : *grad_node) {
      grad_op.SetId(OpBase::GenerateUniqueId());
      grad_op.SetPlace(place);
      ClearNoNeedBufferInputs(&grad_op);
    }
    return grad_node;
  } else {
    return nullptr;
  }
}

std::shared_ptr<GradOpNode> CreateGradOpNode(
    const framework::OperatorBase& op,
    const NameTensorMap& ins,
    const NameTensorMap& outs,
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    const platform::Place& place,
    const std::map<std::string, std::string>& inplace_map) {
  // Do Nothing in Eager Mode.
  return nullptr;
}

}  // namespace imperative
}  // namespace paddle
