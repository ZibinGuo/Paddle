/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"

#include <mutex>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_op_kpfirst_list.h"
#include "paddle/phi/backends/xpu/xpu_op_list.h"

namespace paddle {
namespace platform {

// ops_string contains op_list(e.g., 'mul,mul_grad'), parse the op string and
// insert op to op set
static void tokenize(const std::string& ops,
                     char delim,
                     std::unordered_set<std::string>* op_set) {
  std::string::size_type beg = 0;
  for (uint64_t end = 0; (end = ops.find(delim, end)) != std::string::npos;
       ++end) {
    op_set->insert(ops.substr(beg, end - beg));
    beg = end + 1;
  }

  op_set->insert(ops.substr(beg));
}

bool is_in_xpu_debug_black_list(const std::string& op_name) {
  static bool inited = false;
  static std::unordered_set<std::string> xpu_debug_black_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_PADDLE_DEBUG_BLACK_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_PADDLE_DEBUG_BLACK_LIST"));
        tokenize(ops, ',', &xpu_debug_black_list);
      }
      inited = true;
      VLOG(3) << "XPU Debug Black List: ";
      for (auto iter = xpu_debug_black_list.begin();
           iter != xpu_debug_black_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_debug_black_list.find(op_name) != xpu_debug_black_list.end()) {
    return true;
  }
  return false;
}

bool is_in_xpu_debug_run_dev2_black_list(const std::string& op_name) {
  static bool inited = false;
  static std::unordered_set<std::string> xpu_debug_black_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_PADDLE_DEBUG_RUN_DEV2_BLACK_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_PADDLE_DEBUG_RUN_DEV2_BLACK_LIST"));
        tokenize(ops, ',', &xpu_debug_black_list);
      }
      inited = true;
      VLOG(3) << "XPU Debug Run Dev2 Black List: ";
      for (auto iter = xpu_debug_black_list.begin();
           iter != xpu_debug_black_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_debug_black_list.find(op_name) != xpu_debug_black_list.end()) {
    return true;
  }
  return false;
}

bool is_in_xpu_debug_black_id_list(const std::string& op_id) {
  static bool inited = false;
  static std::unordered_set<std::string> xpu_debug_black_id_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_PADDLE_DEBUG_BLACK_ID_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_PADDLE_DEBUG_BLACK_ID_LIST"));
        tokenize(ops, ',', &xpu_debug_black_id_list);
      }
      inited = true;
      VLOG(3) << "XPU Debug Black ID List: ";
      for (auto iter = xpu_debug_black_id_list.begin();
           iter != xpu_debug_black_id_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_debug_black_id_list.find(op_id) != xpu_debug_black_id_list.end()) {
    return true;
  }
  return false;
}

bool is_in_xpu_debug_white_list(const std::string& op_name) {
  static bool inited = false;
  static std::unordered_set<std::string> xpu_debug_white_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_PADDLE_DEBUG_WHITE_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_PADDLE_DEBUG_WHITE_LIST"));
        tokenize(ops, ',', &xpu_debug_white_list);
      }
      inited = true;
      VLOG(3) << "XPU Debug White List: ";
      for (auto iter = xpu_debug_white_list.begin();
           iter != xpu_debug_white_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_debug_white_list.find(op_name) != xpu_debug_white_list.end()) {
    return true;
  }
  return false;
}

bool is_in_xpu_debug_white_id_list(const std::string& op_id) {
  static bool inited = false;
  static std::unordered_set<std::string> xpu_debug_white_id_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_PADDLE_DEBUG_WHITE_ID_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_PADDLE_DEBUG_WHITE_ID_LIST"));
        tokenize(ops, ',', &xpu_debug_white_id_list);
      }
      inited = true;
      VLOG(3) << "XPU Debug White ID List: ";
      for (auto iter = xpu_debug_white_id_list.begin();
           iter != xpu_debug_white_id_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_debug_white_id_list.find(op_id) != xpu_debug_white_id_list.end()) {
    return true;
  }
  return false;
}

platform::Place& xpu_debug_run_dev2() {
  static platform::Place dev2 = platform::CPUPlace();
  static bool inited = false;
  static std::string device = "CPU";
  if (!inited) {
    if (std::getenv("XPU_PADDLE_DEBUG_RUN_DEV2") != nullptr) {
      std::string ops(std::getenv("XPU_PADDLE_DEBUG_RUN_DEV2"));
      if (ops == "1") {
        dev2 = platform::XPUPlace();
        device = "XPU";
      }
    }
    inited = true;
    VLOG(3) << "XPU Paddle Debug Run Dev2: " << device;
  }
  return dev2;
}

#ifdef PADDLE_WITH_XPU_KP
bool is_in_xpu_kpwhite_list(const std::string& op_name) {
  static bool inited = false;
  static std::unordered_set<std::string> xpu_kpwhite_list;
  static std::mutex s_mtx;
  if (!inited) {
    std::lock_guard<std::mutex> guard(s_mtx);
    if (!inited) {
      if (std::getenv("XPU_KPWHITE_LIST") != nullptr) {
        std::string ops(std::getenv("XPU_KPWHITE_LIST"));
        tokenize(ops, ',', &xpu_kpwhite_list);
      }
      inited = true;
      VLOG(3) << "XPU kpwhite List: ";
      for (auto iter = xpu_kpwhite_list.begin(); iter != xpu_kpwhite_list.end();
           ++iter) {
        VLOG(3) << *iter << " ";
      }
    }
  }
  if (xpu_kpwhite_list.find(op_name) != xpu_kpwhite_list.end()) {
    return true;
  }
  return false;
}
#endif

std::vector<vartype::Type> get_xpu_op_support_type(
    const std::string& op_name, phi::backends::xpu::XPUVersion version) {
  auto& ops = version == phi::backends::xpu::XPUVersion::XPU1
                  ? phi::backends::xpu::get_kl1_ops()
                  : phi::backends::xpu::get_kl2_ops();
  std::vector<vartype::Type> res;
  if (ops.find(op_name) != ops.end()) {
    auto& dtypes = ops[op_name];
    for (auto& type : dtypes) {
      res.push_back(static_cast<vartype::Type>(phi::TransToProtoVarType(type)));
    }
  }
  return res;
}

XPUOpListMap get_xpu_op_list(phi::backends::xpu::XPUVersion version) {
  auto& ops = version == phi::backends::xpu::XPUVersion::XPU1
                  ? phi::backends::xpu::get_kl1_ops()
                  : phi::backends::xpu::get_kl2_ops();
  XPUOpListMap res;
  for (auto& op : ops) {
    std::vector<vartype::Type> op_types;
    for (auto& item : op.second) {
      op_types.push_back(
          static_cast<vartype::Type>(phi::TransToProtoVarType(item)));
    }
    res[op.first] = std::move(op_types);
  }
  return res;
}

}  // namespace platform
}  // namespace paddle
#endif
