/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// The vendored nvtx3.hpp predates the NVTX v3 headers that CUDA >= 12 ships and
// that PyTorch pulls in; having both declares two different inline versioning
// namespaces inside `nvtx3`, making every name in it ambiguous. These ranges are
// profiling annotations only, so build without them.
#define MINKOWSKI_DISABLE_NVTX 1

#ifndef MINKOWSKI_DISABLE_NVTX
#include "nvtx3.hpp"
#endif

namespace cudf {
/**
 * @brief Tag type for libcudf's NVTX domain.
 *
 */
struct libcudf_domain {
  static constexpr char const* name{"libcudf"};  ///< Name of the libcudf domain
};

/**
 * @brief Alias for an NVTX range in the libcudf domain.
 *
 */
#ifndef MINKOWSKI_DISABLE_NVTX
using thread_range = ::nvtx3::domain_thread_range<libcudf_domain>;
#endif

}  // namespace cudf

/**
 * @brief Convenience macro for generating an NVTX range in the `libcudf` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    CUDF_FUNC_RANGE();
 *    ...
 * }
 * ```
 *
 */
#ifdef MINKOWSKI_DISABLE_NVTX
#define CUDF_FUNC_RANGE() static_cast<void>(0)
#else
#define CUDF_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(cudf::libcudf_domain)
#endif
