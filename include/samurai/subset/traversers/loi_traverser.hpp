// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../list_of_intervals.hpp"
#include "range_traverser.hpp"

namespace samurai
{
    template <typename TValue>
    using LOITraverser = RangeTraverser<typename ListOfIntervals<TValue>::const_iterator>;

} // namespace samurai
