// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "set_traverser_base.hpp"

namespace samurai
{
    template <std::size_t Dim, class TInterval>
    class LevelCellArray;

    template <class LCA>
    using LCATraverser = RangeTraverser<typename std::vector<typename LCA::interval_t>::const_iterator>;

} // namespace samurai
