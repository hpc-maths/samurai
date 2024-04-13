
// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    enum class CellFlag
    {
        keep    = 1,
        coarsen = 2,
        refine  = 4,
        enlarge = 8
    };
} // namespace samurai
