
// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
