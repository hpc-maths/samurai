#pragma once

namespace samurai
{
    enum class CellFlag
    {
        keep = 1,
        coarsen = 2,
        refine = 4,
        enlarge = 8
    };
}