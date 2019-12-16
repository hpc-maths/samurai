#pragma once

namespace mure
{
    enum class CellFlag
    {
        keep = 1,
        coarsen = 2,
        refine = 4
    };
}