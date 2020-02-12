#pragma once

namespace mure
{
    enum class MeshType
    {
        cells = 0,
        cells_and_ghosts = 1,
        proj_cells = 2,
        all_cells = 3,
        union_cells = 4
    };
}
