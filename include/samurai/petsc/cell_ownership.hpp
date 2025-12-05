#pragma once
#include <vector>

namespace samurai
{
    namespace petsc
    {
        struct CellOwnership
        {
            bool is_computed = false;

            std::size_t n_local_cells = 0; // owned cells + ghost cells
            std::size_t n_owned_cells = 0;

            // Owner rank of each cell in the local mesh
            std::vector<int> owner_rank; // cppcheck-suppress unusedStructMember
            // Renumbering of the cells: first all the owned cells, then all the ghosts.
            // This is used to split the ordering of the unknowns (first all the owned unknowns, then all the ghosts).
            // Note that the cell numbers start at index 0, and the ghost numbers also start at index 0!
            std::vector<int> cell_indices; // cppcheck-suppress unusedStructMember
        };
    }
}
