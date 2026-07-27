#pragma once
#include <algorithm>

#include "../arguments.hpp"
#include "cell_ownership.hpp"
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {

        struct Numbering
        {
            // Local index in the PETSc vector/matrix for each local unknown
            std::vector<PetscInt> local_indices;
            // Global index in the PETSc vector/matrix for each local unknown
            std::vector<PetscInt> global_indices;
            // Mapping from local to global unknown indices
            std::vector<PetscInt> local_to_global_mapping;

            SAMURAI_INLINE void resize(PetscInt n_local_unknowns)
            {
                local_indices.resize(static_cast<std::size_t>(n_local_unknowns));
                global_indices.resize(static_cast<std::size_t>(n_local_unknowns));
                local_to_global_mapping.resize(static_cast<std::size_t>(n_local_unknowns));
            }

            template <int n_unknowns_per_cell, typename return_type = std::size_t>
            SAMURAI_INLINE return_type unknown_index([[maybe_unused]] const CellOwnership& ownership,
                                                     PetscInt shift,
                                                     std::size_t cell_index,
                                                     [[maybe_unused]] int component_index) const
            {
#ifdef SAMURAI_WITH_MPI
                cell_index = static_cast<std::size_t>(ownership.cell_indices[cell_index]);
#endif
                if constexpr (n_unknowns_per_cell == 1)
                {
                    return static_cast<return_type>(shift + static_cast<PetscInt>(cell_index));
                }
                else
                {
                    return static_cast<return_type>(shift + static_cast<PetscInt>(cell_index) * n_unknowns_per_cell + component_index);
                }
            }
        };

#ifdef SAMURAI_WITH_MPI

        PetscInt compute_rank_shift(PetscInt n_values)
        {
            PetscInt rank_shift = 0;
            MPI_Exscan(&n_values, &rank_shift, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
            return rank_shift;
        }

        template <int n_unknowns_per_cell, class Mesh>
        void compute_global_numbering(const Mesh& mesh,
                                      Numbering& numbering,
                                      PetscInt rank_shift,
                                      PetscInt block_shift_owned,
                                      PetscInt block_shift_ghosts)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            mpi::communicator world;
            int rank = world.rank();

            assert(rank == 0 || rank_shift > 0);

            const auto& ownership = mesh.cell_ownership();

            assert(mesh.nb_cells() == ownership.n_local_cells);

            std::size_t min_level = mesh[mesh_id_t::reference].min_level();
            std::size_t max_level = mesh[mesh_id_t::reference].max_level();

            auto& local_indices  = numbering.local_indices;
            auto& global_indices = numbering.global_indices;

            constexpr int UNSET = -1;

            auto owned_unknown_index = [&](std::size_t cell_index, int i_comp)
            {
                return numbering.template unknown_index<n_unknowns_per_cell, std::size_t>(ownership, block_shift_owned, cell_index, i_comp);
            };

            auto ghost_unknown_index = [&](std::size_t cell_index, int i_comp)
            {
                return numbering.template unknown_index<n_unknowns_per_cell, std::size_t>(ownership, block_shift_ghosts, cell_index, i_comp);
            };

            auto n_owned_unknowns = ownership.n_owned_cells * static_cast<std::size_t>(n_unknowns_per_cell);
            assert(local_indices.size() - static_cast<std::size_t>(block_shift_owned) >= n_owned_unknowns);
            auto owned_local_index_begin  = local_indices.begin() + static_cast<std::ptrdiff_t>(block_shift_owned);
            auto owned_local_index_end    = owned_local_index_begin + static_cast<std::ptrdiff_t>(n_owned_unknowns);
            auto owned_global_index_begin = global_indices.begin() + static_cast<std::ptrdiff_t>(block_shift_owned);
            auto owned_global_index_end   = owned_global_index_begin + static_cast<std::ptrdiff_t>(n_owned_unknowns);
            std::fill(owned_local_index_begin, owned_local_index_end, UNSET);
            std::fill(owned_global_index_begin, owned_global_index_end, UNSET);

            auto n_ghost_unknowns = (ownership.n_local_cells - ownership.n_owned_cells) * static_cast<std::size_t>(n_unknowns_per_cell);
            assert(local_indices.size() - static_cast<std::size_t>(block_shift_ghosts) >= n_ghost_unknowns);
            auto ghost_local_index_begin  = local_indices.begin() + static_cast<std::ptrdiff_t>(block_shift_ghosts);
            auto ghost_local_index_end    = ghost_local_index_begin + static_cast<std::ptrdiff_t>(n_ghost_unknowns);
            auto ghost_global_index_begin = global_indices.begin() + static_cast<std::ptrdiff_t>(block_shift_ghosts);
            auto ghost_global_index_end   = ghost_global_index_begin + static_cast<std::ptrdiff_t>(n_ghost_unknowns);
            std::fill(ghost_local_index_begin, ghost_local_index_end, UNSET);
            std::fill(ghost_global_index_begin, ghost_global_index_end, UNSET);

            //-------------//
            // Owned cells //
            //-------------//

            PetscInt local_index  = static_cast<PetscInt>(block_shift_owned);
            PetscInt global_index = rank_shift + static_cast<PetscInt>(block_shift_owned);

            if (args::print_petsc_numbering)
            {
                sleep(static_cast<unsigned int>(rank));
                std::cout << fmt::format("[{}]: n_owned_unknowns = {}, n_ghost_unknowns = {}\n", rank, n_owned_unknowns, n_ghost_unknowns);
                std::cout << fmt::format("[{}]: OWNED local_index = [{},{}], global_index = [{},{}]\n",
                                         rank,
                                         local_index,
                                         local_index + static_cast<PetscInt>(n_owned_unknowns) - 1,
                                         global_index,
                                         global_index + static_cast<PetscInt>(n_owned_unknowns) - 1);
            }

            for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
            {
                if (ownership.owner_rank[cell_index] == rank)
                {
                    for (int i = 0; i < n_unknowns_per_cell; ++i)
                    {
                        local_indices[owned_unknown_index(cell_index, i)]  = local_index++;
                        global_indices[owned_unknown_index(cell_index, i)] = global_index++;
                    }
                }
            }

            //-------------//
            // Ghost cells //
            //-------------//

            local_index = static_cast<PetscInt>(block_shift_ghosts);

            if (args::print_petsc_numbering)
            {
                std::cout << fmt::format("rank {}: GHOSTS local_index = [{},{}]\n",
                                         rank,
                                         local_index,
                                         local_index + static_cast<PetscInt>(n_ghost_unknowns) - 1);
            }

            for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
            {
                if (ownership.owner_rank[cell_index] != rank)
                {
                    for (int i = 0; i < n_unknowns_per_cell; ++i)
                    {
                        local_indices[ghost_unknown_index(cell_index, i)] = local_index++;
                    }
                }
            }

            // Exchange global indices of the local cells/ghosts with neighbouring MPI processes.
            //
            // The exchange is positional: the n-th value received is the n-th value the
            // neighbour sent. That only holds if both sides walk the same cells in the same
            // order, and the two conditions below are what make them:
            //
            //   - **one value per shared unknown, owned or not.** Filtering on ownership while
            //     sending and on the *other* rank's ownership while receiving compares two
            //     ranks' opinions, and where those disagree - which is what the ownership
            //     correction passes exist to repair, and cannot always - the sequences drift
            //     and every later ghost gets another cell's global index. The corruption is
            //     silent: it produces a valid-looking index in another rank's range, and the
            //     only symptom is petsc refusing an unexpected column much later. Sending
            //     UNSET for what I do not own costs one integer per shared cell and removes
            //     the possibility;
            //   - **the same level range and the same operand order on both sides.** Both are
            //     derived from the pair, not from one side's mesh: the levels from both
            //     meshes' extent, the operand order from the rank numbers.
            auto shared_cells = [&](const auto& neighbour, std::size_t level)
            {
                const auto& mine  = mesh[mesh_id_t::reference][level];
                const auto& other = neighbour.mesh[mesh_id_t::reference][level];
                return rank < neighbour.rank ? intersection(mine, other) : intersection(other, mine);
            };

            auto shared_max_level = [&](const auto& neighbour)
            {
                return std::max(max_level, neighbour.mesh[mesh_id_t::reference].max_level());
            };

            // SEND
            std::vector<mpi::request> req;
            std::vector<std::vector<PetscInt>> to_send_by_neighbour(mesh.mpi_neighbourhood().size());
            std::size_t i_neigh = 0;
            for (auto& neighbour : mesh.mpi_neighbourhood())
            {
                auto& to_send = to_send_by_neighbour[i_neigh];

                for (std::size_t level = 0; level <= shared_max_level(neighbour); ++level)
                {
                    auto shared = shared_cells(neighbour, level);
                    for_each_cell(mesh,
                                  shared,
                                  [&](auto& cell)
                                  {
                                      auto cell_index  = static_cast<std::size_t>(cell.index);
                                      const bool owned = ownership.owner_rank[cell_index] == rank;
                                      for (int i = 0; i < n_unknowns_per_cell; ++i)
                                      {
                                          to_send.push_back(owned ? global_indices[owned_unknown_index(cell_index, i)] : UNSET);
                                      }
                                  });
                }
                req.push_back(world.isend(neighbour.rank /* dest */, neighbour.rank /* tag */, to_send));
                i_neigh++;
            }

            // RECEIVE
            for (auto& neighbour : mesh.mpi_neighbourhood())
            {
                std::vector<PetscInt> to_recv;
                std::size_t read = 0;
                world.recv(neighbour.rank /* source */, rank /* tag */, to_recv);
                for (std::size_t level = 0; level <= shared_max_level(neighbour); ++level)
                {
                    auto shared = shared_cells(neighbour, level);
                    for_each_cell(mesh,
                                  shared,
                                  [&](auto& cell)
                                  {
                                      auto cell_index = static_cast<std::size_t>(cell.index);
                                      for (int i = 0; i < n_unknowns_per_cell; ++i)
                                      {
                                          const PetscInt sent = to_recv[read++];
                                          if (ownership.owner_rank[cell_index] != neighbour.rank)
                                          {
                                              continue;
                                          }
                                          // The neighbour says it does not own a cell this rank
                                          // considers its ghost. That is an ownership
                                          // disagreement, and it used to become a wrong global
                                          // index; say so instead.
                                          if (sent == UNSET)
                                          {
                                              std::cerr << fmt::format(
                                                  "[{}] rank {} does not own the cell at level {} that this rank holds as its ghost: "
                                                  "the two disagree on ownership, so no global index exists for it.\n",
                                                  rank,
                                                  neighbour.rank,
                                                  level);
                                              continue;
                                          }
                                          assert(global_indices[ghost_unknown_index(cell_index, i)] == UNSET);
                                          global_indices[ghost_unknown_index(cell_index, i)] = sent;
                                      }
                                  });
                }
            }

            mpi::wait_all(req.begin(), req.end());
            if (args::print_petsc_numbering)
            {
                sleep(static_cast<unsigned int>(rank));

                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    for (int i = 0; i < n_unknowns_per_cell; ++i)
                    {
                        if (ownership.owner_rank[cell_index] == rank)
                        {
                            std::cout << fmt::format("[{}]:          cell_index {} (owned by {}): CI{} L{} G{}\n",
                                                     world.rank(),
                                                     cell_index,
                                                     ownership.owner_rank[cell_index],
                                                     ownership.cell_indices[cell_index],
                                                     local_indices[owned_unknown_index(cell_index, i)],
                                                     global_indices[owned_unknown_index(cell_index, i)]);
                        }
                        else
                        {
                            std::cout << fmt::format("[{}]:          cell_index {} (owned by {}): CI{} L{} G{}\n",
                                                     world.rank(),
                                                     cell_index,
                                                     ownership.owner_rank[cell_index],
                                                     ownership.cell_indices[cell_index],
                                                     local_indices[ghost_unknown_index(cell_index, i)],
                                                     global_indices[ghost_unknown_index(cell_index, i)]);
                        }
                    }
                }
            }
        }
#endif

        // Two local unknowns mapping to the same global index is silent corruption: petsc
        // accepts the mapping and the only symptom is a refused column much later, in another
        // rank's range. Sorting a copy makes the check O(n log n) instead of O(n^2), which is
        // what it costs to be affordable at all - the quadratic version could only ever live
        // inside an assert, and asserts are compiled out of the builds that run in CI.
        inline bool has_duplicates(const std::vector<PetscInt>& local_to_global_mapping)
        {
            std::vector<PetscInt> sorted(local_to_global_mapping);
            std::sort(sorted.begin(), sorted.end());
            return std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end();
        }
    }
}
