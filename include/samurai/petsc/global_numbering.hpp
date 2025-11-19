#pragma once
#include "../arguments.hpp"
#include "../field.hpp"
#include "cell_ownership.hpp"
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {

        struct Numbering
        {
            CellOwnership* ownership = nullptr;
#ifdef SAMURAI_WITH_MPI
            // Local index in the PETSc vector/matrix for each local unknown
            std::vector<PetscInt> local_indices;
            // Global index in the PETSc vector/matrix for each local unknown
            std::vector<PetscInt> global_indices;
            // Mapping from local to global unknown indices
            std::vector<PetscInt> local_to_global_mapping;
#endif
            inline void resize(PetscInt n_local_unknowns)
            {
                local_indices.resize(static_cast<std::size_t>(n_local_unknowns));
                global_indices.resize(static_cast<std::size_t>(n_local_unknowns));
                local_to_global_mapping.resize(static_cast<std::size_t>(n_local_unknowns));
            }

            template <int n_unknowns_per_cell, typename return_type = std::size_t>
            inline return_type unknown_index(PetscInt shift, std::size_t cell_index, [[maybe_unused]] int component_index) const
            {
#ifdef SAMURAI_WITH_MPI
                cell_index = static_cast<std::size_t>(ownership->cell_indices[cell_index]);
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

            const auto& ownership = *numbering.ownership;

            std::size_t min_level = mesh[mesh_id_t::reference].min_level();
            std::size_t max_level = mesh[mesh_id_t::reference].max_level();

            auto& local_indices  = numbering.local_indices;
            auto& global_indices = numbering.global_indices;

            constexpr int UNSET = -1;

            auto owned_unknown_index = [&](std::size_t cell_index, int i_comp)
            {
                return numbering.template unknown_index<n_unknowns_per_cell, std::size_t>(block_shift_owned, cell_index, i_comp);
            };

            auto ghost_unknown_index = [&](std::size_t cell_index, int i_comp)
            {
                return numbering.template unknown_index<n_unknowns_per_cell, std::size_t>(block_shift_ghosts, cell_index, i_comp);
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

            // Exchange global indices of the local cells/ghosts with neighbouring MPI processes

            // SEND
            std::vector<mpi::request> req;
            std::vector<std::vector<PetscInt>> to_send_by_neighbour(mesh.mpi_neighbourhood().size());
            std::size_t i_neigh = 0;
            for (auto& neighbour : mesh.mpi_neighbourhood())
            {
                auto& to_send = to_send_by_neighbour[i_neigh];

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto intersecting_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level],
                                                                      neighbour.mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  intersecting_cells_and_ghosts,
                                  [&](auto& cell)
                                  {
                                      auto cell_index = static_cast<std::size_t>(cell.index);
                                      if (ownership.owner_rank[cell_index] == rank)
                                      {
                                          for (int i = 0; i < n_unknowns_per_cell; ++i)
                                          {
                                              to_send.push_back(global_indices[owned_unknown_index(cell_index, i)]);
                                          }
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
                for (std::size_t level = 0; level <= max_level; ++level)
                {
                    auto intersecting_cells_and_ghosts = intersection(neighbour.mesh[mesh_id_t::reference][level],
                                                                      mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  intersecting_cells_and_ghosts,
                                  [&](auto& cell)
                                  {
                                      auto cell_index = static_cast<std::size_t>(cell.index);
                                      if (ownership.owner_rank[cell_index] == neighbour.rank)
                                      {
                                          for (int i = 0; i < n_unknowns_per_cell; ++i)
                                          {
                                              assert(global_indices[ghost_unknown_index(cell_index, i)] == UNSET);
                                              global_indices[ghost_unknown_index(cell_index, i)] = to_recv[read++];
                                          }
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

            if (args::print_petsc_numbering)
            {
                auto owner_rank_field = make_scalar_field<int>("owner_rank", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    owner_rank_field[cell_index] = ownership.owner_rank[cell_index];
                }

                // auto global_indices_field = make_scalar_field<PetscInt>("petsc_global_index", mesh);
                // for_each_cell(mesh[mesh_id_t::reference],
                //               [&](auto& cell)
                //               {
                //                   global_indices_field[cell] = global_indices[static_cast<std::size_t>(cell.index)];
                //               });

                // auto local_indices_field = make_scalar_field<PetscInt>("petsc_local_index", mesh);
                // for_each_cell(mesh[mesh_id_t::reference],
                //               [&](auto& cell)
                //               {
                //                   local_indices_field[cell] = local_indices[static_cast<std::size_t>(cell.index)];
                //               });

                auto samurai_cell_indices_field = make_scalar_field<std::size_t>("samurai_cell_index", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    samurai_cell_indices_field[cell_index] = static_cast<std::size_t>(cell_index);
                }

                auto petsc_cell_indices_field = make_scalar_field<int>("petsc_cell_index", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    petsc_cell_indices_field[cell_index] = ownership.cell_indices[cell_index];
                }

                save(fs::current_path(),
                     "petsc_indices",
                     {true, true},
                     mesh,
                     owner_rank_field,
                     // global_indices_field,
                     // local_indices_field,
                     samurai_cell_indices_field,
                     petsc_cell_indices_field);
            }
        }

        bool has_duplicates(const std::vector<PetscInt>& local_to_global_mapping)
        {
            bool duplicate_found = false;
            for (std::size_t i = 0; i < local_to_global_mapping.size(); ++i)
            {
                for (std::size_t j = i + 1; j < local_to_global_mapping.size(); ++j)
                {
                    if (local_to_global_mapping[i] == local_to_global_mapping[j])
                    {
                        duplicate_found = true;
                        break;
                    }
                }
                if (duplicate_found)
                {
                    break;
                }
            }
            return duplicate_found;
        }
#endif
    }
}
