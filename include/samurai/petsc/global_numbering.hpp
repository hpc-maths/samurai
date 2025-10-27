#pragma once
#include "../arguments.hpp"
#include "../field.hpp"
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {

        struct Numbering
        {
            std::size_t n_cells       = 0;
            std::size_t n_owned_cells = 0;
#ifdef SAMURAI_WITH_MPI
            // Owner rank of each cell in the local mesh
            std::vector<int> ownership;
            // Renumbering of the cells: first all the owned cells, then all the ghosts.
            // This is used to split the ordering of the unknowns (first all the owned unknowns, then all the ghosts).
            // Note that the cells start at index 0, and the ghosts also start at index 0!
            std::vector<int> cell_indices;
            // Local index in the PETSc vector/matrix for each local unknown
            std::vector<PetscInt> local_indices;
            // Global index in the PETSc vector/matrix for each local unknown
            std::vector<PetscInt> global_indices;
#endif

            template <class Mesh>
            void compute_ownership(const Mesh& mesh)
            {
                n_cells = mesh.nb_cells();
#ifndef SAMURAI_WITH_MPI
                n_owned_cells = n_cells;
#else
                using mesh_id_t = typename Mesh::mesh_id_t;

                mpi::communicator world;
                int rank = world.rank();

                std::size_t min_level = mesh[mesh_id_t::reference].min_level();
                std::size_t max_level = mesh[mesh_id_t::reference].max_level();

                // auto& ownership = numbering.ownership;

                constexpr int UNSET = -1;

                // Stores the owner rank of each cell
                ownership.resize(n_cells);
                std::fill(ownership.begin(), ownership.end(), UNSET);

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    // All cells and ghosts that intersect with the subdomain are owned by the current rank
                    auto local_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level], self(mesh.subdomain()).on(level));
                    for_each_cell(mesh,
                                  local_cells_and_ghosts,
                                  [&](auto& cell)
                                  {
                                      ownership[static_cast<std::size_t>(cell.index)] = rank;
                                  });

                    // All the boundary ghosts of locally owned boundary cells are also owned by the local rank
                    auto domain_bdry_outer_layer = domain_boundary_outer_layer(mesh, level, Mesh::config::ghost_width);
                    auto boundary_ghosts         = intersection(domain_bdry_outer_layer, mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  boundary_ghosts,
                                  [&](auto& ghost)
                                  {
                                      ownership[static_cast<std::size_t>(ghost.index)] = rank;
                                  });
                }

                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    for (std::size_t level = min_level; level <= max_level; ++level)
                    {
                        // Ghosts that intersect with a neighbour subdomain are owned by the neighbour rank
                        auto neighbour_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level],
                                                                       self(neighbour.mesh.subdomain()).on(level));
                        for_each_cell(mesh,
                                      neighbour_cells_and_ghosts,
                                      [&](auto& cell)
                                      {
                                          // assert(ownership[static_cast<std::size_t>(cell.index)] == UNSET);
                                          if (ownership[static_cast<std::size_t>(cell.index)] == UNSET)
                                          {
                                              ownership[static_cast<std::size_t>(cell.index)] = neighbour.rank;
                                          }
                                          else
                                          {
                                              std::cout << fmt::format("rank {}: cell {} on level {} owned by {}\n",
                                                                       rank,
                                                                       cell.index,
                                                                       level,
                                                                       ownership[static_cast<std::size_t>(cell.index)]);
                                              ownership[static_cast<std::size_t>(cell.index)] = 10;
                                              assert(false);
                                          }
                                      });

                        // All the boundary ghosts of attached to boundary cells owned by a neighbour are also owned by that neighbour
                        auto domain_bdry_outer_layer = domain_boundary_outer_layer(neighbour.mesh, level, Mesh::config::ghost_width);
                        auto boundary_ghosts         = intersection(domain_bdry_outer_layer, mesh[mesh_id_t::reference][level]);
                        for_each_cell(mesh,
                                      boundary_ghosts,
                                      [&](auto& ghost)
                                      {
                                          assert(ownership[static_cast<std::size_t>(ghost.index)] == UNSET);
                                          ownership[static_cast<std::size_t>(ghost.index)] = neighbour.rank;
                                      });
                    }
                }

                // For the remaining cells and ghosts, if the reference intersects with the reference of a neighbour,
                // then the smallest rank owns the intersecting cells/ghosts
                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    int min_rank = std::min(rank, neighbour.rank);

                    for (std::size_t level = min_level; level <= max_level; ++level)
                    {
                        auto intersecting_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level],
                                                                          neighbour.mesh[mesh_id_t::reference][level]);
                        for_each_cell(mesh,
                                      intersecting_cells_and_ghosts,
                                      [&](auto& cell)
                                      {
                                          auto cell_index = static_cast<std::size_t>(cell.index);
                                          if (ownership[cell_index] == UNSET)
                                          {
                                              ownership[cell_index] = min_rank;
                                          }
                                      });
                    }
                }

                // Finally, the cells and ghosts that are not owned by any rank (ownership == UNSET) are owned by the current rank.
                for_each_cell(mesh[mesh_id_t::reference],
                              [&](auto& cell)
                              {
                                  auto cell_index = static_cast<std::size_t>(cell.index);
                                  if (ownership[cell_index] == UNSET)
                                  {
                                      ownership[cell_index] = rank;
                                  }
                              });

                n_owned_cells = 0;
                for (std::size_t i = 0; i < ownership.size(); ++i)
                {
                    n_owned_cells += (ownership[i] == rank) ? 1 : 0;
                }

                // Renumbering of the cells
                cell_indices.resize(n_cells);
                PetscInt new_cell_index = 0;
                for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index)
                {
                    if (ownership[cell_index] == rank)
                    {
                        cell_indices[cell_index] = new_cell_index++;
                    }
                }
                new_cell_index = 0;
                for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index)
                {
                    if (ownership[cell_index] != rank)
                    {
                        cell_indices[cell_index] = new_cell_index++;
                    }
                }
#endif
            }

            template <int n_unknowns_per_cell, typename return_type = std::size_t>
            inline return_type unknown_index(PetscInt shift, std::size_t cell_index, [[maybe_unused]] int component_index) const
            {
#ifdef SAMURAI_WITH_MPI
                cell_index = static_cast<std::size_t>(cell_indices[cell_index]);
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

            std::size_t min_level = mesh[mesh_id_t::reference].min_level();
            std::size_t max_level = mesh[mesh_id_t::reference].max_level();

            auto& ownership      = numbering.ownership;
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

            auto n_owned_unknowns = numbering.n_owned_cells * static_cast<std::size_t>(n_unknowns_per_cell);
            assert(local_indices.size() - static_cast<std::size_t>(block_shift_owned) >= n_owned_unknowns);
            auto owned_local_index_begin  = local_indices.begin() + static_cast<std::ptrdiff_t>(block_shift_owned);
            auto owned_local_index_end    = owned_local_index_begin + static_cast<std::ptrdiff_t>(n_owned_unknowns);
            auto owned_global_index_begin = global_indices.begin() + static_cast<std::ptrdiff_t>(block_shift_owned);
            auto owned_global_index_end   = owned_global_index_begin + static_cast<std::ptrdiff_t>(n_owned_unknowns);
            std::fill(owned_local_index_begin, owned_local_index_end, UNSET);
            std::fill(owned_global_index_begin, owned_global_index_end, UNSET);

            auto n_ghost_unknowns = (numbering.n_cells - numbering.n_owned_cells) * static_cast<std::size_t>(n_unknowns_per_cell);
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
                if (ownership[cell_index] == rank)
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
                if (ownership[cell_index] != rank)
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
                                      if (ownership[cell_index] == rank)
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
                                      if (ownership[cell_index] == neighbour.rank)
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
                        if (ownership[cell_index] == rank)
                        {
                            std::cout << fmt::format("[{}]:          cell_index {} (owned by {}): CI{} L{} G{}\n",
                                                     world.rank(),
                                                     cell_index,
                                                     ownership[cell_index],
                                                     numbering.cell_indices[cell_index],
                                                     local_indices[owned_unknown_index(cell_index, i)],
                                                     global_indices[owned_unknown_index(cell_index, i)]);
                        }
                        else
                        {
                            std::cout << fmt::format("[{}]:          cell_index {} (owned by {}): CI{} L{} G{}\n",
                                                     world.rank(),
                                                     cell_index,
                                                     ownership[cell_index],
                                                     numbering.cell_indices[cell_index],
                                                     local_indices[ghost_unknown_index(cell_index, i)],
                                                     global_indices[ghost_unknown_index(cell_index, i)]);
                        }
                    }
                }
            }

            if (args::print_petsc_numbering)
            {
                auto ownership_field = make_scalar_field<int>("owner_rank", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    ownership_field[cell_index] = ownership[cell_index];
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
                    petsc_cell_indices_field[cell_index] = numbering.cell_indices[cell_index];
                }

                save(fs::current_path(),
                     "petsc_indices",
                     {true, true},
                     mesh,
                     ownership_field,
                     // global_indices_field,
                     // local_indices_field,
                     samurai_cell_indices_field,
                     petsc_cell_indices_field);
            }
        }
#endif
    }
}
