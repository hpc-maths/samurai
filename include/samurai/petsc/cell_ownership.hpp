#pragma once
#include "../algorithm.hpp"

namespace samurai
{
    namespace petsc
    {

        struct CellOwnership
        {
            std::size_t n_local_cells = 0; // owned cells + ghost cells
            std::size_t n_owned_cells = 0;
#ifdef SAMURAI_WITH_MPI
            // Owner rank of each cell in the local mesh
            std::vector<int> owner_rank;
            // Renumbering of the cells: first all the owned cells, then all the ghosts.
            // This is used to split the ordering of the unknowns (first all the owned unknowns, then all the ghosts).
            // Note that the cell numbers start at index 0, and the ghost numbers also start at index 0!
            std::vector<int> cell_indices;
#endif

            template <class Mesh>
            void compute(const Mesh& mesh)
            {
                n_local_cells = mesh.nb_cells();
#ifndef SAMURAI_WITH_MPI
                n_owned_cells = n_local_cells;
#else
                using mesh_id_t = typename Mesh::mesh_id_t;

                mpi::communicator world;
                int rank = world.rank();

                std::size_t min_level = mesh[mesh_id_t::reference].min_level();
                std::size_t max_level = mesh[mesh_id_t::reference].max_level();

                // auto& owner_rank = numbering.owner_rank;

                constexpr int UNSET = -1;

                // Stores the owner rank of each cell
                owner_rank.resize(n_local_cells);
                std::fill(owner_rank.begin(), owner_rank.end(), UNSET);

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    // All cells and ghosts that intersect with the subdomain are owned by the current rank
                    auto local_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level], self(mesh.subdomain()).on(level));
                    for_each_cell(mesh,
                                  local_cells_and_ghosts,
                                  [&](auto& cell)
                                  {
                                      owner_rank[static_cast<std::size_t>(cell.index)] = rank;
                                  });

                    // All the boundary ghosts of locally owned boundary cells are also owned by the local rank
                    auto domain_bdry_outer_layer = domain_boundary_outer_layer(mesh, level, Mesh::config::ghost_width);
                    auto boundary_ghosts         = intersection(domain_bdry_outer_layer, mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  boundary_ghosts,
                                  [&](auto& ghost)
                                  {
                                      owner_rank[static_cast<std::size_t>(ghost.index)] = rank;
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
                                          // assert(owner_rank[static_cast<std::size_t>(cell.index)] == UNSET);
                                          if (owner_rank[static_cast<std::size_t>(cell.index)] == UNSET)
                                          {
                                              owner_rank[static_cast<std::size_t>(cell.index)] = neighbour.rank;
                                          }
                                          else
                                          {
                                              std::cout << fmt::format("rank {}: cell {} on level {} owned by {}\n",
                                                                       rank,
                                                                       cell.index,
                                                                       level,
                                                                       owner_rank[static_cast<std::size_t>(cell.index)]);
                                              owner_rank[static_cast<std::size_t>(cell.index)] = 10;
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
                                          assert(owner_rank[static_cast<std::size_t>(ghost.index)] == UNSET);
                                          owner_rank[static_cast<std::size_t>(ghost.index)] = neighbour.rank;
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
                                          if (owner_rank[cell_index] == UNSET)
                                          {
                                              owner_rank[cell_index] = min_rank;
                                          }
                                      });
                    }
                }

                // Finally, the cells and ghosts that are not owned by any rank (owner_rank == UNSET) are owned by the current rank.
                for_each_cell(mesh[mesh_id_t::reference],
                              [&](auto& cell)
                              {
                                  auto cell_index = static_cast<std::size_t>(cell.index);
                                  if (owner_rank[cell_index] == UNSET)
                                  {
                                      owner_rank[cell_index] = rank;
                                  }
                              });

                n_owned_cells = 0;
                for (std::size_t i = 0; i < owner_rank.size(); ++i)
                {
                    n_owned_cells += (owner_rank[i] == rank) ? 1 : 0;
                }

                // Renumbering of the cells
                cell_indices.resize(n_local_cells);
                PetscInt new_cell_index = 0;
                for (std::size_t cell_index = 0; cell_index < n_local_cells; ++cell_index)
                {
                    if (owner_rank[cell_index] == rank)
                    {
                        cell_indices[cell_index] = new_cell_index++;
                    }
                }
                new_cell_index = 0;
                for (std::size_t cell_index = 0; cell_index < n_local_cells; ++cell_index)
                {
                    if (owner_rank[cell_index] != rank)
                    {
                        cell_indices[cell_index] = new_cell_index++;
                    }
                }
#endif
            }
        };

    }
}
