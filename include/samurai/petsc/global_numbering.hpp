#pragma once
#include "../field.hpp"
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {
        template <class Mesh>
        void compute_global_numbering(const Mesh& mesh,
                                      std::vector<PetscInt>& local_indices,
                                      std::vector<PetscInt>& global_indices,
                                      std::vector<int>& ownership)
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            mpi::communicator world;
            int rank = world.rank();

            std::size_t min_level = mesh[mesh_id_t::reference].min_level();
            std::size_t max_level = mesh[mesh_id_t::reference].max_level();

            //------------------------------------------//
            // Affect ownership to the cells and ghosts //
            //------------------------------------------//

            // stores the owner rank of each cell
            ownership.resize(mesh.nb_cells());
            std::fill(ownership.begin(), ownership.end(), -1); // -1 means not owned by any rank

            // All cells and ghosts that intersect with the subdomain are owned by the current rank
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto local_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level], self(mesh.subdomain()).on(level));
                // local_cells_and_ghosts(
                //     [&](const auto& i, const auto& index)
                //     {
                //         ownership(level, i, index) = rank;
                //     });
                for_each_cell(mesh,
                              local_cells_and_ghosts,
                              [&](auto& cell)
                              {
                                  ownership[static_cast<std::size_t>(cell.index)] = rank;
                              });
            }

            for (auto& neighbour : mesh.mpi_neighbourhood())
            {
                int min_rank = std::min(rank, neighbour.rank);

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    // Cells and ghosts that intersect with a neighbour subdomain are owned by the neighbour rank
                    auto neighbour_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level],
                                                                   self(neighbour.mesh.subdomain()).on(level));
                    // neighbour_cells_and_ghosts(
                    //     [&](const auto& i, const auto& index)
                    //     {
                    //         ownership(level, i, index) = neighbour.rank;
                    //     });
                    for_each_cell(mesh,
                                  neighbour_cells_and_ghosts,
                                  [&](auto& cell)
                                  {
                                      ownership[static_cast<std::size_t>(cell.index)] = neighbour.rank;
                                  });

                    // For the remaining cells and ghosts, if the reference intersects with the reference of a neighbour,
                    // then the smallest rank owns the intersecting cells/ghosts
                    auto intersecting_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level],
                                                                      neighbour.mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  intersecting_cells_and_ghosts,
                                  [&](auto& cell)
                                  {
                                      auto cell_index = static_cast<std::size_t>(cell.index);
                                      if (ownership[cell_index] == -1)
                                      {
                                          ownership[cell_index] = min_rank;
                                      }
                                  });
                }
            }

            // Finally, the cells and ghosts that are not owned by any rank (ownership == -1) are owned by the current rank.
            // This typically concerns the boundary ghosts.
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              auto cell_index = static_cast<std::size_t>(cell.index);
                              if (ownership[cell_index] == -1)
                              {
                                  ownership[cell_index] = rank;
                              }
                          });

            //----------------------------------//
            // Compute the local/global indices //
            //----------------------------------//

            local_indices.resize(mesh.nb_cells());
            global_indices.resize(mesh.nb_cells());
            std::fill(local_indices.begin(), local_indices.end(), -1);   // -1 means that the local index has not been set
            std::fill(global_indices.begin(), global_indices.end(), -1); // -1 means that the global index has not been set

            // 1. Count the number of local DOFs
            PetscInt n_local_dofs = 0;
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              n_local_dofs += (ownership[static_cast<std::size_t>(cell.index)] == rank) ? 1 : 0;
                          });

            // 2. Sums up the DOFs owned by the previous MPI processes to get the global offset, i.e. where to start counting the global
            // indices on the current rank

            PetscInt offset = 0;
            MPI_Exscan(&n_local_dofs, &offset, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

            std::cout << fmt::format("rank {}: n_local_dofs = {}, offset = {}\n", rank, n_local_dofs, offset);

            // 3. Compute local/global indices of the locally owned cells and ghosts
            PetscInt local_index  = 0;
            PetscInt global_index = offset;
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              auto cell_index = static_cast<std::size_t>(cell.index);
                              if (ownership[cell_index] == rank)
                              {
                                  local_indices[cell_index]  = local_index++;
                                  global_indices[cell_index] = global_index++;
                              }
                          });
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              auto cell_index = static_cast<std::size_t>(cell.index);
                              if (ownership[cell_index] != rank)
                              {
                                  local_indices[cell_index] = local_index++;
                              }
                          });

            // 4. Exchange global indices of the local cells/ghosts with neighbouring MPI processes

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
                                          to_send.push_back(global_indices[cell_index]);
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
                                          assert(global_indices[cell_index] == -1);
                                          global_indices[cell_index] = to_recv[read++];
                                      }
                                  });
                }
            }

            mpi::wait_all(req.begin(), req.end());

            if (rank == 1)
            {
                sleep(1);
            }

            for (std::size_t i = 0; i < global_indices.size(); ++i)
            {
                // assert(m_global_indices[i] != -1);
                std::cout << fmt::format("[{}]: cell_index {} (owned by {}): L{} G{}\n",
                                         world.rank(),
                                         i,
                                         ownership[i],
                                         local_indices[i],
                                         global_indices[i]);
            }

            auto ownership_field = make_scalar_field<int>("owner_rank", mesh);
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              ownership_field[cell] = ownership[static_cast<std::size_t>(cell.index)];
                          });

            auto global_indices_field = make_scalar_field<PetscInt>("petsc_global_index", mesh);
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              global_indices_field[cell] = global_indices[static_cast<std::size_t>(cell.index)];
                          });

            auto local_indices_field = make_scalar_field<PetscInt>("petsc_local_index", mesh);
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              local_indices_field[cell] = local_indices[static_cast<std::size_t>(cell.index)];
                          });

            auto samurai_cell_indices_field = make_scalar_field<std::size_t>("samurai_cell_index", mesh);
            for_each_cell(mesh[mesh_id_t::reference],
                          [&](auto& cell)
                          {
                              samurai_cell_indices_field[cell] = static_cast<std::size_t>(cell.index);
                          });
            save(fs::current_path(),
                 "petsc_indices",
                 {true, true},
                 mesh,
                 ownership_field,
                 global_indices_field,
                 local_indices_field,
                 samurai_cell_indices_field);

            // exit(1);
        }
    }
}
