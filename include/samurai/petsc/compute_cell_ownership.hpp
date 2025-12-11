#pragma once
#include "../algorithm.hpp"
#include "../arguments.hpp"
#include "../field.hpp"
#include "../io/hdf5.hpp"

namespace samurai
{
    namespace petsc
    {

        struct MismatchInfo
        {
            std::size_t cell_index;
            std::size_t level;
            std::vector<int> possible_owners;
            int owner_rank;

            std::size_t cell_index_on_neighbour;
            int owner_rank_on_neighbour;

            template <class Archive>
            void serialize(Archive& ar, const unsigned int /*version*/)
            {
                ar & cell_index;
                ar & level;
                ar & possible_owners;
                ar & owner_rank;
                ar & cell_index_on_neighbour;
                ar & owner_rank_on_neighbour;
            }
        };

        template <class Mesh>
        void save_numbering(const Mesh& mesh)
        {
            auto owner_rank_field = make_scalar_field<int>("owner_rank", mesh);
            for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
            {
                owner_rank_field[cell_index] = mesh.cell_ownership().owner_rank[cell_index];
            }

            auto samurai_cell_indices_field = make_scalar_field<std::size_t>("samurai_cell_index", mesh);
            for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
            {
                samurai_cell_indices_field[cell_index] = static_cast<std::size_t>(cell_index);
            }

            auto petsc_cell_indices_field = make_scalar_field<int>("petsc_cell_index", mesh);
            for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
            {
                petsc_cell_indices_field[cell_index] = mesh.cell_ownership().cell_indices[cell_index];
            }

            save(fs::current_path(), "petsc_indices", {true, true}, mesh, owner_rank_field, samurai_cell_indices_field, petsc_cell_indices_field);
            std::cout << "PETSc numbering saved to 'petsc_indices.xdmf'." << std::endl;
        }

        template <class Mesh>
        void compute_cell_ownership(Mesh& mesh)
        {
            auto& ownership = mesh.cell_ownership();
            if (ownership.is_computed)
            {
                return;
            }

            auto& n_owned_cells = ownership.n_owned_cells;
            auto& n_local_cells = ownership.n_local_cells;

            n_local_cells = mesh.nb_cells();
#ifndef SAMURAI_WITH_MPI
            n_owned_cells = n_local_cells;
#else
            std::vector<int>& owner_rank   = ownership.owner_rank;
            std::vector<int>& cell_indices = ownership.cell_indices;

            using mesh_id_t  = typename Mesh::mesh_id_t;
            using lca_t      = typename Mesh::lca_type;
            using interval_t = typename Mesh::interval_t;
            using index_t    = xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<Mesh::dim - 1>>;

            mpi::communicator world;
            int rank = world.rank();

            std::size_t min_level = mesh[mesh_id_t::reference].min_level();
            std::size_t max_level = mesh[mesh_id_t::reference].max_level();

            constexpr int UNSET = std::numeric_limits<int>::max();

            // Stores the owner rank of each cell
            owner_rank.resize(n_local_cells);
            std::fill(owner_rank.begin(), owner_rank.end(), UNSET);

            // Local cells are locally owned.
            for_each_cell(mesh,
                          [&](auto& cell)
                          {
                              owner_rank[static_cast<std::size_t>(cell.index)] = rank;
                          });

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                // All the boundary ghosts of locally owned boundary cells are also locally owned.
                auto domain_bdry_outer_layer = domain_boundary_outer_layer(mesh, level, Mesh::config::ghost_width);
                auto boundary_ghosts         = intersection(domain_bdry_outer_layer, mesh[mesh_id_t::reference][level]);
                for_each_cell(mesh,
                              boundary_ghosts,
                              [&](auto& ghost)
                              {
                                  owner_rank[static_cast<std::size_t>(ghost.index)] = rank;
                              });

                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    // Ghosts that corresponds to neighbour's cells are owned by that neighbour.
                    auto neighbour_cells = intersection(neighbour.mesh[mesh_id_t::cells][level], mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  neighbour_cells,
                                  [&](auto& cell)
                                  {
                                      owner_rank[static_cast<std::size_t>(cell.index)] = neighbour.rank;
                                  });

                    // Boundary ghosts associated with neighbour's boundary cells are also owned by that neighbour
                    auto nghb_domain_bdry_outer_layer = domain_boundary_outer_layer(neighbour.mesh, level, Mesh::config::ghost_width);
                    auto nghb_boundary_ghosts         = intersection(nghb_domain_bdry_outer_layer, mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  nghb_boundary_ghosts,
                                  [&](auto& ghost)
                                  {
                                      owner_rank[static_cast<std::size_t>(ghost.index)] = neighbour.rank;
                                  });
                }
            }

            // Boundary ghosts at lower levels might still be unset (if they have no children in this rank).
            // This can happen if a coarse ghost is outside the boundary owned by another subdomain.
            // I think this can't happen unless the resolution is sufficiently small, so that coarse ghosts of one subdomain
            // goes all the way to the domain boundary of the other subdomain without having its boundary cells as ghosts.
            for (std::size_t level = min_level; level <= max_level - 1; ++level)
            {
                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    auto nghb_domain_bdry_outer_layer = domain_boundary_outer_layer(neighbour.mesh, level + 1, Mesh::config::ghost_width);
                    auto nghb_domain_bdry_outer_layer_no_children = difference(nghb_domain_bdry_outer_layer,
                                                                               mesh[mesh_id_t::reference][level + 1]);
                    auto nghb_boundary_ghosts = intersection(nghb_domain_bdry_outer_layer_no_children, mesh[mesh_id_t::reference][level])
                                                    .on(level);
                    for_each_cell(mesh,
                                  nghb_boundary_ghosts,
                                  [&](auto& ghost)
                                  {
                                      assert(owner_rank[static_cast<std::size_t>(ghost.index)] == UNSET);
                                      if (owner_rank[static_cast<std::size_t>(ghost.index)] == UNSET)
                                      {
                                          owner_rank[static_cast<std::size_t>(ghost.index)] = neighbour.rank;
                                      }
                                  });
                }
            }

            // The projection ghosts are owned by the minimum rank of their children
            for (std::size_t level = max_level - 1; level >= min_level; --level)
            {
                lca_t proj_ghost_lca(level, mesh.origin_point(), mesh.scaling_factor());
                index_t index_ghost;

                auto projection_ghosts = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::reference][level + 1]).on(level);
                for_each_cell(mesh,
                              projection_ghosts,
                              [&](auto& ghost)
                              {
                                  auto& ghost_owner_rank = owner_rank.at(static_cast<std::size_t>(ghost.index));
                                  if (ghost_owner_rank == UNSET)
                                  {
                                      index_ghost = xt::view(ghost.indices, xt::range(1, Mesh::dim));
                                      proj_ghost_lca.add_point_back(ghost.indices[0], index_ghost);
                                      auto children = intersection(self(proj_ghost_lca).on(level + 1), mesh[mesh_id_t::reference][level + 1]);
                                      for_each_cell(mesh,
                                                    children,
                                                    [&](auto& child)
                                                    {
                                                        ghost_owner_rank = std::min(ghost_owner_rank,
                                                                                    owner_rank.at(static_cast<std::size_t>(child.index)));
                                                    });
                                      proj_ghost_lca.clear();
                                  }
                              });

                // For the projection ghosts that have no children in this rank, we look for children in neighbour ranks
                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    auto projection_ghosts_nghb = intersection(mesh[mesh_id_t::reference][level],
                                                               difference(neighbour.mesh[mesh_id_t::reference][level + 1],
                                                                          mesh[mesh_id_t::reference][level + 1]))
                                                      .on(level);
                    for_each_cell(mesh,
                                  projection_ghosts_nghb,
                                  [&](auto& ghost)
                                  {
                                      auto& ghost_owner_rank = owner_rank.at(static_cast<std::size_t>(ghost.index));
                                      if (ghost_owner_rank == UNSET)
                                      {
                                          ghost_owner_rank = neighbour.rank;
                                      }
                                  });
                }

                if (level == 0)
                {
                    break;
                }
            }

            // The prediction ghosts are owned by the rank of their parent
            for (std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                lca_t pred_ghost_lca(level, mesh.origin_point(), mesh.scaling_factor());
                index_t index_ghost;

                auto prediction_ghosts = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::reference][level - 1]).on(level);
                for_each_cell(
                    mesh,
                    prediction_ghosts,
                    [&](auto& ghost)
                    {
                        auto& ghost_owner_rank = owner_rank[static_cast<std::size_t>(ghost.index)];
                        if (ghost_owner_rank == UNSET)
                        {
                            index_ghost = xt::view(ghost.indices, xt::range(1, Mesh::dim));
                            pred_ghost_lca.add_point_back(ghost.indices[0], index_ghost);
                            auto parent_set = intersection(self(pred_ghost_lca).on(level - 1), mesh[mesh_id_t::reference][level - 1]);
                            for_each_cell(mesh,
                                          parent_set,
                                          [&](auto& parent)
                                          {
                                              assert(ghost_owner_rank == UNSET && "there must be only one parent");
                                              ghost_owner_rank = owner_rank[static_cast<std::size_t>(parent.index)];
                                          });
                            pred_ghost_lca.clear();
                        }
                    });
            }

            // The remaining shared ghosts are owned by the rank whose subdomain's gravity center is the closest.
            // It can typpically handle the outer corners and other remaining boundary ghosts.
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    auto intersecting_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level],
                                                                      neighbour.mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  intersecting_cells_and_ghosts,
                                  [&](auto& cell)
                                  {
                                      if (owner_rank[static_cast<std::size_t>(cell.index)] == UNSET)
                                      {
                                          auto diff     = cell.center() - mesh.gravity_center();
                                          auto distance = std::sqrt(samurai::math::sum(diff * diff));

                                          auto nghb_diff     = cell.center() - neighbour.mesh.gravity_center();
                                          auto nghb_distance = std::sqrt(samurai::math::sum(nghb_diff * nghb_diff));

                                          owner_rank[static_cast<std::size_t>(cell.index)] = distance < nghb_distance ? rank : neighbour.rank;
                                      }
                                  });
                }
            }

            // Finally, the ghosts that are not owned by any rank at this point are owned by the current rank.
            for (std::size_t cell_index = 0; cell_index < n_local_cells; ++cell_index)
            {
                if (owner_rank[cell_index] == UNSET)
                {
                    owner_rank[cell_index] = rank;
                }
            }

            //--------------------------------------------//
            // Look for owner mismatches and correct them //
            //--------------------------------------------//
            // An owner mismatch is when two neighbouring ranks disagree on the owner of a cell.
            // Mismatches can happen when two neighbouring ranks have intersecting ghosts although they are not registered in the
            // neighbourhood of each other.

            // We register, for all cells, the possible owners (neighbour ranks that also have this cell).
            // This will be used to choose another owner in case of mismatch.
            std::vector<std::vector<int>> possible_owners(n_local_cells);

            for (std::size_t cell_index = 0; cell_index < n_local_cells; ++cell_index)
            {
                possible_owners[cell_index].push_back(rank);
            }
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    auto intersecting_cells_and_ghosts = intersection(mesh[mesh_id_t::reference][level],
                                                                      neighbour.mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  intersecting_cells_and_ghosts,
                                  [&](auto& cell)
                                  {
                                      possible_owners[static_cast<std::size_t>(cell.index)].push_back(neighbour.rank);
                                  });
                }
            }

            // The process of checking for mismatches and correcting them is repeated until no mismatch is found.
            int n_mismatch_checks         = 0;
            const int max_mismatch_checks = world.size() + 2; // arbitrary limit
            bool owner_mismatch           = false;
            std::vector<std::vector<MismatchInfo>> mismatches(mesh.mpi_neighbourhood().size());
            bool unsolvable_mismatch_found = false;

            while (n_mismatch_checks < max_mismatch_checks && !unsolvable_mismatch_found)
            {
                for (auto& m : mismatches)
                {
                    m.clear();
                }
                owner_mismatch = false;
                n_mismatch_checks++;

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
                                          to_send.push_back(owner_rank[static_cast<std::size_t>(cell.index)]);
                                          to_send.push_back(static_cast<PetscInt>(cell.index));
                                      });
                    }
                    req.push_back(world.isend(neighbour.rank /* dest */, neighbour.rank /* tag */, to_send));
                    i_neigh++;
                }

                // RECEIVE
                std::size_t i_neighbour = 0;
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
                                          auto neighbour_owner_rank    = to_recv[read++];
                                          auto cell_index_on_neighbour = to_recv[read++];
                                          // MISMATCH DETECTED!!!
                                          if (owner_rank[static_cast<std::size_t>(cell.index)] != neighbour_owner_rank)
                                          {
                                              owner_mismatch = true;
                                              mismatches[i_neighbour].emplace_back(static_cast<std::size_t>(cell.index),
                                                                                   level,
                                                                                   possible_owners[static_cast<std::size_t>(cell.index)],
                                                                                   owner_rank[static_cast<std::size_t>(cell.index)],
                                                                                   cell_index_on_neighbour,
                                                                                   neighbour_owner_rank);

                                              // std::cout << fmt::format(
                                              //     "[{}] Error: owner mismatch in cell {} on level {} (owned here by {} != {} on [{}] with
                                              //     cell_index {})\n", rank, cell.index, level,
                                              //     owner_rank[static_cast<std::size_t>(cell.index)],
                                              //     neighbour_owner_rank,
                                              //     neighbour.rank,
                                              //     cell_index_on_neighbour);
                                              // owner_rank[static_cast<std::size_t>(cell.index)] = world.size() * 2; // mark error
                                          }
                                      });
                    }
                    i_neighbour++;
                }

                mpi::wait_all(req.begin(), req.end());

                owner_mismatch = mpi::all_reduce(world, owner_mismatch, std::logical_or());
                if (!owner_mismatch)
                {
                    // No more mismatches, get out
                    break;
                }
                // else // export for analysis
                // {
                //     auto owner_rank_field = make_scalar_field<int>("owner_rank", mesh);
                //     for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                //     {
                //         owner_rank_field[cell_index] = owner_rank[cell_index];
                //     }
                //     auto samurai_cell_indices_field = make_scalar_field<std::size_t>("samurai_cell_index", mesh);
                //     for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                //     {
                //         samurai_cell_indices_field[cell_index] = static_cast<std::size_t>(cell_index);
                //     }
                //     save(fs::current_path(), "owner_mismatch", {true, true}, mesh, owner_rank_field, samurai_cell_indices_field);

                //     std::cerr << "Cell ownership mismatch detected. Exiting." << std::endl;
                //     exit(EXIT_FAILURE);
                // }

                // Send mismatches to neighbours
                req.clear();
                i_neighbour = 0;
                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    req.push_back(world.isend(neighbour.rank /* dest */, neighbour.rank /* tag */, mismatches[i_neighbour]));
                    i_neighbour++;
                }

                // Receive mismatches from neighbours and change owner ranks if needed
                i_neighbour = 0;
                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    std::vector<MismatchInfo> neighbour_mismatches;
                    world.recv(neighbour.rank /* source */, rank /* tag */, neighbour_mismatches);
                    for (auto& neighbour_mismatch : neighbour_mismatches)
                    {
                        // If the owner_rank I have chosen is not in the possible_owners of the neighbour, we must choose another one that
                        // is possible for both.
                        auto& my_cell_index = neighbour_mismatch.cell_index_on_neighbour;
                        if (std::find(neighbour_mismatch.possible_owners.begin(),
                                      neighbour_mismatch.possible_owners.end(),
                                      owner_rank[my_cell_index])
                            == neighbour_mismatch.possible_owners.end())
                        {
                            // Find the intersection between the neighbour's possible_owners and my possible_owners
                            std::vector<int> intersection;
                            for (auto& neighbour_possible_owner : neighbour_mismatch.possible_owners)
                            {
                                if (std::find(possible_owners[my_cell_index].begin(),
                                              possible_owners[my_cell_index].end(),
                                              neighbour_possible_owner)
                                    != possible_owners[my_cell_index].end())
                                {
                                    intersection.push_back(neighbour_possible_owner);
                                }
                            }
                            if (intersection.empty())
                            {
                                std::cerr << fmt::format(
                                    "[{}] Error: cannot resolve ownership mismatch for cell {} (cell_index {} in rank [{}]). ",
                                    rank,
                                    my_cell_index,
                                    neighbour_mismatch.cell_index,
                                    neighbour.rank);
                                std::cerr << fmt::format("[{}] Possible owners: [{}] --> ({}), [{}] --> ({}).\n",
                                                         rank,
                                                         rank,
                                                         fmt::join(possible_owners[my_cell_index], ", "),
                                                         neighbour.rank,
                                                         fmt::join(neighbour_mismatch.possible_owners, ", "));
                                unsolvable_mismatch_found = true;
                                break;
                            }
                            // Update possible_owners to the intersection
                            possible_owners[my_cell_index] = intersection;
                            // Choose the minimum rank in the intersection as the new owner
                            owner_rank[my_cell_index] = *std::min_element(possible_owners[my_cell_index].begin(),
                                                                          possible_owners[my_cell_index].end());
                        }
                        // else, the owner rank is acceptable for both ranks, we don't change it
                    }
                    neighbour_mismatches.clear();
                    i_neighbour++;
                }
                mpi::wait_all(req.begin(), req.end());
            }

            // Print the mismatches
            if ((n_mismatch_checks == max_mismatch_checks || unsolvable_mismatch_found) && owner_mismatch)
            {
                auto owner_rank_field = make_scalar_field<int>("owner_rank", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    owner_rank_field[cell_index] = owner_rank[cell_index];
                }
                std::size_t i_neigh = 0;
                for (auto& neighbour : mesh.mpi_neighbourhood())
                {
                    for (auto& mismatch : mismatches[i_neigh])
                    {
                        std::cout << fmt::format(
                            "[{}] Mismatch for cell {} at level {} (owned here by {} and owned by {} on [{}] with cell_index {})\n",
                            rank,
                            mismatch.cell_index,
                            mismatch.level,
                            owner_rank[mismatch.cell_index],
                            mismatch.owner_rank_on_neighbour,
                            neighbour.rank,
                            mismatch.cell_index_on_neighbour);
                        owner_rank_field[mismatch.cell_index] = 2 * world.size(); // mark error
                    }
                    i_neigh++;
                }

                auto samurai_cell_indices_field = make_scalar_field<std::size_t>("samurai_cell_index", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    samurai_cell_indices_field[cell_index] = static_cast<std::size_t>(cell_index);
                }
                save(fs::current_path(), "owner_mismatch", {true, true}, mesh, owner_rank_field, samurai_cell_indices_field);
                std::cerr << fmt::format("[{}] Error: cell ownership mismatch detected. ", rank);
                if (!unsolvable_mismatch_found)
                {
                    std::cerr << fmt::format("Maximum number of correction passes reached ({}). ", max_mismatch_checks);
                }
                std::cerr << fmt::format("See 'owner_mismatch.xdmf' for details.\n", rank);
                std::cerr << fmt::format(
                    "[{}] This usually happens when low level ghosts are shared between subdomains that are not in each other's direct neighbourhood. To solve this issue, try increasing the min level.\n",
                    rank);
                exit(EXIT_FAILURE);
            }

            //--------------------------//
            // Renumbering of the cells //
            //--------------------------//

            n_owned_cells = 0;
            for (std::size_t i = 0; i < owner_rank.size(); ++i)
            {
                n_owned_cells += (owner_rank[i] == rank) ? 1U : 0;
            }

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

            if (args::print_petsc_numbering)
            {
                sleep(static_cast<unsigned int>(rank));
                std::cout << fmt::format("[{}]: Cell ownership: owned: {}, total: {}\n", world.rank(), n_owned_cells, n_local_cells);
                for_each_cell(mesh[mesh_id_t::reference],
                              [&](auto& cell)
                              {
                                  std::cout << fmt::format("[{}]:          cell_index {} level {} (owned by {}): CI{}\n",
                                                           world.rank(),
                                                           cell.index,
                                                           cell.level,
                                                           owner_rank[static_cast<std::size_t>(cell.index)],
                                                           cell_indices[static_cast<std::size_t>(cell.index)]);
                              });
                save_numbering(mesh);
            }
#endif
            ownership.is_computed = true;
        }

    }
}
