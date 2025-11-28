#pragma once
#include "../algorithm.hpp"
#include "../arguments.hpp"
#include "../field.hpp"
#include "../io/hdf5.hpp"

namespace samurai
{
    namespace petsc
    {

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

#ifndef NDEBUG
            //---------------------------------------//
            // Check that there is no owner mismatch //
            //---------------------------------------//

            bool owner_mismatch = false;

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
                                      auto neighbour_owner_rank = to_recv[read++];
                                      // assert(owner_rank[static_cast<std::size_t>(cell.index)] == neighbour_owner_rank);
                                      if (owner_rank[static_cast<std::size_t>(cell.index)] != neighbour_owner_rank)
                                      {
                                          owner_mismatch = true;
                                          std::cout << fmt::format(
                                              "[{}] Error: owner mismatch in cell {} on level {} (owned by {} != {} on [{}])\n",
                                              world.rank(),
                                              cell.index,
                                              level,
                                              owner_rank[static_cast<std::size_t>(cell.index)],
                                              neighbour_owner_rank,
                                              neighbour.rank);
                                          owner_rank[static_cast<std::size_t>(cell.index)] = world.size() * 2; // mark error
                                      }
                                  });
                }
            }

            mpi::wait_all(req.begin(), req.end());

            if (owner_mismatch)
            {
                auto owner_rank_field = make_scalar_field<int>("owner_rank", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    owner_rank_field[cell_index] = owner_rank[cell_index];
                }
                auto samurai_cell_indices_field = make_scalar_field<std::size_t>("samurai_cell_index", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    samurai_cell_indices_field[cell_index] = static_cast<std::size_t>(cell_index);
                }
                save(fs::current_path(), "owner_mismatch", {true, true}, mesh, owner_rank_field, samurai_cell_indices_field);

                std::cerr << "Cell ownership mismatch detected. Exiting." << std::endl;
                exit(EXIT_FAILURE);
            }
#endif

            //--------------------------//
            // Renumbering of the cells //
            //--------------------------//

            n_owned_cells = 0;
            for (std::size_t i = 0; i < owner_rank.size(); ++i)
            {
                n_owned_cells += (owner_rank[i] == rank) ? 1 : 0;
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

            ownership.is_computed = true;
#endif
        }

    }
}
