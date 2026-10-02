#pragma once
#include "../algorithm.hpp"
#include "../arguments.hpp"
#include "../field.hpp"
#include "../io/hdf5.hpp"

#include <limits>
#include <stdexcept>

namespace samurai
{
    namespace petsc
    {

        struct MismatchInfo
        {
            std::size_t cell_index;
            std::size_t level;
            int owner_rank;
            int rule; ///< which ownership rule decided, here

            std::size_t cell_index_on_neighbour;
            int owner_rank_on_neighbour;
            int rule_on_neighbour;

            template <class Archive>
            void serialize(Archive& ar, const unsigned int /*version*/)
            {
                ar & cell_index;
                ar & level;
                ar & owner_rank;
                ar & rule;
                ar & cell_index_on_neighbour;
                ar & owner_rank_on_neighbour;
                ar & rule_on_neighbour;
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

        /**
         * Decide, for every cell this rank holds, which rank owns its unknown.
         *
         * The unknowns of the petsc system are the cells of the reference mesh of every rank, and
         * a cell held by several ranks must have exactly one owner: the rank that assembles its
         * row and in whose index range it lives. The exchange that gives a ghost its global index
         * is positional, so every rank holding a cell has to reach the **same** owner **without
         * asking anyone**: the rule below is a function of data every holder has, and it is
         * evaluated identically everywhere. A single check pass then verifies that neighbours
         * agree, and a disagreement is an error, not something to negotiate - it means two ranks
         * hold a common cell without being registered as neighbours, which is the one situation
         * in which they cannot have the same data.
         *
         * The rules, in increasing order of precedence. The principle behind them: a row is
         * assembled by the rank that *classifies* the cell - as a real cell, a projection ghost,
         * a prediction ghost or a boundary ghost - in its **own** mesh, because that
         * classification is what the assembly visits (`for_each_projection_ghost`,
         * `for_each_prediction_ghost`, the boundary stencils of the real boundary cells) and
         * what guarantees that the rank holds every cell the row reads. So the owner has to be
         * one of the ranks that classify the cell, and among them the lowest rank.
         *
         *   4. **Any other cell** - one no rank's real cells give a meaning to, such as a coarse
         *      prediction margin far from every cell, or an outer corner - gets an identity row
         *      from its owner, so any consistent choice does: the holder whose subdomain's
         *      gravity centre is the closest, the tie going to the lowest rank. The holders are
         *      this rank and every neighbour whose reference mesh contains the cell.
         *   3. **A projection ghost** - a scheme ghost of some rank whose children are real cells
         *      of that rank - is owned by the lowest such rank; **a prediction ghost** - a scheme
         *      ghost of some rank whose parent is a real cell of that rank - likewise. The two
         *      cannot overlap: a cell with a real parent has no real children.
         *   2. **A real cell** is owned by the rank it belongs to.
         *   1. **An outer ghost the boundary conditions fill** - a cell outside the domain within
         *      @c ghost_width of a real boundary cell, in one Cartesian direction - is owned by the
         *      lowest rank owning such a boundary cell. The boundary-condition row of such a
         *      ghost is assembled while visiting the real boundary cells of the owning rank, so
         *      any other owner would leave the row empty and the condition silently dropped.
         *
         * Every input to these rules - the neighbours' cells, scheme ghosts, domain and gravity
         * centre - travels with the neighbourhood exchange, so every holder evaluates the same
         * function on the same data. What was here before decided ownership from local
         * heuristics - "the minimum rank of the children *this rank* holds", "the closer of the
         * two gravity centres, pairwise" - whose answers differ from rank to rank, and then
         * tried to repair the disagreements over a bounded number of correction passes. Pairwise
         * closeness is not transitive, so three ranks sharing a coarse cell could each name a
         * different owner and the passes never converged; the failure appeared as soon as the
         * coarse levels carried a wider prediction margin and more cells were shared.
         */
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

            using mesh_id_t = typename Mesh::mesh_id_t;

            mpi::communicator world;
            const int rank = world.rank();

            const std::size_t min_level = mesh[mesh_id_t::reference].min_level();
            const std::size_t max_level = mesh[mesh_id_t::reference].max_level();

            constexpr int UNSET = std::numeric_limits<int>::max();

            auto squared_distance = [](const auto& cell, const auto& centre)
            {
                auto diff = cell.center() - centre;
                return samurai::math::sum(diff * diff);
            };

            //---------------------------------------------------------//
            // Rule 4: the closest gravity centre among the holders    //
            //---------------------------------------------------------//
            // Evaluated first, on every cell, so that the other rules can simply overwrite it.
            // The comparison is a total order on (distance, rank), so the answer does not depend
            // on the order in which the holders are visited.

            owner_rank.assign(n_local_cells, rank);
            std::vector<int> rule(n_local_cells, 4); // the rule that decided each cell, for the mismatch report
            std::vector<double> best_distance(n_local_cells, std::numeric_limits<double>::infinity());

            for_each_cell(mesh[mesh_id_t::reference],
                          [&](const auto& cell)
                          {
                              best_distance[static_cast<std::size_t>(cell.index)] = squared_distance(cell, mesh.gravity_center());
                          });

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                for (const auto& neighbour : mesh.mpi_neighbourhood())
                {
                    auto shared = intersection(mesh[mesh_id_t::reference][level], neighbour.mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  shared,
                                  [&](const auto& cell)
                                  {
                                      const auto cell_index = static_cast<std::size_t>(cell.index);
                                      const double distance = squared_distance(cell, neighbour.mesh.gravity_center());
                                      if (distance < best_distance[cell_index]
                                          || (distance == best_distance[cell_index] && neighbour.rank < owner_rank[cell_index]))
                                      {
                                          best_distance[cell_index] = distance;
                                          owner_rank[cell_index]    = neighbour.rank;
                                      }
                                  });
                }
            }

            //-----------------------------------------------------------------//
            // Rule 3: projection and prediction ghosts follow the rank that   //
            // classifies them, the lowest one when several do                 //
            //-----------------------------------------------------------------//

            std::vector<int> classifier(n_local_cells, UNSET);

            auto claim_classified_ghosts = [&](const auto& holder_mesh, int holder_rank)
            {
                const auto& holder_cells  = holder_mesh[mesh_id_t::cells];
                const auto& holder_ghosts = holder_mesh[mesh_id_t::cells_and_ghosts];

                auto claim = [&](auto&& set)
                {
                    for_each_cell(mesh,
                                  set,
                                  [&](const auto& ghost)
                                  {
                                      auto& owner = classifier[static_cast<std::size_t>(ghost.index)];
                                      owner       = std::min(owner, holder_rank);
                                  });
                };

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (level < max_level)
                    {
                        auto projection_ghosts = intersection(mesh[mesh_id_t::reference][level], holder_ghosts[level], holder_cells[level + 1])
                                                     .on(level);
                        claim(projection_ghosts);
                    }
                    if (level > min_level)
                    {
                        auto prediction_ghosts = intersection(mesh[mesh_id_t::reference][level], holder_ghosts[level], holder_cells[level - 1])
                                                     .on(level);
                        claim(prediction_ghosts);
                    }
                }
            };

            claim_classified_ghosts(mesh, rank);
            for (const auto& neighbour : mesh.mpi_neighbourhood())
            {
                claim_classified_ghosts(neighbour.mesh, neighbour.rank);
            }

            for (std::size_t cell_index = 0; cell_index < n_local_cells; ++cell_index)
            {
                if (classifier[cell_index] != UNSET)
                {
                    owner_rank[cell_index] = classifier[cell_index];
                    rule[cell_index]       = 3;
                }
            }

            //--------------------//
            // Rule 2: real cells //
            //--------------------//

            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              owner_rank[static_cast<std::size_t>(cell.index)] = rank;
                              rule[static_cast<std::size_t>(cell.index)]       = 2;
                          });

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                for (const auto& neighbour : mesh.mpi_neighbourhood())
                {
                    auto neighbour_cells = intersection(neighbour.mesh[mesh_id_t::cells][level], mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  neighbour_cells,
                                  [&](const auto& cell)
                                  {
                                      owner_rank[static_cast<std::size_t>(cell.index)] = neighbour.rank;
                                      rule[static_cast<std::size_t>(cell.index)]       = 2;
                                  });
                }
            }

            //--------------------------------------------------------------//
            // Rule 1: the outer ghosts of a real boundary cell follow it   //
            //--------------------------------------------------------------//

            std::vector<int> boundary_owner(n_local_cells, UNSET);

            auto claim_boundary_ghosts = [&](const auto& holder_mesh, int holder_rank)
            {
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto outer_layer     = domain_boundary_outer_layer(holder_mesh, level, holder_mesh.ghost_width());
                    auto boundary_ghosts = intersection(outer_layer, mesh[mesh_id_t::reference][level]);
                    for_each_cell(mesh,
                                  boundary_ghosts,
                                  [&](const auto& ghost)
                                  {
                                      auto& owner = boundary_owner[static_cast<std::size_t>(ghost.index)];
                                      owner       = std::min(owner, holder_rank);
                                  });
                }
            };

            claim_boundary_ghosts(mesh, rank);
            for (const auto& neighbour : mesh.mpi_neighbourhood())
            {
                claim_boundary_ghosts(neighbour.mesh, neighbour.rank);
            }

            for (std::size_t cell_index = 0; cell_index < n_local_cells; ++cell_index)
            {
                if (boundary_owner[cell_index] != UNSET)
                {
                    owner_rank[cell_index] = boundary_owner[cell_index];
                    rule[cell_index]       = 1;
                }
            }

            //-----------------------------------//
            // Check that the neighbours agree   //
            //-----------------------------------//
            // The exchange is positional, so both sides must walk the same cells in the same
            // order: the same levels, and the intersection built with its operands in the same
            // order - derived from the rank numbers, not from which side is sending.

            auto shared_cells = [&](const auto& neighbour, std::size_t level)
            {
                const auto& mine  = mesh[mesh_id_t::reference][level];
                const auto& other = neighbour.mesh[mesh_id_t::reference][level];
                return rank < neighbour.rank ? intersection(mine, other) : intersection(other, mine);
            };

            // The level range comes from both meshes, not from this rank's: a rank whose
            // reference mesh starts at a coarser level than its neighbour's would otherwise
            // walk more levels than the neighbour does and read the values off by a level.
            auto shared_max_level = [&](const auto& neighbour)
            {
                return std::max(max_level, neighbour.mesh[mesh_id_t::reference].max_level());
            };

            std::vector<std::vector<MismatchInfo>> mismatches(mesh.mpi_neighbourhood().size());
            bool owner_mismatch = false;

            {
                std::vector<mpi::request> req;
                std::vector<std::vector<PetscInt>> to_send_by_neighbour(mesh.mpi_neighbourhood().size());
                std::size_t i_neighbour = 0;
                for (const auto& neighbour : mesh.mpi_neighbourhood())
                {
                    auto& to_send = to_send_by_neighbour[i_neighbour];
                    for (std::size_t level = 0; level <= shared_max_level(neighbour); ++level)
                    {
                        auto shared = shared_cells(neighbour, level);
                        for_each_cell(mesh,
                                      shared,
                                      [&](const auto& cell)
                                      {
                                          to_send.push_back(owner_rank[static_cast<std::size_t>(cell.index)]);
                                          to_send.push_back(static_cast<PetscInt>(cell.index));
                                          to_send.push_back(rule[static_cast<std::size_t>(cell.index)]);
                                          for (std::size_t d = 0; d < Mesh::dim; ++d)
                                          {
                                              to_send.push_back(static_cast<PetscInt>(cell.indices[d]));
                                          }
                                      });
                    }
                    req.push_back(world.isend(neighbour.rank /* dest */, neighbour.rank /* tag */, to_send));
                    i_neighbour++;
                }

                i_neighbour = 0;
                for (const auto& neighbour : mesh.mpi_neighbourhood())
                {
                    std::vector<PetscInt> to_recv;
                    std::size_t read = 0;
                    world.recv(neighbour.rank /* source */, rank /* tag */, to_recv);
                    for (std::size_t level = 0; level <= shared_max_level(neighbour); ++level)
                    {
                        auto shared = shared_cells(neighbour, level);
                        for_each_cell(mesh,
                                      shared,
                                      [&](const auto& cell)
                                      {
                                          const auto neighbour_owner_rank    = static_cast<int>(to_recv[read++]);
                                          const auto cell_index_on_neighbour = static_cast<std::size_t>(to_recv[read++]);
                                          const auto rule_on_neighbour       = static_cast<int>(to_recv[read++]);
                                          const auto cell_index              = static_cast<std::size_t>(cell.index);
                                          for (std::size_t d = 0; d < Mesh::dim; ++d)
                                          {
                                              const auto coord = to_recv[read++];
                                              if (coord != static_cast<PetscInt>(cell.indices[d]))
                                              {
                                                  throw std::runtime_error(fmt::format(
                                                      "[{}] compute_cell_ownership: the ownership exchange with rank {} is misaligned at "
                                                      "level {}: this rank walks the cell at ({}) where the neighbour sent its cell {} at "
                                                      "coordinate {} = {}. The two ranks do not walk the same shared cells.",
                                                      rank,
                                                      neighbour.rank,
                                                      level,
                                                      fmt::join(cell.indices, ", "),
                                                      cell_index_on_neighbour,
                                                      d,
                                                      coord));
                                              }
                                          }
                                          if (owner_rank[cell_index] != neighbour_owner_rank)
                                          {
                                              owner_mismatch = true;
                                              mismatches[i_neighbour].push_back({cell_index,
                                                                                 level,
                                                                                 owner_rank[cell_index],
                                                                                 rule[cell_index],
                                                                                 cell_index_on_neighbour,
                                                                                 neighbour_owner_rank,
                                                                                 rule_on_neighbour});
                                          }
                                      });
                    }
                    i_neighbour++;
                }

                mpi::wait_all(req.begin(), req.end());
            }

            owner_mismatch = mpi::all_reduce(world, owner_mismatch, std::logical_or());

            if (owner_mismatch)
            {
                auto owner_rank_field = make_scalar_field<int>("owner_rank", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    owner_rank_field[cell_index] = owner_rank[cell_index];
                }
                std::size_t i_neighbour = 0;
                for (const auto& neighbour : mesh.mpi_neighbourhood())
                {
                    for (const auto& mismatch : mismatches[i_neighbour])
                    {
                        std::cerr << fmt::format(
                            "[{}] Mismatch for cell {} at level {} (owned here by {} by rule {}, and owned by {} by rule {} "
                            "on [{}] with cell_index {})\n",
                            rank,
                            mismatch.cell_index,
                            mismatch.level,
                            mismatch.owner_rank,
                            mismatch.rule,
                            mismatch.owner_rank_on_neighbour,
                            mismatch.rule_on_neighbour,
                            neighbour.rank,
                            mismatch.cell_index_on_neighbour);
                        owner_rank_field[mismatch.cell_index] = 2 * world.size(); // mark error
                    }
                    i_neighbour++;
                }

                auto samurai_cell_indices_field = make_scalar_field<std::size_t>("samurai_cell_index", mesh);
                for (std::size_t cell_index = 0; cell_index < mesh.nb_cells(); ++cell_index)
                {
                    samurai_cell_indices_field[cell_index] = static_cast<std::size_t>(cell_index);
                }
                save(fs::current_path(), "owner_mismatch", {true, true}, mesh, owner_rank_field, samurai_cell_indices_field);
                std::cout.flush();
                std::cerr << fmt::format("[{}] Error: cell ownership mismatch detected. See 'owner_mismatch.xdmf' for details.\n", rank);
                std::cerr << fmt::format("[{}] The ownership rule is the same on every rank, so two ranks can only disagree when they hold "
                                         "a common cell without being registered as neighbours of each other. This usually happens when "
                                         "low level ghosts are shared between subdomains that are not in each other's direct "
                                         "neighbourhood. To solve this issue, try increasing the min level.\n",
                                         rank);
                throw std::runtime_error(fmt::format("[{}] Cell ownership mismatch detected. See 'owner_mismatch.xdmf' for details.", rank));
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
                std::cerr << fmt::format("[{}]: Cell ownership: owned: {}, total: {}\n", world.rank(), n_owned_cells, n_local_cells);
                for_each_cell(mesh[mesh_id_t::reference],
                              [&](auto& cell)
                              {
                                  std::cerr << fmt::format("[{}]:          cell_index {} level {} (owned by {}): CI{}\n",
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
