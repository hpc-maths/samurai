// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

#include <xtensor/containers/xfixed.hpp>

#include "../algorithm.hpp"

using namespace xt::placeholders;

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    namespace detail
    {
        // Walks the periodic ghosts of `level`, one periodic direction after the
        // other, and tells the visitor, in this order for each direction:
        // - copy(i_ghosts, index_ghosts, i_cells, index_cells) for each interval
        //   of ghosts whose periodic image is held by this rank;
        // - with MPI, send(k, i_cells, index_cells) for each interval of cells
        //   whose periodic image is a ghost of the k-th neighbour, then
        //   exchange(), then receive(k, i_ghosts, index_ghosts) for each interval
        //   of ghosts whose periodic image belongs to the k-th neighbour, then
        //   end_exchange();
        // - end_direction().
        template <class Mesh, class Visitor>
        void visit_periodic_ghosts(std::size_t level, const Mesh& mesh, Visitor& visitor)
        {
            using mesh_id_t           = typename Mesh::mesh_id_t;
            using lca_type            = typename Mesh::lca_type;
            using interval_value_t    = typename Mesh::interval_t::value_t;
            using box_t               = Box<interval_value_t, Mesh::dim>;
            constexpr std::size_t dim = Mesh::dim;

            const auto& domain      = mesh.domain();
            const auto& min_indices = domain.min_indices();
            const auto& max_indices = domain.max_indices();
            const auto& mesh_ref    = mesh[mesh_id_t::reference];

            const std::size_t delta_l = domain.level() - level;

            xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> min_corner;
            xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> max_corner;
            xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> shift;

            for (std::size_t d = 0; d < dim; ++d)
            {
                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();
                shift[d]      = 0;
            }
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (!mesh.is_periodic(d))
                {
                    continue;
                }
                shift[d]                  = (max_indices[d] - min_indices[d]) >> delta_l;
                const auto shift_interval = shift[0];
                const auto shift_index    = xt::view(shift, xt::range(1, _));

                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (min_indices[d] >> delta_l);

                lca_type lca_min_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l);

                lca_type lca_max_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (min_indices[d] >> delta_l);
                max_corner[d] = (min_indices[d] >> delta_l) + mesh.ghost_width();

                lca_type lca_min_p(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l);
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();

                lca_type lca_max_p(level, box_t(min_corner, max_corner));

                auto set1 = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                         intersection(mesh_ref[level], lca_max_p));
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        visitor.copy(i, index, i - shift_interval, index - shift_index);
                    });
                auto set2 = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                         intersection(mesh_ref[level], lca_min_m));
                set2(
                    [&](const auto& i, const auto& index)
                    {
                        visitor.copy(i, index, i + shift_interval, index + shift_index);
                    });
#ifdef SAMURAI_WITH_MPI
                std::size_t k = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];

                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            visitor.send(k, i - shift_interval, index - shift_index);
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            visitor.send(k, i + shift_interval, index + shift_index);
                        });
                    ++k;
                }
                visitor.exchange();
                k = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];

                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            visitor.receive(k, i, index);
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            visitor.receive(k, i, index);
                        });
                    ++k;
                }
                visitor.end_exchange();
#endif
                visitor.end_direction();
                // reset for the next direction
                shift[d]      = 0;
                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();
            }
        }

        // Copies the periodic images held by this rank with `copy_values` and
        // exchanges the others with the neighbours.
        template <class Field, class Func>
        struct FieldPeriodicVisitor
        {
            using value_t = typename Field::value_type;

            std::size_t level;
            Field& field;
            Func& copy_values;
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world{};
            std::vector<std::vector<value_t>> out = std::vector<std::vector<value_t>>(field.mesh().mpi_neighbourhood().size());
            std::vector<std::vector<value_t>> in  = std::vector<std::vector<value_t>>(field.mesh().mpi_neighbourhood().size());
            std::vector<std::ptrdiff_t> read      = std::vector<std::ptrdiff_t>(field.mesh().mpi_neighbourhood().size());
            std::vector<mpi::request> req{};
#endif

            void copy(const auto& i_ghosts, const auto& index_ghosts, const auto& i_cells, const auto& index_cells)
            {
                copy_values(i_ghosts, index_ghosts, i_cells, index_cells);
            }

#ifdef SAMURAI_WITH_MPI
            void send(std::size_t k, const auto& i, const auto& index)
            {
                const auto& data = field(level, i, index);
                std::copy(data.begin(), data.end(), std::back_inserter(out[k]));
            }

            void exchange()
            {
                const auto& neighbours = field.mesh().mpi_neighbourhood();
                for (std::size_t k = 0; k < neighbours.size(); ++k)
                {
                    req.push_back(world.isend(neighbours[k].rank, neighbours[k].rank, out[k]));
                }
                for (std::size_t k = 0; k < neighbours.size(); ++k)
                {
                    world.recv(neighbours[k].rank, world.rank(), in[k]);
                    read[k] = 0;
                }
            }

            void receive(std::size_t k, const auto& i, const auto& index)
            {
                auto data    = field(level, i, index);
                const auto n = std::ssize(data);
                std::copy(in[k].cbegin() + read[k], in[k].cbegin() + read[k] + n, data.begin());
                read[k] += n;
            }

            // The requests MUST not outlive the direction: calling wait_all() again on
            // completed boost::mpi serialized requests corrupts the MPI request state.
            void end_exchange()
            {
                mpi::wait_all(req.begin(), req.end());
                req.clear();
                for (auto& o : out)
                {
                    o.clear();
                }
            }
#endif

            void end_direction()
            {
            }
        };

        // Runs of consecutive cells, given by the storage index of the first one.
        struct PeriodicRun
        {
            std::size_t dest;
            std::size_t src;
            std::size_t len;
        };

        // The periodic ghosts of one level as storage indices, one entry per
        // periodic direction in the order of visit_periodic_ghosts.
        struct PeriodicGhostTable
        {
            struct Direction
            {
                std::vector<PeriodicRun> copies;
                std::vector<std::vector<PeriodicRun>> send; // per neighbour, the cells to send (src, len)
                std::vector<std::vector<PeriodicRun>> recv; // per neighbour, the ghosts received (dest, len)
            };

            std::vector<Direction> directions;
        };

        template <class Mesh>
        struct PeriodicTableBuilder
        {
            std::size_t level;
            const Mesh& mesh;
            PeriodicGhostTable table{};
            PeriodicGhostTable::Direction current{};

            std::size_t offset(const auto& i, const auto& index) const
            {
                return static_cast<std::size_t>(memory_offset(mesh, {level, i.start, index}));
            }

            void copy(const auto& i_ghosts, const auto& index_ghosts, const auto& i_cells, const auto& index_cells)
            {
                current.copies.push_back({offset(i_ghosts, index_ghosts), offset(i_cells, index_cells), i_ghosts.size()});
            }

            void send(std::size_t k, const auto& i, const auto& index)
            {
                current.send.resize(mesh.mpi_neighbourhood().size());
                current.send[k].push_back({0, offset(i, index), i.size()});
            }

            void exchange()
            {
                current.send.resize(mesh.mpi_neighbourhood().size());
                current.recv.resize(mesh.mpi_neighbourhood().size());
            }

            void receive(std::size_t k, const auto& i, const auto& index)
            {
                current.recv[k].push_back({offset(i, index), 0, i.size()});
            }

            void end_exchange()
            {
            }

            void end_direction()
            {
                table.directions.push_back(std::move(current));
                current = {};
            }
        };

        template <class Mesh>
        PeriodicGhostTable make_periodic_ghost_table(std::size_t level, const Mesh& mesh)
        {
            PeriodicTableBuilder<Mesh> builder{level, mesh};
            visit_periodic_ghosts(level, mesh, builder);
            return std::move(builder.table);
        }

        // Same writes as the copies and the exchanges of visit_periodic_ghosts,
        // from the table. The components of a cell are next to each other, as in
        // projection_op_.
        template <class Field>
        void update_periodic_ghosts(Field& field, const PeriodicGhostTable& table)
        {
            using value_t            = typename Field::value_type;
            constexpr std::size_t nc = Field::n_comp;
            auto* data               = field.data();
#ifdef SAMURAI_WITH_MPI
            mpi::communicator world;
            const auto& neighbours = field.mesh().mpi_neighbourhood();
#endif
            for (const auto& direction : table.directions)
            {
                for (const auto& run : direction.copies)
                {
                    std::copy(data + run.src * nc, data + (run.src + run.len) * nc, data + run.dest * nc);
                }
#ifdef SAMURAI_WITH_MPI
                if (neighbours.empty())
                {
                    continue;
                }
                std::vector<std::vector<value_t>> out(neighbours.size());
                std::vector<mpi::request> req;
                req.reserve(neighbours.size());
                for (std::size_t k = 0; k < neighbours.size(); ++k)
                {
                    for (const auto& run : direction.send[k])
                    {
                        out[k].insert(out[k].end(), data + run.src * nc, data + (run.src + run.len) * nc);
                    }
                    req.push_back(world.isend(neighbours[k].rank, neighbours[k].rank, out[k]));
                }
                std::vector<value_t> in;
                for (std::size_t k = 0; k < neighbours.size(); ++k)
                {
                    world.recv(neighbours[k].rank, world.rank(), in);
                    auto it = in.cbegin();
                    for (const auto& run : direction.recv[k])
                    {
                        const auto n = static_cast<std::ptrdiff_t>(run.len * nc);
                        std::copy(it, it + n, data + run.dest * nc);
                        it += n;
                    }
                }
                mpi::wait_all(req.begin(), req.end());
#endif
            }
        }

        template <class Mesh>
        concept has_ghost_tables = requires(const Mesh& mesh) { mesh.ghost_tables(); };
    }

    // Calls copy_values(i_ghosts, index_ghosts, i_cells, index_cells) for each
    // interval of periodic ghosts whose image this rank holds, and exchanges the
    // others with the neighbours.
    template <class Field, class Func>
    void iterate_over_periodic_ghosts(std::size_t level, Field& field, Func&& copy_values)
    {
        detail::FieldPeriodicVisitor<Field, std::decay_t<Func>> visitor{level, field, copy_values};
        detail::visit_periodic_ghosts(level, field.mesh(), visitor);
    }

    template <class Field>
    void update_ghost_periodic(std::size_t level, Field& field)
    {
        if constexpr (detail::has_ghost_tables<typename Field::mesh_t>)
        {
            const auto& tables = field.mesh().ghost_tables();
            if (!tables.empty())
            {
                detail::update_periodic_ghosts(field, tables.periodic[level]);
                return;
            }
        }
        iterate_over_periodic_ghosts(level,
                                     field,
                                     [&](const auto& i_ghosts, const auto& index_ghosts, const auto& i_cells, const auto& index_cells)
                                     {
                                         field(level, i_ghosts, index_ghosts) = field(level, i_cells, index_cells);
                                     });
    }

    template <class Field, class... Fields>
    void update_ghost_periodic(std::size_t level, Field& field, Fields&... other_fields)
    {
        update_ghost_periodic(level, field);
        update_ghost_periodic(level, other_fields...);
    }

    template <class Field>
    void update_ghost_periodic(Field& field)
    {
        using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
        auto& mesh            = field.mesh();
        std::size_t min_level = mesh[mesh_id_t::reference].min_level();
        std::size_t max_level = mesh[mesh_id_t::reference].max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            update_ghost_periodic(level, field);
        }
    }

    template <class Field, class... Fields>
    void update_ghost_periodic(Field& field, Fields&... other_fields)
    {
        update_ghost_periodic(field);
        update_ghost_periodic(other_fields...);
    }

    template <class Tag>
    void update_tag_periodic(std::size_t level, Tag& tag)
    {
#ifdef SAMURAI_WITH_MPI
        using tag_value_type = typename Tag::value_type;
#endif
        using mesh_id_t           = typename Tag::mesh_t::mesh_id_t;
        using lca_type            = typename Tag::mesh_t::lca_type;
        using interval_value_t    = typename Tag::interval_t::value_t;
        using box_t               = Box<interval_value_t, Tag::dim>;
        constexpr std::size_t dim = Tag::dim;

        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> shift;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> min_corner;
        xt::xtensor_fixed<interval_value_t, xt::xshape<dim>> max_corner;

        auto& mesh           = tag.mesh();
        const auto& mesh_ref = mesh[mesh_id_t::reference];

        auto& domain     = mesh.domain();
        auto min_indices = domain.min_indices();
        auto max_indices = domain.max_indices();

        const std::size_t delta_l = domain.level() - level;

        for (std::size_t d = 0; d < dim; ++d)
        {
            shift[d]      = 0;
            min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
            max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();
        }
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
#endif // SAMURAI_WITH_MPI
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (mesh.is_periodic(d))
            {
#ifdef SAMURAI_WITH_MPI
                // Scoped to the current dimension on purpose: sharing the request
                // vector across dimensions (or across the two passes below) would
                // wait_all() again on already-completed boost::mpi serialized
                // requests, which corrupts the MPI request state (intermittent
                // hangs/crashes). See the same fix in iterate_over_periodic_ghosts.
                std::vector<mpi::request> req;
                req.reserve(mesh.mpi_neighbourhood().size());
                std::vector<std::vector<tag_value_type>> tag_data_out(mesh.mpi_neighbourhood().size());
                std::vector<tag_value_type> tag_data_in;
#endif // SAMURAI_WITH_MPI
                shift[d]                  = (max_indices[d] - min_indices[d]) >> delta_l;
                const auto shift_interval = shift[0];
                const auto shift_index    = xt::view(shift, xt::range(1, _));

                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (min_indices[d] >> delta_l);

                lca_type lca_min_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l);

                lca_type lca_max_m(level, box_t(min_corner, max_corner));

                min_corner[d] = (min_indices[d] >> delta_l);
                max_corner[d] = (min_indices[d] >> delta_l) + mesh.ghost_width();

                lca_type lca_min_p(level, box_t(min_corner, max_corner));

                min_corner[d] = (max_indices[d] >> delta_l);
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();

                lca_type lca_max_p(level, box_t(min_corner, max_corner));

                auto set1 = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                         intersection(mesh_ref[level], lca_max_p));
                set1(
                    [&](const auto& i, const auto& index)
                    {
                        tag(level, i, index) |= tag(level, i - shift_interval, index - shift_index);
                        tag(level, i - shift_interval, index - shift_index) |= tag(level, i, index);
                    });
                auto set2 = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                         intersection(mesh_ref[level], lca_min_m));
                set2(
                    [&](const auto& i, const auto& index)
                    {
                        tag(level, i, index) |= tag(level, i + shift_interval, index + shift_index);
                        tag(level, i + shift_interval, index + shift_index) |= tag(level, i, index);
                    });
#ifdef SAMURAI_WITH_MPI
                // first  pass tag(level, i, index) |= tag(level, i - shift_interval, index - shift_index);
                size_t neighbor_id = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    tag_data_out[neighbor_id].clear();
                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& tag_data = tag(level, i - shift_interval, index - shift_index);
                            std::copy(tag_data.begin(), tag_data.end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            const auto& tag_data = tag(level, i + shift_interval, index + shift_index);
                            std::copy(tag_data.begin(), tag_data.end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, tag_data_out[neighbor_id]));
                    ++neighbor_id;
                }
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    world.recv(mpi_neighbor.rank, world.rank(), tag_data_in);
                    auto it       = tag_data_in.cbegin();
                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            for (tag_value_type& tag_xyz : tag(level, i, index))
                            {
                                tag_xyz |= *it;
                                ++it;
                            }
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            for (tag_value_type& tag_xyz : tag(level, i, index))
                            {
                                tag_xyz |= *it;
                                ++it;
                            }
                        });
                }
                mpi::wait_all(req.begin(), req.end());
                // second pass tag(level, i - shift_interval, index - shift_index) |= tag(level, i, index);
                // Fresh request vector: the first-pass requests are done, waiting
                // on them again would corrupt the MPI request state.
                req.clear();
                neighbor_id = 0;
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    tag_data_out[neighbor_id].clear();
                    auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min_p), shift),
                                                 intersection(mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max_m), -shift),
                                                 intersection(mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            std::copy(tag(level, i, index).begin(), tag(level, i, index).end(), std::back_inserter(tag_data_out[neighbor_id]));
                        });
                    req.push_back(world.isend(mpi_neighbor.rank, mpi_neighbor.rank, tag_data_out[neighbor_id]));
                    ++neighbor_id;
                }
                for (const auto& mpi_neighbor : mesh.mpi_neighbourhood())
                {
                    const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];
                    world.recv(mpi_neighbor.rank, world.rank(), tag_data_in);
                    auto it       = tag_data_in.cbegin();
                    auto set1_mpi = intersection(translate(intersection(mesh_ref[level], lca_min_p), shift),
                                                 intersection(neighbor_mesh_ref[level], lca_max_p));
                    set1_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            auto tag_data = tag(level, i - shift_interval, index - shift_index);
                            for (auto tag_it = tag_data.begin(); tag_it != tag_data.end(); ++tag_it, ++it)
                            {
                                *tag_it |= *it;
                            }
                        });
                    auto set2_mpi = intersection(translate(intersection(mesh_ref[level], lca_max_m), -shift),
                                                 intersection(neighbor_mesh_ref[level], lca_min_m));
                    set2_mpi(
                        [&](const auto& i, const auto& index)
                        {
                            auto tag_data = tag(level, i + shift_interval, index + shift_index);
                            for (auto tag_it = tag_data.begin(); tag_it != tag_data.end(); ++tag_it, ++it)
                            {
                                *tag_it |= *it;
                            }
                        });
                }
                mpi::wait_all(req.begin(), req.end());
#endif // SAMURAI_WITH_MPI
                /* reset variables for next iterations. */
                shift[d]      = 0;
                min_corner[d] = (min_indices[d] >> delta_l) - mesh.ghost_width();
                max_corner[d] = (max_indices[d] >> delta_l) + mesh.ghost_width();
            }
        }
    }
}
