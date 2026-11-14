#pragma once
#include "flux_based_scheme.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <stdexcept>

namespace samurai
{
    namespace detail
    {
        template <bool enable_finer_level_flux>
        SAMURAI_INLINE auto get_dest_level(std::size_t level, int finer_level_flux, std::size_t max_level)
        {
            if constexpr (enable_finer_level_flux)
            {
                return finer_level_flux < 0 ? max_level : std::min(level + static_cast<std::size_t>(finer_level_flux), max_level);
            }
            else
            {
                return level;
            }
        }

        /**
         * A 1D prediction as a dense matrix: row i predicts the fine cell `first + i`, delta_l levels
         * finer, from the coarse cells [lo, lo + n_cols). Fine and coarse cells are numbered from the
         * same coarse cell 0, so the fine cell t lies in the coarse cell t >> delta_l.
         */
        struct PredictionMatrix1D
        {
            int lo             = 0;
            std::size_t n_rows = 0;
            std::size_t n_cols = 0;
            std::vector<double> coeffs; // row-major

            double operator()(std::size_t row, std::size_t col) const
            {
                return coeffs[row * n_cols + col];
            }
        };

        /**
         * The rows [first, first + n_rows) of the 1D prediction of radius `prediction_stencil_radius`.
         * The columns are exactly the coarse cells these rows read. A prediction map in dim > 1 is
         * the tensor product of 1D maps, so it reads the cells of a box of such columns, the same
         * cells as portion() reads.
         */
        template <std::size_t prediction_stencil_radius>
        PredictionMatrix1D prediction_matrix_1d(std::size_t delta_l, int first, std::size_t n_rows)
        {
            const int mask = (1 << delta_l) - 1;

            // The coarse cells read by the fine cell t, with their weights.
            auto map_of = [&](std::size_t row) -> const auto&
            {
                const int t = first + static_cast<int>(row);
                return prediction<prediction_stencil_radius>(delta_l, t & mask).coeff;
            };
            auto coarse_cell = [&](std::size_t row, const auto& entry)
            {
                const int t = first + static_cast<int>(row);
                return entry.first[0] + (t >> delta_l);
            };

            int lo = std::numeric_limits<int>::max();
            int hi = std::numeric_limits<int>::min();
            for (std::size_t row = 0; row < n_rows; ++row)
            {
                for (const auto& entry : map_of(row))
                {
                    lo = std::min(lo, coarse_cell(row, entry));
                    hi = std::max(hi, coarse_cell(row, entry));
                }
            }

            PredictionMatrix1D m;
            m.lo     = lo;
            m.n_rows = n_rows;
            m.n_cols = static_cast<std::size_t>(hi - lo + 1);
            m.coeffs.assign(m.n_rows * m.n_cols, 0.);
            std::vector<bool> read(m.n_cols, false);
            for (std::size_t row = 0; row < n_rows; ++row)
            {
                for (const auto& entry : map_of(row))
                {
                    const auto col                 = static_cast<std::size_t>(coarse_cell(row, entry) - lo);
                    m.coeffs[row * m.n_cols + col] = entry.second;
                    read[col]                      = true;
                }
            }
            // No gap: every column is read by some row.
            assert(std::find(read.begin(), read.end(), false) == read.end());
            return m;
        }
    }

    /**
     * Specialization of @ref FluxBasedScheme for non-linear schemes.
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
        : public FluxBasedSchemeBase<cfg, bdry_cfg>
    {
      public:

        using base_class = FluxBasedSchemeBase<cfg, bdry_cfg>;

        using base_class::dim;
        using base_class::n_comp;
        using base_class::output_n_comp;
        using size_type = typename base_class::size_type;

        using typename base_class::field_value_type;
        using typename base_class::input_field_t;
        using typename base_class::mesh_id_t;
        using typename base_class::mesh_t;
        using typename base_class::output_field_t;

        using interval_t       = typename mesh_t::interval_t;
        using interval_value_t = typename interval_t::value_t;

        static constexpr std::size_t stencil_size = cfg::stencil_size;

        using base_class::base_class;
        using base_class::flux_definition;
        using base_class::include_boundary_fluxes;

      private:

        int m_finer_level_flux = args::finer_level_flux;

      public:

        int finer_level_flux() const
        {
            return m_finer_level_flux;
        }

        int& finer_level_flux()
        {
            return m_finer_level_flux;
        }

        bool enable_finer_level_flux() const
        {
            return m_finer_level_flux != 0;
        }

        void enable_max_level_flux(bool enable)
        {
            if (enable && dim > 1 && stencil_size > 4 && !args::refine_boundary) // cppcheck-suppress knownConditionTrueFalse
            {
                std::cerr << "Warning: for stencils larger than 4, computing fluxes at max_level may cause issues close to the boundary."
                          << std::endl;
            }
            m_finer_level_flux = enable ? -1 : 0;
        }

        bool enable_max_level_flux() const
        {
            return m_finer_level_flux == -1;
        }

      private:

        using base_class::h_factor;

      public:

        SAMURAI_INLINE field_value_type flux_value_cmpnent(const FluxValue<cfg>& flux_value, [[maybe_unused]] size_type field_i) const
        {
            if constexpr (output_field_t::is_scalar)
            {
                return flux_value;
            }
            else
            {
                return flux_value(static_cast<flux_index_type>(field_i));
            }
        }

      private:

        // With a prediction stencil radius of 0 and at most 4 cells, the fine stencil values are copies of
        // the coarse ones (see compute_stencil_values()).
        static constexpr bool copies_coarse_values = mesh_t::config_t::prediction_stencil_radius == 0 && stencil_size <= 4;

        /**
         * We store here everything that we compute only once per level
         */
        template <bool enable_finer_level_flux>
        struct FluxParameters
        {
            std::size_t flux_direction = 0;
            std::size_t max_level      = 0;
            std::size_t level          = 0;
            std::size_t delta_l        = 0;
            double left_factor         = 0;
            double right_factor        = 0;
            double cell_length         = 0; // cell length at the level where the flux is computed
            int finer_level_flux       = 0; // this is the number of levels to go up to compute the flux
            std::size_t n_fine_fluxes  = 0; // number of fluxes at max_level to sum up together, or 1 if enable_max_finer_flux=false
            std::size_t n_children     = 0; // number of 1D children at level + delta_l of one cell at the current level

            // 1D predictions to level + delta_l, set when delta_l > 0. Along the flux direction: the
            // stencil_size fine cells around the interface. Across it: the children of one coarse cell.
            detail::PredictionMatrix1D along;
            detail::PredictionMatrix1D across;

            void set_level(std::size_t l)
            {
                assert(l <= max_level);
                level   = l;
                delta_l = (finer_level_flux < 0) ? max_level - l : std::min(l + static_cast<std::size_t>(finer_level_flux), max_level) - l;
                n_fine_fluxes = (1 << ((dim - 1) * delta_l));
                n_children    = std::size_t{1} << delta_l;

                if constexpr (enable_finer_level_flux && !copies_coarse_values)
                {
                    if (delta_l > 0)
                    {
                        static constexpr auto radius = static_cast<std::size_t>(mesh_t::config_t::prediction_stencil_radius);
                        // Fine cells are numbered from the first child of the cell right of the interface.
                        along  = detail::prediction_matrix_1d<radius>(delta_l, -static_cast<int>(stencil_size / 2), stencil_size);
                        across = detail::prediction_matrix_1d<radius>(delta_l, 0, n_children);
                    }
                }
            }
        };

        SAMURAI_INLINE void
        copy_stencil_values(const input_field_t& field, const StencilCells<cfg>& cells, StencilValues<cfg>& stencil_values) const
        {
            for (std::size_t s = 0; s < stencil_size; ++s)
            {
                stencil_values[s] = field[cells[s]];
            }
        }

        // Copy the coarse cells [lo, lo + n) of component `comp` into `box`, x fastest.
        template <std::size_t... K>
        static void read_coarse_box(const input_field_t& field,
                                    [[maybe_unused]] std::size_t comp,
                                    std::size_t level,
                                    const std::array<interval_value_t, dim>& lo,
                                    const std::array<std::size_t, dim>& n,
                                    std::vector<field_value_type>& box,
                                    std::index_sequence<K...>)
        {
            std::size_t n_rows = 1;
            for (std::size_t k = 1; k < dim; ++k)
            {
                n_rows *= n[k];
            }
            box.resize(n_rows * n[0]);

            const interval_t row_interval{lo[0], lo[0] + static_cast<interval_value_t>(n[0])};
            for (std::size_t row = 0; row < n_rows; ++row)
            {
                std::array<interval_value_t, dim> index = lo;
                for (std::size_t k = 1, rest = row; k < dim; ++k)
                {
                    index[k] += static_cast<interval_value_t>(rest % n[k]);
                    rest /= n[k];
                }
                auto copy_row = [&](const auto& values)
                {
                    for (std::size_t i = 0; i < n[0]; ++i)
                    {
                        box[row * n[0] + i] = values(i);
                    }
                };
                if constexpr (input_field_t::is_scalar)
                {
                    copy_row(field(level, row_interval, index[K + 1]...));
                }
                else
                {
                    copy_row(field(comp, level, row_interval, index[K + 1]...));
                }
            }
        }

        // Stencil values of all the fine fluxes of an interface, for delta_l > 0.
        // The prediction is a tensor product of 1D predictions: read the coarse box once, apply
        // `along` in the flux direction, then `across` in each other direction. The fine flux
        // c1 + n_children * c2 uses the children c1 and c2 in the other directions, lowest first.
        template <bool enable_finer_level_flux>
        void predict_stencil_values(const FluxParameters<enable_finer_level_flux>& flux_params,
                                    const StencilCells<cfg>& cells,
                                    const input_field_t& field,
                                    std::vector<StencilValues<cfg>>& stencil_values_list) const
        {
            using fine_values_t = std::array<field_value_type, stencil_size>; // the fine values along the flux direction

            const std::size_t d = flux_params.flux_direction;
            const auto& along   = flux_params.along;
            const auto& across  = flux_params.across;
            const auto& right   = cells[stencil_size / 2].indices; // cell 0 of the matrices
            assert(cells[stencil_size / 2 - 1].indices[d] + 1 == right[d]);

            // The coarse box: along.n_cols cells in the flux direction, across.n_cols in the others.
            std::array<interval_value_t, dim> lo;
            std::array<std::size_t, dim> n;
            std::array<std::size_t, dim> stride{}; // x fastest
            std::array<std::size_t, dim> other_dirs{};
            std::size_t n_coarse_across = 1; // coarse positions across the flux direction
            for (std::size_t k = 0, o = 0; k < dim; ++k)
            {
                lo[k]     = right[k] + (k == d ? along.lo : across.lo);
                n[k]      = k == d ? along.n_cols : across.n_cols;
                stride[k] = k == 0 ? 1 : stride[k - 1] * n[k - 1];
                if (k != d)
                {
                    other_dirs[o++] = k;
                    n_coarse_across *= across.n_cols;
                }
            }

            struct Buffers
            {
                std::vector<field_value_type> box;
                std::vector<fine_values_t> in;
                std::vector<fine_values_t> out;
            };

            thread_local Buffers buf;

            for (std::size_t comp = 0; comp < n_comp; ++comp)
            {
                read_coarse_box(field, comp, flux_params.level, lo, n, buf.box, std::make_index_sequence<dim - 1>{});

                // Along the flux direction: one fine_values_t per coarse position across, lowest direction fastest.
                buf.in.resize(n_coarse_across);
                for (std::size_t j = 0; j < n_coarse_across; ++j)
                {
                    std::size_t offset = 0;
                    for (std::size_t o = 0, rest = j; o + 1 < dim; ++o)
                    {
                        offset += (rest % across.n_cols) * stride[other_dirs[o]];
                        rest /= across.n_cols;
                    }
                    fine_values_t values{};
                    for (std::size_t i = 0; i < along.n_cols; ++i)
                    {
                        const field_value_type u = buf.box[offset + i * stride[d]];
                        for (std::size_t s = 0; s < stencil_size; ++s)
                        {
                            values[s] += along(s, i) * u;
                        }
                    }
                    buf.in[j] = values;
                }

                // Across: one direction at a time. buf.in is [outer][coarse][inner]: the inner
                // directions are already children, the outer ones still coarse.
                std::size_t inner = 1;
                std::size_t outer = n_coarse_across;
                for (std::size_t o = 0; o + 1 < dim; ++o)
                {
                    outer /= across.n_cols;
                    buf.out.resize(outer * across.n_rows * inner);
                    for (std::size_t p = 0; p < outer; ++p)
                    {
                        for (std::size_t c = 0; c < across.n_rows; ++c)
                        {
                            for (std::size_t q = 0; q < inner; ++q)
                            {
                                fine_values_t values{};
                                for (std::size_t j = 0; j < across.n_cols; ++j)
                                {
                                    const double coeff = across(c, j);
                                    const auto& src    = buf.in[(p * across.n_cols + j) * inner + q];
                                    for (std::size_t s = 0; s < stencil_size; ++s)
                                    {
                                        values[s] += coeff * src[s];
                                    }
                                }
                                buf.out[(p * across.n_rows + c) * inner + q] = values;
                            }
                        }
                    }
                    std::swap(buf.in, buf.out);
                    inner *= across.n_rows;
                }

                for (std::size_t f = 0; f < flux_params.n_fine_fluxes; ++f)
                {
                    for (std::size_t s = 0; s < stencil_size; ++s)
                    {
                        if constexpr (input_field_t::is_scalar)
                        {
                            stencil_values_list[f][s] = buf.in[f][s];
                        }
                        else
                        {
                            stencil_values_list[f][s](static_cast<size_type>(comp)) = buf.in[f][s];
                        }
                    }
                }
            }
        }

        template <bool enable_finer_level_flux>
        void compute_stencil_values(const FluxParameters<enable_finer_level_flux>& flux_params,
                                    const StencilCells<cfg>& cells,
                                    const input_field_t& field,
                                    std::vector<StencilValues<cfg>>& stencil_values_list) const
        {
            if constexpr (!enable_finer_level_flux)
            {
                copy_stencil_values(field, cells, stencil_values_list[0]);
            }
            else if constexpr (copies_coarse_values)
            {
                for (std::size_t fine_flux_index = 0; fine_flux_index < flux_params.n_fine_fluxes; ++fine_flux_index)
                {
                    copy_stencil_values(field, cells, stencil_values_list[fine_flux_index]);
                }
            }
            else
            {
                static_assert(!enable_finer_level_flux || stencil_size % 2 == 0, "not implemented for odd stencil sizes");

                if (flux_params.delta_l == 0)
                {
                    copy_stencil_values(field, cells, stencil_values_list[0]);
                }
                else
                {
                    // We need `stencil_size` cells at level + delta_l, half to the left, half to the right of the
                    // interface, for every child of the interface. WENO5 example with delta_l = 1:
                    //
                    //                  x   x   x | x   x   x
                    //            |___|___|___|___|___|___|___|___|
                    //    |_______|_______|_______|_______|_______|_______|
                    //                       left | right
                    predict_stencil_values(flux_params, cells, field, stencil_values_list);
                }
            }
        }

        template <bool enable_finer_level_flux, class InterfaceIterator, class StencilIterator, class FluxFunction, class Func>
        void process_interior_interfaces(const FluxParameters<enable_finer_level_flux>& flux_params,
                                         InterfaceIterator& interface_it,
                                         StencilIterator& comput_stencil_it,
                                         const FluxFunction& flux_function,
                                         const input_field_t& field,
                                         Func&& apply_contrib)
        {
            std::vector<StencilValues<cfg>> stencil_values_list(flux_params.n_fine_fluxes);
            FluxValuePair<cfg> flux_values;
            StencilData<cfg> data(comput_stencil_it.cells());

            data.cell_length = flux_params.cell_length;

            for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
            {
                compute_stencil_values<enable_finer_level_flux>(flux_params, comput_stencil_it.cells(), field, stencil_values_list);

                for (std::size_t k = 0; k < flux_params.n_fine_fluxes; ++k)
                {
                    flux_function(flux_values, data, stencil_values_list[k]);
                    flux_values[0] *= flux_params.left_factor;
                    flux_values[1] *= flux_params.right_factor;
                    apply_contrib(interface_it.cells()[0], flux_values[0]);
                    apply_contrib(interface_it.cells()[1], flux_values[1]);
                }

                interface_it.move_next();
                comput_stencil_it.move_next();
            }
        }

        template <bool enable_finer_level_flux, bool direction, class InterfaceIterator, class StencilIterator, class FluxFunction, class Func>
        void process_boundary_interfaces(InterfaceIterator& interface_it,
                                         StencilIterator& comput_stencil_it,
                                         const FluxFunction& flux_function,
                                         const input_field_t& field,
                                         double factor,
                                         Func&& apply_contrib)
        {
            static_assert(!enable_finer_level_flux);

            FluxValuePair<cfg> flux_values;
            StencilValues<cfg> stencil_values;
            StencilData<cfg> data(comput_stencil_it.cells());

            data.cell_length = data.cells[0].length;

            for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
            {
                copy_stencil_values(field, data.cells, stencil_values);

                flux_function(flux_values, data, stencil_values);
                if constexpr (direction)
                {
                    flux_values[0] *= factor;
                    apply_contrib(interface_it.cells()[0], flux_values[0]);
                }
                else // opposite direction
                {
                    flux_values[1] *= -factor;
                    apply_contrib(interface_it.cells()[0], flux_values[1]);
                }

                interface_it.move_next();
                comput_stencil_it.move_next();
            }
        }

      public:

        /**
         * This function is used in the Explicit class to iterate over the interior interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, bool enable_finer_level_flux, class Func>
        void for_each_interior_interface(std::size_t d, input_field_t& field, Func&& apply_contrib)
        {
            auto& mesh = field.mesh();

            // The neighbor subdomain may have a finer level or a coarser level than the current subdomain.
            // To ensure that we compute the fluxes at every jump interface, we need to go one level below the min_level and one level above
            // the max_level of the current subdomain.
            auto min_level = std::max(mesh.min_level(), mesh[mesh_id_t::cells].min_level() - 1);
            auto max_level = std::min(mesh.max_level(), mesh[mesh_id_t::cells].max_level() + 1);

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();
            if (!flux_function)
            {
                return;
            }

            FluxParameters<enable_finer_level_flux> flux_params;
            flux_params.max_level        = mesh.max_level();
            flux_params.flux_direction   = d;
            flux_params.finer_level_flux = m_finer_level_flux;

            // Same level
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto h = mesh.cell_length(level);
                auto h_face = mesh.cell_length(detail::get_dest_level<enable_finer_level_flux>(level, m_finer_level_flux, mesh.max_level()));
                auto factor = h_factor(h_face, h);

                flux_params.set_level(level);
                flux_params.left_factor  = factor;
                flux_params.right_factor = factor;
                flux_params.cell_length  = h_face;

                for_each_interior_interface__same_level<run_type, Get::Intervals>(mesh,
                                                                                  level,
                                                                                  flux_def.direction,
                                                                                  flux_def.stencil,
                                                                                  [&](auto& interface_it, auto& comput_stencil_it)
                                                                                  {
                                                                                      process_interior_interfaces<enable_finer_level_flux>(
                                                                                          flux_params,
                                                                                          interface_it,
                                                                                          comput_stencil_it,
                                                                                          flux_function,
                                                                                          field,
                                                                                          std::forward<Func>(apply_contrib));
                                                                                  });
            }

            // Level jumps (level -- level+1)
            // Using MPI the max_level of the subdomain can be lower than the global max_level.
            // If a jump occurs with a neighbor subdomain, we have to check if max_level + 1 is reached.
            // There is no jump above the max_level of the mesh.
#ifdef SAMURAI_WITH_MPI
            const auto jump_max_level = std::min(max_level + 1, mesh.max_level());
#else
            const auto jump_max_level = max_level;
#endif
            for (std::size_t level = min_level; level < jump_max_level; ++level)
            {
                auto h_l   = mesh.cell_length(level);
                auto h_lp1 = mesh.cell_length(level + 1);

                auto h_face = mesh.cell_length(
                    detail::get_dest_level<enable_finer_level_flux>(level + 1, m_finer_level_flux, mesh.max_level()));

                flux_params.set_level(level + 1);
                flux_params.cell_length = h_face;

                //         |__|   l+1
                //    |____|      l
                //    --------->
                //    direction
                {
                    flux_params.left_factor  = h_factor(h_face, h_l);
                    flux_params.right_factor = h_factor(h_face, h_lp1);

                    for_each_interior_interface__level_jump_direction<run_type, Get::Intervals>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_it, auto& comput_stencil_it)
                        {
                            process_interior_interfaces<enable_finer_level_flux>(flux_params,
                                                                                 interface_it,
                                                                                 comput_stencil_it,
                                                                                 flux_function,
                                                                                 field,
                                                                                 std::forward<Func>(apply_contrib));
                        });
                }
                //    |__|        l+1
                //       |____|   l
                //    --------->
                //    direction
                {
                    flux_params.left_factor  = h_factor(h_face, h_lp1);
                    flux_params.right_factor = h_factor(h_face, h_l);

                    for_each_interior_interface__level_jump_opposite_direction<run_type, Get::Intervals>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_it, auto& comput_stencil_it)
                        {
                            process_interior_interfaces<enable_finer_level_flux>(flux_params,
                                                                                 interface_it,
                                                                                 comput_stencil_it,
                                                                                 flux_function,
                                                                                 field,
                                                                                 std::forward<Func>(apply_contrib));
                        });
                }
            }
        }

        /**
         * This function is used in the Explicit class to iterate over the boundary interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, bool enable_finer_level_flux, class Func>
        void for_each_boundary_interface(std::size_t d, input_field_t& field, Func&& apply_contrib)
        {
            auto& mesh = field.mesh();

            if (mesh.periodicity()[d])
            {
                return; // no boundary in this direction
            }

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();
            if (!flux_function)
            {
                return;
            }

            for_each_level(mesh,
                           [&](auto level)
                           {
                               auto h      = mesh.cell_length(level);
                               auto factor = h_factor(h, h);

                               // Boundary in direction
                               for_each_boundary_interface__direction<run_type, Get::Intervals>(
                                   mesh,
                                   level,
                                   flux_def.direction,
                                   flux_def.stencil,
                                   [&](auto& interface_it, auto& comput_stencil_it)
                                   {
                                       static constexpr bool direction = true;
                                       process_boundary_interfaces<false, direction>(interface_it,
                                                                                     comput_stencil_it,
                                                                                     flux_function,
                                                                                     field,
                                                                                     factor,
                                                                                     std::forward<Func>(apply_contrib));
                                   });

                               // Boundary in opposite direction
                               for_each_boundary_interface__opposite_direction<run_type, Get::Intervals>(
                                   mesh,
                                   level,
                                   flux_def.direction,
                                   flux_def.stencil,
                                   [&](auto& interface_it, auto& comput_stencil_it)
                                   {
                                       static constexpr bool direction = false; // opposite direction
                                       process_boundary_interfaces<false, direction>(interface_it,
                                                                                     comput_stencil_it,
                                                                                     flux_function,
                                                                                     field,
                                                                                     -factor,
                                                                                     std::forward<Func>(apply_contrib));
                                   });
                           });
        }

        /**
         * This function is used in the Assembly class to iterate over the interior interfaces
         * and receive the Jacobian coefficients.
         */
        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Func>
        void for_each_interior_interface_and_coeffs(input_field_t& field, Func&& apply_contrib) const
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            for (std::size_t d = 0; d < dim; ++d)
            {
                auto& flux_def = flux_definition()[d];

                auto jacobian_function = flux_def.jacobian_function ? flux_def.jacobian_function
                                                                    : flux_def.jacobian_function_as_conservative();

                if (!jacobian_function)
                {
                    throw std::runtime_error(fmt::format("The jacobian function of operator '{}' has not been implemented.\n"
                                                         "Use option -snes_mf or -snes_fd for an automatic computation of the jacobian matrix.",
                                                         this->name()));
                }

                // Worker variables
                StencilValues<cfg> stencil_values;
                StencilJacobianPair<cfg> jacobians;

                this->template for_each_interior_interface_group<run_type, Get::Intervals, include_periodic>(
                    d,
                    mesh,
                    min_level,
                    max_level,
                    [&](double h_face, double left_factor, double right_factor, auto&& for_each_interface)
                    {
                        for_each_interface(
                            [&](auto& interface_it, auto& comput_stencil_it)
                            {
                                StencilData<cfg> data(comput_stencil_it.cells());
                                data.cell_length = h_face;

                                for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                                {
                                    copy_stencil_values(field, comput_stencil_it.cells(), stencil_values);

                                    jacobian_function(jacobians, data, stencil_values);
                                    jacobians[0] *= left_factor;
                                    jacobians[1] *= right_factor;
                                    apply_contrib(interface_it.cells(), comput_stencil_it.cells(), jacobians[0], jacobians[1]);

                                    interface_it.move_next();
                                    comput_stencil_it.move_next();
                                }
                            });
                    });
            }
        }

        /**
         * This function is used in the Assembly class to iterate over the boundary interfaces
         * and receive the Jacobian coefficients.
         */
        template <Run run_type = Run::Sequential, class Func>
        void for_each_boundary_interface_and_coeffs(input_field_t& field, Func&& apply_contrib) const
        {
            auto& mesh = field.mesh();
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (mesh.periodicity()[d])
                {
                    continue; // no boundary in this direction
                }

                auto& flux_def = flux_definition()[d];

                auto jacobian_function = flux_def.jacobian_function ? flux_def.jacobian_function
                                                                    : flux_def.jacobian_function_as_conservative();

                // Worker variables
                StencilValues<cfg> stencil_values;
                StencilJacobianPair<cfg> jacobians;

                for_each_level(mesh,
                               [&](auto level)
                               {
                                   auto h      = mesh.cell_length(level);
                                   auto factor = h_factor(h, h);

                                   // Boundary in direction
                                   for_each_boundary_interface__direction<run_type, Get::Intervals>(
                                       mesh,
                                       level,
                                       flux_def.direction,
                                       flux_def.stencil,
                                       [&](auto& interface_it, auto& comput_stencil_it)
                                       {
                                           StencilData<cfg> data(comput_stencil_it.cells());
                                           data.cell_length = h;

                                           for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                                           {
                                               copy_stencil_values(field, comput_stencil_it.cells(), stencil_values);

                                               jacobian_function(jacobians, data, stencil_values);
                                               jacobians[0] *= factor;
                                               apply_contrib(interface_it.cells()[0], comput_stencil_it.cells(), jacobians[0]);

                                               interface_it.move_next();
                                               comput_stencil_it.move_next();
                                           }
                                       });

                                   // Boundary in opposite direction
                                   for_each_boundary_interface__opposite_direction<run_type, Get::Intervals>(
                                       mesh,
                                       level,
                                       flux_def.direction,
                                       flux_def.stencil,
                                       [&](auto& interface_it, auto& comput_stencil_it)
                                       {
                                           StencilData<cfg> data(comput_stencil_it.cells());
                                           data.cell_length = h;

                                           for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                                           {
                                               copy_stencil_values(field, comput_stencil_it.cells(), stencil_values);

                                               jacobian_function(jacobians, data, stencil_values);
                                               jacobians[1] *= factor;
                                               apply_contrib(interface_it.cells()[0], comput_stencil_it.cells(), jacobians[1]);

                                               interface_it.move_next();
                                               comput_stencil_it.move_next();
                                           }
                                       });
                               });
            }
        }
    };

} // end namespace samurai
