#pragma once
#include "../../../arguments.hpp"
#include "flux_based_scheme.hpp"

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
    }

    /**
     * @class FluxBasedScheme
     *    Implementation of non-linear schemes
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedScheme<cfg, bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear>>
        : public FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>
    {
      public:

        using base_class = FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>;

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
        using cell_indices_t   = typename mesh_t::cell_t::indices_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

        static constexpr std::size_t stencil_size = cfg::stencil_size;

      private:

        FluxDefinition<cfg> m_flux_definition;
        bool m_include_boundary_fluxes = true;
        int m_finer_level_flux         = args::finer_level_flux;
        bool m_exact_reconstruction    = args::exact_reconstruction;

      public:

        explicit FluxBasedScheme(const FluxDefinition<cfg>& flux_definition)
            : m_flux_definition(flux_definition)
        {
        }

        const auto& flux_definition() const
        {
            return m_flux_definition;
        }

        auto& flux_definition()
        {
            return m_flux_definition;
        }

        void include_boundary_fluxes(bool include)
        {
            m_include_boundary_fluxes = include;
        }

        bool include_boundary_fluxes() const
        {
            return m_include_boundary_fluxes;
        }

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

        // When computing fluxes at a finer level, reconstruct the finer stencil values with the real
        // finer cells across refinement boundaries instead of the flat prediction (see
        // reconstruct_exact). Default: the global --exact-reconstruction option.
        void set_exact_reconstruction(bool exact)
        {
            m_exact_reconstruction = exact;
        }

        bool exact_reconstruction() const
        {
            return m_exact_reconstruction;
        }

        void enable_max_level_flux(bool enable)
        {
            if (enable && dim > 1 && stencil_size > 4 && !args::refine_boundary) // cppcheck-suppress knownConditionTrueFalse
            {
                std::cout << "Warning: for stencils larger than 4, computing fluxes at max_level may cause issues close to the boundary."
                          << std::endl;
            }
            m_finer_level_flux = enable ? -1 : 0;
        }

        bool enable_max_level_flux() const
        {
            return m_finer_level_flux == -1;
        }

      private:

        SAMURAI_INLINE auto h_factor(double h_face, double h_cell) const
        {
            double face_measure = std::pow(h_face, dim - 1);
            double cell_measure = std::pow(h_cell, dim);
            return face_measure / cell_measure;
        }

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

        /**
         * We store here everything that we compute only once per level
         */
        using lca_t = LevelCellArray<dim, interval_t>;

        template <bool enable_finer_level_flux>
        struct FluxParameters
        {
            std::size_t flux_direction       = 0;
            std::size_t max_level            = 0;
            std::size_t level                = 0;
            std::size_t delta_l              = 0;
            bool exact                       = false;   // exact reconstruction of the finer stencil values
            const std::vector<lca_t>* stored = nullptr; // (cells U proj_cells) per level, only if exact
            double left_factor               = 0;
            double right_factor              = 0;
            double cell_length               = 0; // cell length at the level where the flux is computed
            int finer_level_flux             = 0; // this is the number of levels to go up to compute the flux
            std::size_t n_fine_fluxes        = 0; // number of fluxes at max_level to sum up together, or 1 if enable_max_finer_flux=false
            int n_children_at_max_level      = 0; // number of 1D children at max_level of one cell at the current level
            std::size_t coarse_stencil_size  = 0; // number of cells at the current level used to predict the required values at
                                                  // max_level

            void set_level(std::size_t l)
            {
                level   = l;
                delta_l = (finer_level_flux < 0) ? max_level - l : std::min(l + static_cast<std::size_t>(finer_level_flux), max_level) - l;
                n_fine_fluxes           = (1 << ((dim - 1) * delta_l));
                n_children_at_max_level = 1 << delta_l;

                coarse_stencil_size = 2;
                while (stencil_size > static_cast<std::size_t>(n_children_at_max_level) * coarse_stencil_size)
                {
                    coarse_stencil_size += 2;
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

        // Read one stored cell f(comp, level, coord) of the (possibly vector) input field, unpacking
        // the transverse directions - the reader used by the exact reconstruction.
        template <std::size_t... K>
        static field_value_type read_field_cell(const input_field_t& field,
                                                std::size_t comp,
                                                std::size_t level,
                                                const std::array<interval_value_t, dim>& c,
                                                std::index_sequence<K...>)
        {
            if constexpr (input_field_t::is_scalar)
            {
                (void)comp;
                return field(level, interval_t{c[0], c[0] + 1}, c[K + 1]...)[0];
            }
            else
            {
                return field(comp, level, interval_t{c[0], c[0] + 1}, c[K + 1]...)[0];
            }
        }

        using exact_memos_t   = std::array<exact_reconstruction_memo_t<dim, interval_value_t>, n_comp>;
        using recons_box_t    = detail::ra_box<dim, interval_value_t>;
        using recons_values_t = std::array<std::vector<double>, n_comp>;

        template <bool enable_finer_level_flux>
        SAMURAI_INLINE void predict_value(typename cfg::input_field_t::local_data_type& predicted_value,
                                          const input_field_t& field,
                                          const cell_indices_t& coarse_cell_indices,
                                          const cell_indices_t& fine_cell_indices,
                                          const FluxParameters<enable_finer_level_flux>& flux_params,
                                          exact_memos_t& memos,
                                          const recons_box_t& box,
                                          const recons_values_t& box_values)
        {
            const std::size_t level   = flux_params.level;
            const std::size_t delta_l = flux_params.delta_l;

            if constexpr (!enable_finer_level_flux)
            {
                portion(predicted_value, field, level, delta_l, coarse_cell_indices, fine_cell_indices);
            }
            else
            {
                if (delta_l == 0)
                {
                    portion(predicted_value, field, level, delta_l, coarse_cell_indices, fine_cell_indices);
                    return;
                }

                // The whole finer stencil block was reconstructed at once into box_values (flat or
                // exact, per component; see build_recons_box) - just read this cell from it.
                std::array<interval_value_t, dim> g;
                for (std::size_t k = 0; k < dim; ++k)
                {
                    g[k] = (coarse_cell_indices[k] << delta_l) + fine_cell_indices[k];
                }
                if (box.contains(g))
                {
                    const std::size_t off = box.offset(g);
                    if constexpr (input_field_t::is_scalar)
                    {
                        predicted_value = box_values[0][off];
                    }
                    else
                    {
                        for (std::size_t comp = 0; comp < n_comp; ++comp)
                        {
                            predicted_value(static_cast<size_type>(comp)) = box_values[comp][off];
                        }
                    }
                    return;
                }

                // Safety fallback (should not trigger with a correctly-sized box): reconstruct this
                // single cell, matching the flat / exact choice.
                if (!flux_params.exact)
                {
                    portion(predicted_value, field, level, delta_l, coarse_cell_indices, fine_cell_indices);
                    return;
                }
                static constexpr std::size_t r = mesh_t::config_t::prediction_stencil_radius;
                constexpr auto tseq            = std::make_index_sequence<dim - 1>{};
                auto reconstruct_component     = [&](std::size_t comp) -> field_value_type
                {
                    auto read = [&](std::size_t l, const std::array<interval_value_t, dim>& c) -> double
                    {
                        return read_field_cell(field, comp, l, c, tseq);
                    };
                    return reconstruct_exact<r>(level, delta_l, g, read, *flux_params.stored, memos[comp]);
                };
                if constexpr (input_field_t::is_scalar)
                {
                    predicted_value = reconstruct_component(0);
                }
                else
                {
                    for (std::size_t comp = 0; comp < n_comp; ++comp)
                    {
                        predicted_value(static_cast<size_type>(comp)) = reconstruct_component(comp);
                    }
                }
            }
        }

        // Reconstruct the whole finer stencil block of one interface at once (flat via
        // reconstruct_flat_box, or exact via reconstruct_exact_box), per component, into box_values.
        // The block is the stencil_size fine cells around the interface (flux direction, plus a small
        // margin) times the 2^{delta_l} children in each transverse direction - covering every cell
        // predict_value reads. This replaces the per-cell portion() with one dense cascade.
        //
        // Deliberately no interface band here, unlike LBMScheme::stream_exact_reconstruction. Such a
        // band would be safe - away from any refinement, the exact and flat cascades are the same code
        // up to the is_stored predicate, so they agree bit-for-bit, and over-inclusion would cost
        // nothing at all - but it cannot pay for itself. Measured on the 2D WENO5 burgers case
        // (min_level 2, max_level 8, --finer-level-flux -1): the whole cost of the is_stored lookups
        // is 0.48 s out of 25.3 s (1.9%), while the finer-level flux machinery itself accounts for
        // 14.6 s (59%). A band could only reclaim a fraction of that 1.9%, in exchange for a coverage
        // invariant to keep correct. If this path ever needs to be faster, the 14.6 s is the target.
        void build_recons_box(const FluxParameters<true>& flux_params,
                              const StencilCells<cfg>& cells,
                              const input_field_t& field,
                              recons_box_t& box,
                              recons_values_t& box_values) const
        {
            static constexpr std::size_t r = mesh_t::config_t::prediction_stencil_radius;
            constexpr auto tseq            = std::make_index_sequence<dim - 1>{};
            const std::size_t level        = flux_params.level;
            const std::size_t delta_l      = flux_params.delta_l;
            const auto flux_dir            = flux_params.flux_direction;
            const auto half                = static_cast<interval_value_t>(stencil_size / 2);
            const auto& right              = cells[stencil_size / 2]; // first coarse cell right of the interface

            for (std::size_t k = 0; k < dim; ++k)
            {
                if (k == flux_dir)
                {
                    const interval_value_t interface = right.indices[k] << delta_l; // fine coord of the interface
                    box.lo[k]                        = interface - half - static_cast<interval_value_t>(r);
                    box.hi[k]                        = interface + half + static_cast<interval_value_t>(r);
                }
                else
                {
                    box.lo[k] = right.indices[k] << delta_l;
                    box.hi[k] = (right.indices[k] + 1) << delta_l;
                }
            }

            for (std::size_t comp = 0; comp < n_comp; ++comp)
            {
                auto read = [&](std::size_t l, const std::array<interval_value_t, dim>& c) -> double
                {
                    return read_field_cell(field, comp, l, c, tseq);
                };
                if (flux_params.exact)
                {
                    reconstruct_exact_box<r>(level, level + delta_l, box.lo, box.hi, read, *flux_params.stored, box_values[comp]);
                }
                else
                {
                    reconstruct_flat_box<r>(level, level + delta_l, box.lo, box.hi, read, box_values[comp]);
                }
            }
        }

        template <bool enable_finer_level_flux>
        void compute_stencil_values(const FluxParameters<enable_finer_level_flux>& flux_params,
                                    const StencilCells<cfg>& cells,
                                    const input_field_t& field,
                                    std::vector<StencilValues<cfg>>& stencil_values_list)
        {
            if constexpr (!enable_finer_level_flux)
            {
                copy_stencil_values(field, cells, stencil_values_list[0]);
            }
            else if constexpr (mesh_t::config_t::prediction_stencil_radius == 0 && stencil_size <= 4)
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
                    // We need `stencil_size` cells at max_level, half to the left, half to the right of the interface.
                    // WENO5 example with delta_l = 1:
                    //
                    // We have a stencil of 3+3 coarse cells, and we want 3+3 cells at the upper level.
                    //
                    //                |___|___|___|___|___|___|
                    //    |_______|_______|_______|_______|_______|_______|
                    //                       left | right

                    const auto& left  = cells[stencil_size / 2 - 1];
                    const auto& right = cells[stencil_size / 2];

                    // To get 3+3 children (x marks), we need to predict the children of 2+2 coarse cells (o marks).
                    //
                    //                  x   x   x   x   x   x
                    //            |___|___|___|___|___|___|___|___|
                    //    |_______|_______|_______|_______|_______|_______|
                    //                o       o   |   o       o

                    auto& flux_direction          = flux_params.flux_direction;
                    auto& n_children_at_max_level = flux_params.n_children_at_max_level;

                    cell_indices_t left_coarse_cell_indices;
                    cell_indices_t right_coarse_cell_indices;

                    cell_indices_t left_fine_cell_indices;
                    cell_indices_t right_fine_cell_indices;
                    left_fine_cell_indices.fill(0);
                    right_fine_cell_indices.fill(0);

                    std::size_t moving_direction = flux_params.flux_direction == 0 ? 1 : 0; // first other direction

                    // Per-component memos for the safety fallback only (see predict_value).
                    exact_memos_t memos;

                    // Reconstruct the whole finer stencil block of this interface at once (flat or
                    // exact), instead of one portion() per fine cell; predict_value just reads it.
                    recons_box_t box;
                    recons_values_t box_values;
                    build_recons_box(flux_params, cells, field, box, box_values);

                    for (std::size_t fine_flux_index = 0; fine_flux_index < flux_params.n_fine_fluxes; ++fine_flux_index)
                    {
                        left_coarse_cell_indices  = left.indices;
                        right_coarse_cell_indices = right.indices;

                        // Compute the necessary stencil values for the current fine flux.
                        // We need (stencil_size / 2) on each side of the interface.
                        int remaining_values_to_compute = stencil_size / 2;

                        for (std::size_t coarse_cell = 0; coarse_cell < flux_params.coarse_stencil_size / 2; ++coarse_cell)
                        {
                            int n_values_to_compute = std::min(remaining_values_to_compute, n_children_at_max_level);

                            // left side of the interface: we fill from right to left
                            std::size_t index_in_stencil = stencil_size / 2 - 1
                                                         - coarse_cell * static_cast<std::size_t>(n_children_at_max_level);
                            left_fine_cell_indices[flux_direction] = n_children_at_max_level - 1;

                            for (std::size_t s = 0; s < static_cast<std::size_t>(n_values_to_compute); ++s)
                            {
                                predict_value(stencil_values_list[fine_flux_index][index_in_stencil - s],
                                              field,
                                              left_coarse_cell_indices,
                                              left_fine_cell_indices,
                                              flux_params,
                                              memos,
                                              box,
                                              box_values);
                                left_fine_cell_indices[flux_direction]--;
                            }

                            // right side of the interface: we fill from left to right
                            index_in_stencil = stencil_size / 2 + coarse_cell * static_cast<std::size_t>(n_children_at_max_level);
                            right_fine_cell_indices[flux_direction] = 0;

                            for (std::size_t s = 0; s < static_cast<std::size_t>(n_values_to_compute); ++s)
                            {
                                predict_value(stencil_values_list[fine_flux_index][index_in_stencil + s],
                                              field,
                                              right_coarse_cell_indices,
                                              right_fine_cell_indices,
                                              flux_params,
                                              memos,
                                              box,
                                              box_values);
                                right_fine_cell_indices[flux_direction]++;
                            }

                            remaining_values_to_compute -= n_values_to_compute;

                            left_coarse_cell_indices[flux_direction]--;
                            right_coarse_cell_indices[flux_direction]++;
                        }

                        if constexpr (dim > 1)
                        {
                            // Move to next fine flux
                            if (left_fine_cell_indices[moving_direction] < n_children_at_max_level - 1)
                            {
                                left_fine_cell_indices[moving_direction]++;
                                right_fine_cell_indices[moving_direction]++;
                            }
                            else
                            {
                                // if moving_direction = 'y':
                                // - reset 'y' indices
                                // - start moving the fine cells according to the 'z' direction
                                left_fine_cell_indices[moving_direction]  = 0;
                                right_fine_cell_indices[moving_direction] = 0;
                                moving_direction++;
                                if (moving_direction == flux_direction)
                                {
                                    moving_direction++;
                                }
                            }
                        }
                    }
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

            // For the exact reconstruction of the finer stencil values: the (cells U proj_cells) set
            // per level, built once (only when needed). Outlives the loops below.
            std::vector<lca_t> stored;
            if constexpr (enable_finer_level_flux)
            {
                flux_params.exact = m_exact_reconstruction;
                if (m_exact_reconstruction)
                {
                    stored             = make_real_backed_lca(mesh);
                    flux_params.stored = &stored;
                }
            }

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
#ifdef SAMURAI_WITH_MPI
            for (std::size_t level = min_level; level <= max_level; ++level)
#else
            for (std::size_t level = min_level; level < max_level; ++level)
#endif
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

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h      = mesh.cell_length(level);
                    auto factor = h_factor(h, h);

                    for_each_interior_interface__same_level<run_type, Get::Intervals, include_periodic>(
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
                                jacobians[1] *= factor;
                                apply_contrib(interface_it.cells(), comput_stencil_it.cells(), jacobians[0], jacobians[1]);

                                interface_it.move_next();
                                comput_stencil_it.move_next();
                            }
                        });
                }

                // Level jumps (level -- level+1)
                // Using MPI the max_level of the subdomain can be lower than the global max_level.
                // If a jump occurs with a neighbor subdomain, we have to check if max_level + 1 is reached.
#ifdef SAMURAI_WITH_MPI
                for (std::size_t level = min_level; level <= max_level; ++level)
#else
                for (std::size_t level = min_level; level < max_level; ++level)
#endif
                {
                    auto h_l   = mesh.cell_length(level);
                    auto h_lp1 = mesh.cell_length(level + 1);

                    auto h_face = h_lp1;

                    //         |__|   l+1
                    //    |____|      l
                    //    --------->
                    //    direction
                    {
                        auto left_factor  = h_factor(h_face, h_l);
                        auto right_factor = h_factor(h_face, h_lp1);

                        for_each_interior_interface__level_jump_direction<run_type, Get::Intervals, include_periodic>(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
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
                    }
                    //    |__|        l+1
                    //       |____|   l
                    //    --------->
                    //    direction
                    {
                        auto left_factor  = h_factor(h_face, h_lp1);
                        auto right_factor = h_factor(h_face, h_l);

                        for_each_interior_interface__level_jump_opposite_direction<run_type, Get::Intervals, include_periodic>(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
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
                    }
                }
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
