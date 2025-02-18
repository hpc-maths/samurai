#pragma once
#include "flux_based_scheme.hpp"

namespace samurai
{
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
        using base_class::field_size;
        using base_class::output_field_size;
        using size_type = typename base_class::size_type;

        using typename base_class::field_value_type;
        using typename base_class::input_field_t;
        using typename base_class::mesh_id_t;
        using typename base_class::mesh_t;

        using cell_t = typename mesh_t::cell_t;

        using index_t           = typename mesh_t::interval_t::index_t;
        using field_data_view_t = typename NormalFluxDefinition<cfg>::field_data_view_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

        template <bool copies>
        struct BatchMemory
        {
            using stencil_values_t = std::conditional_t<copies, StencilValuesBatch<cfg>, std::vector<field_data_view_t>>;

            // Input
            BatchData<cfg> data;
            stencil_values_t stencil_values;
            // Output
            Batch<FluxValue<cfg>> flux_values;

            void resize(std::size_t size)
            {
                data.interfaces.resize(size);
                data.comput_stencils.resize(size);
                if constexpr (copies)
                {
                    stencil_values.resize(size);
                }
                else
                {
                    stencil_values.reserve(cfg::stencil_size);
                }
                flux_values.resize(size);
            }

            void reset()
            {
                data.interfaces.reset_position();
                data.comput_stencils.reset_position();
                if constexpr (copies)
                {
                    stencil_values.reset_position();
                }
                else
                {
                    stencil_values.clear();
                }
            }

            inline auto capacity()
            {
                return data.interfaces.capacity();
            }

            inline auto current_size()
            {
                return data.interfaces.position();
            }

            inline bool is_full()
            {
                return data.interfaces.position() == data.interfaces.capacity();
            }

            inline bool is_empty()
            {
                return data.interfaces.position() == 0;
            }

            inline auto remaining_size()
            {
                return data.interfaces.capacity() - data.interfaces.position();
            }
        };

      private:

        FluxDefinition<cfg> m_flux_definition;
        BatchMemory<true> m_batch_by_copies;
        BatchMemory<false> m_batch_by_views;

        bool m_include_boundary_fluxes = true;
        bool m_enable_batches          = true;

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

        void enable_batches(bool enable)
        {
            m_enable_batches = enable;
        }

        bool enable_batches() const
        {
            return m_enable_batches;
        }

      private:

        inline auto h_factor(double h_face, double h_cell) const
        {
            double face_measure = std::pow(h_face, dim - 1);
            double cell_measure = std::pow(h_cell, dim);
            return face_measure / cell_measure;
        }

        template <class T> // FluxValue<cfg> or StencilJacobian<cfg>
        inline T contribution(const T& flux_value, double h_face, double h_cell) const
        {
            return h_factor(h_face, h_cell) * flux_value;
        }

      public:

        inline field_value_type flux_value_cmpnent(const FluxValue<cfg>& flux_value, [[maybe_unused]] std::size_t field_i) const
        {
            if constexpr (output_field_size == 1)
            {
                return flux_value;
            }
            else
            {
                return flux_value(static_cast<flux_index_type>(field_i));
            }
        }

      private:

        template <bool copies, class Func>
        void call_flux_function__batch(BatchMemory<copies>& b,
                                       const NormalFluxDefinition<cfg>& flux_def,
                                       double left_factor,
                                       double right_factor,
                                       Func&& apply_contrib)
        {
            b.data.batch_size = b.current_size();
            b.flux_values.resize(b.data.batch_size);

            // times::timers_b.start("Flux computation");
            if constexpr (copies)
            {
                flux_def.cons_flux_function__batch_copies(b.data, b.flux_values, b.stencil_values);
            }
            else
            {
                flux_def.cons_flux_function__batch_views(b.data, b.flux_values, b.stencil_values);
            }
            // times::timers_b.stop("Flux computation");

            b.flux_values *= left_factor;
            apply_contrib(b.data.interfaces[0], b.flux_values);
            b.flux_values *= -1. / left_factor * right_factor; // add minus sign, cancel left factor and apply right one
            apply_contrib(b.data.interfaces[1], b.flux_values);

            b.reset();
        }

        template <bool copies, class Func>
        inline void call_flux_function_boundary__batch(BatchMemory<copies>& b,
                                                       const NormalFluxDefinition<cfg>& flux_def,
                                                       double factor,
                                                       Func&& apply_contrib)
        {
            b.data.batch_size = b.current_size();
            b.flux_values.resize(b.data.batch_size);

            if constexpr (copies)
            {
                flux_def.cons_flux_function__batch_copies(b.data, b.flux_values, b.stencil_values);
            }
            else
            {
                flux_def.cons_flux_function__batch_views(b.data, b.flux_values, b.stencil_values);
            }

            b.flux_values *= factor;
            apply_contrib(b.data.interfaces[0], b.flux_values);

            b.reset();
        }

        template <bool enable_batches, class InterfaceIterator, class StencilIterator, class FluxFunction, class Func>
        void process_interior_interfaces(InterfaceIterator& interface_it,
                                         StencilIterator& comput_stencil_it,
                                         const NormalFluxDefinition<cfg>& flux_def,
                                         FluxFunction& flux_function,
                                         input_field_t& field,
                                         double left_factor,
                                         double right_factor,
                                         Func&& apply_contrib)
        {
            if constexpr (!enable_batches)
            {
                for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                {
                    auto flux_values = flux_function(comput_stencil_it.cells(), field);
                    flux_values[0] *= left_factor;
                    flux_values[1] *= right_factor;
                    apply_contrib(interface_it.cells()[0], flux_values[0]);
                    apply_contrib(interface_it.cells()[1], flux_values[1]);

                    interface_it.move_next();
                    comput_stencil_it.move_next();
                }
            }
            else
            {
                auto interval_size = comput_stencil_it.interval().size();

                if (interval_size >= args::batch_view_min_size)
                {
                    auto& b = m_batch_by_views;

                    if (interval_size > b.capacity())
                    {
                        b.resize(interval_size);
                    }

                    // Views to field values
                    auto interval_step = comput_stencil_it.interval().step;
                    // times::timers_b.start("Views");
                    for (std::size_t s = 0; s < cfg::stencil_size; ++s)
                    {
                        auto start = comput_stencil_it.cells()[s].index;
                        auto end   = start + static_cast<index_t>(interval_size);
                        b.stencil_values.emplace_back(field(start, end, interval_step));
                    }
                    // times::timers_b.stop("Views");

                    copy_to_batch(interface_it, interval_size, b.data.interfaces);
                    copy_to_batch(comput_stencil_it, interval_size, b.data.comput_stencils);

                    call_flux_function__batch(b, flux_def, left_factor, right_factor, std::forward<Func>(apply_contrib));
                }
                else
                {
                    auto& b = m_batch_by_copies;

                    std::size_t to_process = interval_size;
                    while (to_process > 0)
                    {
                        auto n = std::min(to_process, b.remaining_size());

                        // Copy field values
                        // times::timers_b.start("Copies");
                        copy_values_to_batch(comput_stencil_it, n, b.stencil_values, field);
                        // times::timers_b.stop("Copies");

                        copy_to_batch(interface_it, n, b.data.interfaces);
                        copy_to_batch(comput_stencil_it, n, b.data.comput_stencils);

                        to_process -= n;
                        if (b.is_full())
                        {
                            call_flux_function__batch(b, flux_def, left_factor, right_factor, std::forward<Func>(apply_contrib));
                        }
                    }
                }
            }
        }

        template <bool enable_batches, bool direction, class InterfaceIterator, class StencilIterator, class FluxFunction, class Func>
        void process_boundary_interfaces(InterfaceIterator& interface_it,
                                         StencilIterator& comput_stencil_it,
                                         const NormalFluxDefinition<cfg>& flux_def,
                                         FluxFunction& flux_function,
                                         input_field_t& field,
                                         double factor,
                                         Func&& apply_contrib)
        {
            if constexpr (!enable_batches)
            {
                for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
                {
                    auto flux_values = flux_function(comput_stencil_it.cells(), field);
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
            else
            {
                auto interval_size = comput_stencil_it.interval().size();

                if (interval_size >= args::batch_view_min_size)
                {
                    auto& b = m_batch_by_views;

                    if (interval_size > b.capacity())
                    {
                        b.resize(interval_size);
                    }

                    // Views to field values
                    auto interval_step = comput_stencil_it.interval().step;
                    // times::timers_b.start("Views");
                    for (std::size_t s = 0; s < cfg::stencil_size; ++s)
                    {
                        auto start = comput_stencil_it.cells()[s].index;
                        auto end   = start + static_cast<index_t>(interval_size);
                        b.stencil_values.emplace_back(field(start, end, interval_step));
                    }
                    // times::timers_b.stop("Views");

                    copy_to_batch(interface_it, interval_size, b.data.interfaces);
                    copy_to_batch(comput_stencil_it, interval_size, b.data.comput_stencils);

                    call_flux_function_boundary__batch(b, flux_def, factor, std::forward<Func>(apply_contrib));
                }
                else
                {
                    auto& b = m_batch_by_copies;

                    std::size_t to_process = interval_size;
                    while (to_process > 0)
                    {
                        auto n = std::min(to_process, b.remaining_size());

                        // Copy field values
                        copy_values_to_batch(comput_stencil_it, n, b.stencil_values, field);

                        copy_to_batch(interface_it, n, b.data.interfaces);
                        copy_to_batch(comput_stencil_it, n, b.data.comput_stencils);

                        to_process -= n;
                        if (b.is_full())
                        {
                            call_flux_function_boundary__batch(b, flux_def, factor, std::forward<Func>(apply_contrib));
                        }
                    }
                }
            }
        }

      public:

        /**
         * This function is used in the Explicit class to iterate over the interior interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, bool enable_batches, class Func>
        void for_each_interior_interface(std::size_t d, input_field_t& field, Func&& apply_contrib)
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

            if constexpr (enable_batches)
            {
                m_batch_by_copies.resize(args::batch_size);
                m_batch_by_views.resize(args::batch_size);
                if (flux_def.create_temp_variables)
                {
                    m_batch_by_copies.data.temp_variables = flux_def.create_temp_variables();
                }
            }

            // Same level
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto h = mesh.cell_length(level);

                m_batch_by_copies.data.cell_length = h;
                m_batch_by_views.data.cell_length  = h;
                auto factor                        = h_factor(h, h);

                for_each_interior_interface__same_level<run_type, Get::Intervals>(mesh,
                                                                                  level,
                                                                                  flux_def.direction,
                                                                                  flux_def.stencil,
                                                                                  [&](auto& interface_it, auto& comput_stencil_it)
                                                                                  {
                                                                                      process_interior_interfaces<enable_batches>(
                                                                                          interface_it,
                                                                                          comput_stencil_it,
                                                                                          flux_def,
                                                                                          flux_function,
                                                                                          field,
                                                                                          factor,
                                                                                          factor,
                                                                                          std::forward<Func>(apply_contrib));
                                                                                  });

                if constexpr (enable_batches)
                {
                    if (!m_batch_by_copies.is_empty())
                    {
                        call_flux_function__batch(m_batch_by_copies, flux_def, factor, factor, std::forward<Func>(apply_contrib));
                    }
                }
            }

            // Level jumps (level -- level+1)
            for (std::size_t level = min_level; level < max_level; ++level)
            {
                auto h_l   = mesh.cell_length(level);
                auto h_lp1 = mesh.cell_length(level + 1);

                m_batch_by_copies.data.cell_length = h_lp1;
                m_batch_by_views.data.cell_length  = h_lp1;

                //         |__|   l+1
                //    |____|      l
                //    --------->
                //    direction
                {
                    auto left_factor  = h_factor(h_lp1, h_l);
                    auto right_factor = h_factor(h_lp1, h_lp1);

                    for_each_interior_interface__level_jump_direction<run_type, Get::Intervals>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_it, auto& comput_stencil_it)
                        {
                            process_interior_interfaces<enable_batches>(interface_it,
                                                                        comput_stencil_it,
                                                                        flux_def,
                                                                        flux_function,
                                                                        field,
                                                                        left_factor,
                                                                        right_factor,
                                                                        std::forward<Func>(apply_contrib));
                        });
                    if constexpr (enable_batches)
                    {
                        if (!m_batch_by_copies.is_empty())
                        {
                            call_flux_function__batch(m_batch_by_copies, flux_def, left_factor, right_factor, std::forward<Func>(apply_contrib));
                        }
                    }
                }
                //    |__|        l+1
                //       |____|   l
                //    --------->
                //    direction
                {
                    auto left_factor  = h_factor(h_lp1, h_lp1);
                    auto right_factor = h_factor(h_lp1, h_l);

                    for_each_interior_interface__level_jump_opposite_direction<run_type, Get::Intervals>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_it, auto& comput_stencil_it)
                        {
                            process_interior_interfaces<enable_batches>(interface_it,
                                                                        comput_stencil_it,
                                                                        flux_def,
                                                                        flux_function,
                                                                        field,
                                                                        left_factor,
                                                                        right_factor,
                                                                        std::forward<Func>(apply_contrib));
                        });
                    if constexpr (enable_batches)
                    {
                        if (!m_batch_by_copies.is_empty())
                        {
                            call_flux_function__batch(m_batch_by_copies, flux_def, left_factor, right_factor, std::forward<Func>(apply_contrib));
                        }
                    }
                }
            }
        }

        /**
         * This function is used in the Explicit class to iterate over the boundary interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, bool enable_batches, class Func>
        void for_each_boundary_interface(std::size_t d, input_field_t& field, Func&& apply_contrib)
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

            if constexpr (enable_batches)
            {
                m_batch_by_copies.resize(args::batch_size);
                m_batch_by_views.resize(args::batch_size);
                if (flux_def.create_temp_variables)
                {
                    m_batch_by_copies.data.temp_variables = flux_def.create_temp_variables();
                }
            }

            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto h = mesh.cell_length(level);

                m_batch_by_copies.data.cell_length = h;
                m_batch_by_views.data.cell_length  = h;

                auto factor = h_factor(h, h);

                // Boundary in direction
                for_each_boundary_interface__direction<run_type, Get::Intervals>(mesh,
                                                                                 level,
                                                                                 flux_def.direction,
                                                                                 flux_def.stencil,
                                                                                 [&](auto& interface_it, auto& comput_stencil_it)
                                                                                 {
                                                                                     static constexpr bool direction = true;
                                                                                     process_boundary_interfaces<enable_batches, direction>(
                                                                                         interface_it,
                                                                                         comput_stencil_it,
                                                                                         flux_def,
                                                                                         flux_function,
                                                                                         field,
                                                                                         factor,
                                                                                         std::forward<Func>(apply_contrib));
                                                                                 });

                if constexpr (enable_batches)
                {
                    if (!m_batch_by_copies.is_empty())
                    {
                        call_flux_function_boundary__batch(m_batch_by_copies, flux_def, factor, std::forward<Func>(apply_contrib));
                    }
                }

                // Boundary in opposite direction
                for_each_boundary_interface__opposite_direction<run_type, Get::Intervals>(
                    mesh,
                    level,
                    flux_def.direction,
                    flux_def.stencil,
                    [&](auto& interface_it, auto& comput_stencil_it)
                    {
                        static constexpr bool direction = false; // opposite direction
                        process_boundary_interfaces<enable_batches, direction>(interface_it,
                                                                               comput_stencil_it,
                                                                               flux_def,
                                                                               flux_function,
                                                                               field,
                                                                               -factor,
                                                                               std::forward<Func>(apply_contrib));
                    });

                if constexpr (enable_batches)
                {
                    if (!m_batch_by_copies.is_empty())
                    {
                        call_flux_function_boundary__batch(m_batch_by_copies, flux_def, -factor, std::forward<Func>(apply_contrib));
                    }
                }
            }
        }

        /**
         * This function is used in the Assembly class to iterate over the interior interfaces
         * and receive the Jacobian coefficients.
         */
        template <Run run_type = Run::Sequential, class Func>
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
                    std::cerr << "The jacobian function of operator '" << this->name() << "' has not been implemented." << std::endl;
                    std::cerr << "Use option -snes_mf or -snes_fd for an automatic computation of the jacobian matrix." << std::endl;
                    exit(EXIT_FAILURE);
                }

                // Same level
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    auto h = mesh.cell_length(level);

                    for_each_interior_interface__same_level<run_type>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_cells, auto& comput_cells)
                        {
                            auto jacobians          = jacobian_function(comput_cells, field);
                            auto left_cell_contrib  = contribution(jacobians[0], h, h);
                            auto right_cell_contrib = contribution(jacobians[1], h, h);
                            apply_contrib(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
                        });
                }

                // Level jumps (level -- level+1)
                for (std::size_t level = min_level; level < max_level; ++level)
                {
                    auto h_l   = mesh.cell_length(level);
                    auto h_lp1 = mesh.cell_length(level + 1);

                    //         |__|   l+1
                    //    |____|      l
                    //    --------->
                    //    direction
                    {
                        for_each_interior_interface__level_jump_direction<run_type>(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto jacobians          = jacobian_function(comput_cells, field);
                                auto left_cell_contrib  = contribution(jacobians[0], h_lp1, h_l);
                                auto right_cell_contrib = contribution(jacobians[1], h_lp1, h_lp1);
                                apply_contrib(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
                            });
                    }
                    //    |__|        l+1
                    //       |____|   l
                    //    --------->
                    //    direction
                    {
                        for_each_interior_interface__level_jump_opposite_direction<run_type>(
                            mesh,
                            level,
                            flux_def.direction,
                            flux_def.stencil,
                            [&](auto& interface_cells, auto& comput_cells)
                            {
                                auto jacobians          = jacobian_function(comput_cells, field);
                                auto left_cell_contrib  = contribution(jacobians[0], h_lp1, h_lp1);
                                auto right_cell_contrib = contribution(jacobians[1], h_lp1, h_l);
                                apply_contrib(interface_cells, comput_cells, left_cell_contrib, right_cell_contrib);
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
                auto& flux_def = flux_definition()[d];

                auto jacobian_function = flux_def.jacobian_function ? flux_def.jacobian_function
                                                                    : flux_def.jacobian_function_as_conservative();

                for_each_level(
                    mesh,
                    [&](auto level)
                    {
                        auto h = mesh.cell_length(level);

                        // Boundary in direction
                        for_each_boundary_interface__direction<run_type>(mesh,
                                                                         level,
                                                                         flux_def.direction,
                                                                         flux_def.stencil,
                                                                         [&](auto& cell, auto& comput_cells)
                                                                         {
                                                                             auto jacobians    = jacobian_function(comput_cells, field);
                                                                             auto cell_contrib = contribution(jacobians[0], h, h);
                                                                             apply_contrib(cell, comput_cells, cell_contrib);
                                                                         });

                        // Boundary in opposite direction
                        for_each_boundary_interface__opposite_direction<run_type>(mesh,
                                                                                  level,
                                                                                  flux_def.direction,
                                                                                  flux_def.stencil,
                                                                                  [&](auto& cell, auto& comput_cells)
                                                                                  {
                                                                                      auto jacobians = jacobian_function(comput_cells, field);
                                                                                      auto cell_contrib = contribution(jacobians[1], h, h);
                                                                                      apply_contrib(cell, comput_cells, cell_contrib);
                                                                                  });
                    });
            }
        }
    };

} // end namespace samurai
