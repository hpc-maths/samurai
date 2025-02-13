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

        using interval_t       = typename mesh_t::interval_t;
        using interval_value_t = typename interval_t::value_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

        static constexpr std::size_t stencil_size = cfg::stencil_size;

      private:

        FluxDefinition<cfg> m_flux_definition;
        bool m_include_boundary_fluxes = true;
        bool m_enable_max_level_flux   = false;

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

        void enable_max_level_flux(bool enable)
        {
            m_enable_max_level_flux = enable;
        }

        bool enable_max_level_flux() const
        {
            return m_enable_max_level_flux;
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

        inline field_value_type flux_value_cmpnent(const FluxValue<cfg>& flux_value, [[maybe_unused]] size_type field_i) const
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

        template <bool enable_max_level_flux>
        void compute_stencil_values(StencilData<cfg>& data, input_field_t& field, StencilValues<cfg>& stencil_values)
        {
            if constexpr (enable_max_level_flux && mesh_t::config::prediction_order > 0)
            {
                // SAMURAI_ASSERT(stencil_size == 2, "The computation of flux at the finest level is implemented only for stencils of size
                // 2");

                data.cell_length             = field.mesh().cell_length(field.mesh().max_level());
                auto level                   = data.cells[0].level;
                auto delta_l                 = field.mesh().max_level() - level;
                auto n_children_at_max_level = 1 << delta_l;

                // {
                //     const auto& left  = data.cells[0];
                //     const auto& right = data.cells[1];
                //     interval_t ileft{left.indices[0], left.indices[0] + 1};
                //     interval_t iright{right.indices[0], right.indices[0] + 1};

                //     stencil_values[0] = portion(field, level, ileft, delta_l, n_children_at_max_level - 1)[0]; // field(level, i-1);
                //     stencil_values[1] = portion(field, level, iright, delta_l, 0)[0];                          // field(level, i);
                // }

                static_assert(stencil_size % 2 == 0, "not implemented for odd stencil sizes");
                static_assert(dim == 1);

                if (delta_l == 0)
                {
                    for (std::size_t s = 0; s < stencil_size; ++s)
                    {
                        stencil_values[s] = field[data.cells[s]];
                    }
                }
                // We need `stencil_size` cells at max_level, half to the left, half to the right of the interface
                else
                {
                    // WENO5 example with delta_l = 1
                    //
                    // We have a stencil of 3+3 coarse cells, and we want 3+3 cells at the upper level.
                    //
                    //            |___|___|___|___|___|___|___|___|
                    //    |_______|_______|_______|_______|_______|_______|
                    //                       left | right

                    const auto& left  = data.cells[stencil_size / 2 - 1];
                    const auto& right = data.cells[stencil_size / 2];

                    // To get 3+3 children (x marks below), we need to predict the children of 2+2 coarse cells.
                    //
                    //                  x   x   x   x   x   x
                    //            |___|___|___|___|___|___|___|___|
                    //    |_______|_______|_______|_______|_______|_______|
                    //            left-shift      |      right+shift
                    //
                    // Below, n_children_at_max_level = 4 (although we only need 3)

                    interval_value_t shift = 0;
                    while (stencil_size / 2 > static_cast<std::size_t>(n_children_at_max_level))
                    {
                        n_children_at_max_level *= 2;
                        ++shift;
                    }

                    // We then build the following intervals:
                    // i_left=[left-shift, left+1[       i_right=[right, right+shift+1[
                    //
                    //            |___|___|___|___|___|___|___|___|
                    //    |_______|_______|_______|_______|_______|_______|
                    //               l-s      l   | l+1=r    r+s    r+s+1

                    interval_t i_left{left.indices[0] - shift, left.indices[0] + 1};
                    interval_t i_right{right.indices[0], right.indices[0] + 1 + shift};

                    // The predicted values on the fine level are numbered this way:
                    //
                    //              left interval | right interval
                    //              0   1   2   3 | 0   1   2   3
                    //            |___|___|___|___|___|___|___|___|
                    //    |_______|_______|_______|_______|_______|_______|
                    //
                    // We want 1,2,3 on the left, and 0,1,2 on the right.

                    // (stencil_size / 2) values on the left of the interface
                    interval_value_t most_left_index = n_children_at_max_level - static_cast<interval_value_t>(stencil_size / 2);
                    for (std::size_t s = 0; s < stencil_size / 2; ++s)
                    {
                        stencil_values[s] = portion(field, level, i_left, delta_l, most_left_index + static_cast<interval_value_t>(s))[0];
                    }

                    // (stencil_size / 2) values on the right of the interface
                    for (std::size_t s = 0; s < stencil_size / 2; ++s)
                    {
                        if (level == 8 && i_right.start == 32 && s == 2)
                        {
                            std::cout << std::endl;
                            std::cout << data.cells[stencil_size / 2 + s] << std::endl;
                            std::cout << field.mesh()[mesh_id_t::reference][level] << std::endl;
                        }
                        stencil_values[stencil_size / 2 + s] = portion(field, level, i_right, delta_l, static_cast<interval_value_t>(s))[0];
                    }
                }
            }
            else
            {
                data.cell_length = data.cells[0].length;
                for (std::size_t s = 0; s < stencil_size; ++s)
                {
                    stencil_values[s] = field[data.cells[s]];
                }
            }
        }

        template <bool enable_max_level_flux, class InterfaceIterator, class StencilIterator, class FluxFunction, class Func>
        void process_interior_interfaces(InterfaceIterator& interface_it,
                                         StencilIterator& comput_stencil_it,
                                         FluxFunction& flux_function,
                                         input_field_t& field,
                                         double left_factor,
                                         double right_factor,
                                         Func&& apply_contrib)
        {
            FluxValuePair<cfg> flux_values;
            StencilValues<cfg> stencil_values;
            StencilData<cfg> data(comput_stencil_it.cells());

            for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
            {
                compute_stencil_values<enable_max_level_flux>(data, field, stencil_values);
                flux_function(flux_values, data, stencil_values);
                flux_values[0] *= left_factor;
                flux_values[1] *= right_factor;
                apply_contrib(interface_it.cells()[0], flux_values[0]);
                apply_contrib(interface_it.cells()[1], flux_values[1]);

                interface_it.move_next();
                comput_stencil_it.move_next();
            }
        }

        template <bool enable_max_level_flux, bool direction, class InterfaceIterator, class StencilIterator, class FluxFunction, class Func>
        void process_boundary_interfaces(InterfaceIterator& interface_it,
                                         StencilIterator& comput_stencil_it,
                                         FluxFunction& flux_function,
                                         input_field_t& field,
                                         double factor,
                                         Func&& apply_contrib)
        {
            static_assert(!enable_max_level_flux);

            FluxValuePair<cfg> flux_values;
            StencilValues<cfg> stencil_values;
            StencilData<cfg> data(comput_stencil_it.cells());

            data.cell_length = data.cells[0].length;

            for (std::size_t ii = 0; ii < comput_stencil_it.interval().size(); ++ii)
            {
                // compute_stencil_values<false>(data, field, stencil_values);
                for (std::size_t s = 0; s < cfg::stencil_size; ++s)
                {
                    stencil_values[s] = field[data.cells[s]];
                }

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
        template <Run run_type = Run::Sequential, bool enable_max_level_flux, class Func>
        void for_each_interior_interface(std::size_t d, input_field_t& field, Func&& apply_contrib)
        {
            auto& mesh = field.mesh();

            auto min_level = mesh[mesh_id_t::cells].min_level();
            auto max_level = mesh[mesh_id_t::cells].max_level();

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

            double h_max_level = mesh.cell_length(mesh.max_level());

            // Same level
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto h      = mesh.cell_length(level);
                auto h_face = enable_max_level_flux ? h_max_level : h;
                auto factor = h_factor(h_face, h);

                for_each_interior_interface__same_level<run_type, Get::Intervals>(mesh,
                                                                                  level,
                                                                                  flux_def.direction,
                                                                                  flux_def.stencil,
                                                                                  [&](auto& interface_it, auto& comput_stencil_it)
                                                                                  {
                                                                                      process_interior_interfaces<enable_max_level_flux>(
                                                                                          interface_it,
                                                                                          comput_stencil_it,
                                                                                          flux_function,
                                                                                          field,
                                                                                          factor,
                                                                                          factor,
                                                                                          std::forward<Func>(apply_contrib));
                                                                                  });
            }

            // Level jumps (level -- level+1)
            for (std::size_t level = min_level; level < max_level; ++level)
            {
                auto h_l   = mesh.cell_length(level);
                auto h_lp1 = mesh.cell_length(level + 1);

                auto h_face = enable_max_level_flux ? h_max_level : h_lp1;

                //         |__|   l+1
                //    |____|      l
                //    --------->
                //    direction
                {
                    auto left_factor  = h_factor(h_face, h_l);
                    auto right_factor = h_factor(h_face, h_lp1);

                    for_each_interior_interface__level_jump_direction<run_type, Get::Intervals>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_it, auto& comput_stencil_it)
                        {
                            process_interior_interfaces<enable_max_level_flux>(interface_it,
                                                                               comput_stencil_it,
                                                                               flux_function,
                                                                               field,
                                                                               left_factor,
                                                                               right_factor,
                                                                               std::forward<Func>(apply_contrib));
                        });
                }
                //    |__|        l+1
                //       |____|   l
                //    --------->
                //    direction
                {
                    auto left_factor  = h_factor(h_face, h_lp1);
                    auto right_factor = h_factor(h_face, h_l);

                    for_each_interior_interface__level_jump_opposite_direction<run_type, Get::Intervals>(
                        mesh,
                        level,
                        flux_def.direction,
                        flux_def.stencil,
                        [&](auto& interface_it, auto& comput_stencil_it)
                        {
                            process_interior_interfaces<enable_max_level_flux>(interface_it,
                                                                               comput_stencil_it,
                                                                               flux_function,
                                                                               field,
                                                                               left_factor,
                                                                               right_factor,
                                                                               std::forward<Func>(apply_contrib));
                        });
                }
            }
        }

        /**
         * This function is used in the Explicit class to iterate over the boundary interfaces
         * in a specific direction and receive the contribution computed from the stencil.
         */
        template <Run run_type = Run::Sequential, bool enable_max_level_flux, class Func>
        void for_each_boundary_interface(std::size_t d, input_field_t& field, Func&& apply_contrib)
        {
            auto& mesh = field.mesh();

            auto& flux_def = flux_definition()[d];

            auto flux_function = flux_def.flux_function ? flux_def.flux_function : flux_def.flux_function_as_conservative();

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
