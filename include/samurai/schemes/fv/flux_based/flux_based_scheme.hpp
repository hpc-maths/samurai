#pragma once
#include "../../../arguments.hpp"
#include "../../../interface.hpp"
#include "../../../reconstruction.hpp"
#include "../../explicit_scheme.hpp"
#include "../FV_scheme.hpp"
#include "flux_definition.hpp"

namespace samurai
{
    /**
     * Scheme defined by the fluxes between neighbouring cells.
     * The primary template is empty: it is specialized for each scheme type
     * (non-linear, linear heterogeneous, linear homogeneous).
     */
    template <class cfg, class bdry_cfg, class check = void>
    class FluxBasedScheme
    {
    };

    /**
     * Common base of the specializations of FluxBasedScheme. It stores the flux definition
     * and visits the interior interfaces with the factors that turn a flux into cell contributions.
     */
    template <class cfg, class bdry_cfg>
    class FluxBasedSchemeBase : public FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>
    {
      public:

        using base_class = FVScheme<FluxBasedScheme<cfg, bdry_cfg>, cfg, bdry_cfg>;

        using base_class::dim;
        using typename base_class::input_field_t;
        using typename base_class::mesh_id_t;
        using typename base_class::mesh_t;

        using cfg_t      = cfg;
        using bdry_cfg_t = bdry_cfg;

      private:

        FluxDefinition<cfg> m_flux_definition;
        bool m_include_boundary_fluxes = true;

      public:

        explicit FluxBasedSchemeBase(const FluxDefinition<cfg>& flux_definition)
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

      protected:

        // Static, so that the generic lambdas of the derived schemes can call it without capturing 'this'
        // (GCC 12 and 13 crash on that capture through a dependent base).
        static SAMURAI_INLINE auto h_factor(double h_face, double h_cell)
        {
            double face_measure = std::pow(h_face, dim - 1);
            double cell_measure = std::pow(h_cell, dim);
            return face_measure / cell_measure;
        }

        /**
         * Visits the interior interfaces of direction d whose coarser cell has a level in [min_level, max_level],
         * grouped by the factors h_factor() applies to the flux on each side:
         *     - the interfaces between two cells of the same level,
         *     - the level jumps level|level+1, with the coarse cell on the left, then on the right.
         * For each group, calls on_group(h_face, left_factor, right_factor, for_each_interface),
         * where for_each_interface(f) calls f(interface, comput_stencil) on every interface of the group.
         */
        template <Run run_type, Get get_type, bool include_periodic, class Func>
        void
        for_each_interior_interface_group(std::size_t d, const mesh_t& mesh, std::size_t min_level, std::size_t max_level, Func&& on_group) const
        {
            auto& flux_def = flux_definition()[d];

            // Same level
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto h      = mesh.cell_length(level);
                auto factor = h_factor(h, h);

                on_group(h,
                         factor,
                         factor,
                         [&](auto&& f)
                         {
                             for_each_interior_interface__same_level<run_type, get_type, include_periodic>(mesh,
                                                                                                           level,
                                                                                                           flux_def.direction,
                                                                                                           flux_def.stencil,
                                                                                                           f);
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

                // The flux is computed at level l+1
                auto h_face = h_lp1;

                //         |__|   l+1
                //    |____|      l
                //    --------->
                //    direction
                on_group(h_face,
                         h_factor(h_face, h_l),
                         h_factor(h_face, h_lp1),
                         [&](auto&& f)
                         {
                             for_each_interior_interface__level_jump_direction<run_type, get_type, include_periodic>(mesh,
                                                                                                                     level,
                                                                                                                     flux_def.direction,
                                                                                                                     flux_def.stencil,
                                                                                                                     f);
                         });

                //    |__|        l+1
                //       |____|   l
                //    --------->
                //    direction
                on_group(h_face,
                         h_factor(h_face, h_lp1),
                         h_factor(h_face, h_l),
                         [&](auto&& f)
                         {
                             for_each_interior_interface__level_jump_opposite_direction<run_type, get_type, include_periodic>(
                                 mesh,
                                 level,
                                 flux_def.direction,
                                 flux_def.stencil,
                                 f);
                         });
            }
        }

        /**
         * Same as above, over every level where this subdomain has interior interfaces.
         */
        template <Run run_type, Get get_type, bool include_periodic, class Func>
        void for_each_interior_interface_group(std::size_t d, const mesh_t& mesh, Func&& on_group) const
        {
            // The neighbor subdomain may have a finer level or a coarser level than the current subdomain.
            // To ensure that we compute the fluxes at every jump interface, we need to go one level below the min_level and one level above
            // the max_level of the current subdomain.
            auto min_level = std::max(mesh.min_level(), mesh[mesh_id_t::cells].min_level() - 1);
            auto max_level = std::min(mesh.max_level(), mesh[mesh_id_t::cells].max_level() + 1);

            for_each_interior_interface_group<run_type, get_type, include_periodic>(d, mesh, min_level, max_level, std::forward<Func>(on_group));
        }

      public:

        /**
         * Iterates over the interior interfaces of every direction, with the coefficients given by the derived scheme
         * for one direction.
         */
        template <Run run_type = Run::Sequential, Get get_type = Get::Cells, bool include_periodic = true, class Func>
        void for_each_interior_interface_and_coeffs(input_field_t& field, Func&& apply_coeffs) const
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                this->derived_cast().template for_each_interior_interface_and_coeffs<run_type, get_type, include_periodic>(
                    d,
                    field,
                    std::forward<Func>(apply_coeffs));
            }
        }
    };

    template <class cfg>
    auto make_flux_based_scheme(const FluxDefinition<cfg>& flux_definition)
    {
        using bdry_cfg = BoundaryConfigFV<cfg::stencil_size / 2>;

        // cppcheck-suppress knownConditionTrueFalse
        if (args::finer_level_flux != 0 && cfg::dim > 1 && cfg::stencil_size > 4 && !args::refine_boundary)
        {
            std::cerr << "Warning: for stencils larger than 4, computing fluxes at max_level may cause issues close to the boundary."
                      << std::endl;
        }

        return FluxBasedScheme<cfg, bdry_cfg>(flux_definition);
    }

    /**
     * is_FluxBasedScheme
     */
    template <class Scheme, typename = void>
    struct is_FluxBasedScheme : std::false_type
    {
    };

    template <class Scheme>
    struct is_FluxBasedScheme<Scheme,
                              std::enable_if_t<std::is_base_of_v<FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::bdry_cfg_t>, Scheme>
                                               || std::is_same_v<FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::bdry_cfg_t>, Scheme>>>
        : std::true_type
    {
    };

    template <class Scheme>
    constexpr bool is_FluxBasedScheme_v = is_FluxBasedScheme<Scheme>::value;

} // end namespace samurai
