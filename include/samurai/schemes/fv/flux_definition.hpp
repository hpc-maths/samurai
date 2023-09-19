#pragma once
#include <functional>
#include <xtensor/xfixed.hpp>

namespace samurai
{
    /**
     * Matrix type
     */
    namespace detail
    {
        /**
         * Local square matrix to store the coefficients of a vectorial field.
         */
        template <class value_type, std::size_t rows, std::size_t cols>
        struct LocalMatrix
        {
            using Type = xt::xtensor_fixed<value_type, xt::xshape<rows, cols>>;
        };

        /**
         * Template specialization: if rows=cols=1, then just a scalar coefficient
         */
        template <class value_type>
        struct LocalMatrix<value_type, 1, 1>
        {
            using Type = value_type;
        };
    }

    template <class matrix_type>
    matrix_type eye()
    {
        static constexpr auto s = typename matrix_type::shape_type();
        return xt::eye(s[0]);
    }

    template <>
    double eye<double>()
    {
        return 1;
    }

    template <class matrix_type>
    matrix_type zeros()
    {
        matrix_type mat;
        mat.fill(0);
        return mat;
    }

    template <>
    double zeros<double>()
    {
        return 0;
    }

    /*------------------------------------------------------------*/

    enum class FluxType
    {
        NonLinear,
        LinearHomogeneous,
        LinearHeterogeneous
    };

    template <FluxType flux_type_, std::size_t output_field_size_, std::size_t stencil_size_ = 2>
    struct FluxBasedSchemeConfig
    {
        static constexpr FluxType flux_type            = flux_type_;
        static constexpr std::size_t output_field_size = output_field_size_;
        static constexpr std::size_t stencil_size      = stencil_size_;
    };

    template <std::size_t dim, class cfg>
    struct NormalFluxDefinitionBase
    {
        /**
         * Direction of the flux.
         * In 2D, e.g., {1,0} for the flux on the right.
         */
        DirectionVector<dim> direction;

        /**
         * Stencil for the flux computation of the flux in the direction defined above.
         * E.g., if direction = {1,0}, the standard stencil is {{0,0}, {1,0}}.
         * Here, {0,0} captures the current cell and {1,0} its right neighbour.
         * The flux will be computed from {0,0} to {1,0}:
         *
         *                |-------|-------|
         *                | {0,0} | {1,0} |
         *                |-------|-------|
         *                     ------->
         *                    normal flux
         *
         * An enlarged stencil would be {{-1,0}, {0,0}, {1,0}, {1,0}}, i.e. two cells on each side of the interface.
         *
         *       |-------|-------|-------|-------|
         *       |{-1,0} | {0,0} | {1,0} | {2,0} |
         *       |-------|-------|-------|-------|
         *                    ------->
         *                   normal flux
         *
         */
        Stencil<cfg::stencil_size, dim> stencil;
    };

    /**
     * @class NormalFluxDefinition defines how to compute a normal flux.
     * This struct inherits from @class NormalFluxDefinitionBase and is specialized for all flux types (see below).
     */
    template <class cfg, class Field, class enable = void>
    struct NormalFluxDefinition
    {
    };

    /**
     * Specialization of @class NormalFluxDefinition.
     * Defines how to compute a NON-LINEAR normal flux.
     */
    template <class cfg, class Field>
    struct NormalFluxDefinition<cfg, Field, std::enable_if_t<cfg::flux_type == FluxType::NonLinear>>
        : NormalFluxDefinitionBase<Field::dim, cfg>
    {
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type;

        using cell_t          = typename Field::cell_t;
        using stencil_cells_t = std::array<cell_t, cfg::stencil_size>;
        using flux_value_t    = typename detail::LocalMatrix<field_value_type, cfg::output_field_size, 1>::Type;
        using flux_func       = std::function<flux_value_t(stencil_cells_t&, Field&)>;

        flux_func flux_function;

        ~NormalFluxDefinition()
        {
            flux_function = nullptr;
        }
    };

    /**
     * Specialization of @class NormalFluxDefinition.
     * Defines how to compute a LINEAR and HOMOGENEOUS normal flux.
     */
    template <class cfg, class Field>
    struct NormalFluxDefinition<cfg, Field, std::enable_if_t<cfg::flux_type == FluxType::LinearHomogeneous>>
        : NormalFluxDefinitionBase<Field::dim, cfg>
    {
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type;

        using flux_coeff_matrix_t   = typename detail::LocalMatrix<field_value_type, cfg::output_field_size, field_size>::Type;
        using flux_stencil_coeffs_t = xt::xtensor_fixed<flux_coeff_matrix_t, xt::xshape<cfg::stencil_size>>;
        using flux_func             = std::function<flux_stencil_coeffs_t(double)>;

        /**
         * Function returning the coefficients for the computation of the flux w.r.t. the defined stencil, in function of the meshsize h.
         * Note that in this definition, the flux must be linear with respect to the cell values.
         * For instance, considering a scalar field u, we configure the flux Grad(u).n through the function
         *
         *            // Grad(u).n = (u_1 - u_0)/h
         *            auto flux_function(double h)
         *            {
         *                std::array<double, 2> coeffs;
         *                coeffs[0] = -1/h; // current cell    (because, stencil[0] = {0,0})
         *                coeffs[1] =  1/h; // right neighbour (because, stencil[1] = {1,0})
         *                return coeffs;
         *            }
         * If u is now a vectorial field of size S, then coeffs[0] and coeffs[1] become matrices of size SxS.
         * If the field components are independent from each other, then
         *                coeffs[0] = diag(-1/h),
         *                coeffs[1] = diag( 1/h).
         */
        flux_func flux_function;

        ~NormalFluxDefinition()
        {
            flux_function = nullptr;
        }
    };

    /**
     * @class FluxDefinition:
     * Stores one object of @class NormalFluxDefinition for each positive Cartesian direction.
     */
    template <class cfg, class Field>
    class FluxDefinition
    {
      public:

        static constexpr std::size_t dim          = Field::dim;
        static constexpr std::size_t stencil_size = cfg::stencil_size;
        using cfg_stencil2                        = FluxBasedSchemeConfig<cfg::flux_type, cfg::output_field_size, 2>;
        using flux_computation_t                  = NormalFluxDefinition<cfg, Field>;
        using flux_computation_stencil2_t         = NormalFluxDefinition<cfg_stencil2, Field>;

      private:

        std::array<flux_computation_t, dim> m_normal_fluxes;

      public:

        FluxDefinition()
        {
            auto directions = positive_cartesian_directions<dim>();
            for (std::size_t d = 0; d < dim; ++d)
            {
                DirectionVector<dim> direction = xt::view(directions, d);
                m_normal_fluxes[d].direction   = direction;
                if constexpr (stencil_size == 2)
                {
                    m_normal_fluxes[d].stencil = in_out_stencil<dim>(direction); // TODO: stencil for any stencil size
                }
                m_normal_fluxes[d].flux_function = nullptr; // to be set by the user
            }
        }

        /**
         * This constructor sets the same flux function for all directions
         */
        explicit FluxDefinition(typename flux_computation_stencil2_t::flux_func flux_implem)
        {
            static_assert(stencil_size == 2, "stencil_size = 2 required to use this constructor.");

            auto directions = positive_cartesian_directions<dim>();
            for (std::size_t d = 0; d < dim; ++d)
            {
                DirectionVector<dim> direction   = xt::view(directions, d);
                m_normal_fluxes[d].direction     = direction;
                m_normal_fluxes[d].stencil       = in_out_stencil<dim>(direction);
                m_normal_fluxes[d].flux_function = flux_implem;
            }
        }

        flux_computation_t& operator[](std::size_t d)
        {
            assert(d < dim);
            return m_normal_fluxes[d];
        }

        const flux_computation_t& operator[](std::size_t d) const
        {
            assert(d < dim);
            return m_normal_fluxes[d];
        }
    };

    template <class cfg, class Field>
    using FluxValue = typename NormalFluxDefinition<cfg, std::decay_t<Field>>::flux_value_t;

    template <class cfg, class Field>
    using FluxStencilCoeffs = typename NormalFluxDefinition<cfg, std::decay_t<Field>>::flux_stencil_coeffs_t;

} // end namespace samurai
