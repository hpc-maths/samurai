#pragma once
#include "../../boundary.hpp"
#include "flux_definition.hpp"

namespace samurai
{
    enum DirichletEnforcement : int
    {
        Equation,
        Elimination
    };

    template <std::size_t neighbourhood_width_, DirichletEnforcement dirichlet_enfcmt_ = Equation>
    struct BoundaryConfigFV
    {
        static constexpr std::size_t neighbourhood_width       = neighbourhood_width_;
        static constexpr std::size_t stencil_size              = 1 + 2 * neighbourhood_width;
        static constexpr std::size_t nb_ghosts                 = neighbourhood_width;
        static constexpr DirichletEnforcement dirichlet_enfcmt = dirichlet_enfcmt_;
    };

    /**
     * Definition of one ghost equation to enforce the boundary condition.
     */
    template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size>
    struct BoundaryEquationCoeffs
    {
        static constexpr std::size_t field_size = Field::size;
        using field_value_type                  = typename Field::value_type; // double
        using coeffs_t                          = typename detail::LocalMatrix<field_value_type, output_field_size, field_size>::Type;

        using stencil_coeffs_t = std::array<coeffs_t, bdry_stencil_size>;
        using rhs_coeffs_t     = coeffs_t;

        // Index of the ghost in the boundary stencil. The equation coefficients will be added on its row.
        std::size_t ghost_index;
        // Coefficients of the equation
        stencil_coeffs_t stencil_coeffs;
        // Coefficients of the right-hand side
        rhs_coeffs_t rhs_coeffs;
    };

    /**
     * Definition of one ghost equation to enforce the boundary condition.
     * It contains functions depending on h to get the equation coefficients.
     */
    template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size>
    struct BoundaryEquationConfig
    {
        using equation_coeffs_t         = BoundaryEquationCoeffs<Field, output_field_size, bdry_stencil_size>;
        using stencil_coeffs_t          = typename equation_coeffs_t::stencil_coeffs_t;
        using rhs_coeffs_t              = typename equation_coeffs_t::rhs_coeffs_t;
        using get_stencil_coeffs_func_t = std::function<stencil_coeffs_t(double)>;
        using get_rhs_coeffs_func_t     = std::function<rhs_coeffs_t(double)>;

        // Index of the ghost in the boundary stencil. The equation coefficients will be added on its row.
        std::size_t ghost_index;
        // Function to get the coefficients of the equation
        get_stencil_coeffs_func_t get_stencil_coeffs;
        // Function to get the coefficients of the right-hand side
        get_rhs_coeffs_func_t get_rhs_coeffs;
    };

    /**
     * For a boundary direction, defines one equation per boundary ghost.
     */
    template <class Field, std::size_t output_field_size, std::size_t bdry_stencil_size, std::size_t nb_bdry_ghosts>
    struct DirectionalBoundaryConfig
    {
        static constexpr std::size_t dim = Field::dim;
        using bdry_equation_config_t     = BoundaryEquationConfig<Field, output_field_size, bdry_stencil_size>;

        // Direction of the boundary and stencil for the computation of the boundary condition
        DirectionalStencil<bdry_stencil_size, dim> directional_stencil;
        // One equation per boundary ghost
        std::array<bdry_equation_config_t, nb_bdry_ghosts> equations;
    };

    /**
     * Finite Volume scheme.
     * This is the base class of CellBasedScheme and FluxBasedSchemeAssembly.
     * It contains the management of
     *     - the boundary conditions
     *     - the projection/prediction ghosts
     *     - the unused ghosts
     */
    template <class DerivedScheme, class Field, std::size_t output_field_size_, class bdry_cfg_>
    class FVScheme
    {
      public:

        using Mesh                                             = typename Field::mesh_t;
        using mesh_id_t                                        = typename Mesh::mesh_id_t;
        using interval_t                                       = typename Mesh::interval_t;
        using field_value_type                                 = typename Field::value_type; // double
        using bdry_cfg                                         = bdry_cfg_;
        static constexpr std::size_t dim                       = Field::dim;
        static constexpr std::size_t field_size                = Field::size;
        static constexpr std::size_t output_field_size         = output_field_size_;
        static constexpr std::size_t prediction_order          = Mesh::config::prediction_order;
        static constexpr std::size_t bdry_neighbourhood_width  = bdry_cfg::neighbourhood_width;
        static constexpr std::size_t bdry_stencil_size         = bdry_cfg::stencil_size;
        static constexpr std::size_t nb_bdry_ghosts            = bdry_cfg::nb_ghosts;
        static constexpr DirichletEnforcement dirichlet_enfcmt = bdry_cfg::dirichlet_enfcmt;

        using dirichlet_t = Dirichlet<Field>;
        using neumann_t   = Neumann<Field>;

        using directional_bdry_config_t = DirectionalBoundaryConfig<Field, output_field_size, bdry_stencil_size, nb_bdry_ghosts>;

      protected:

        std::string m_name = "(unnamed)";

      public:

        FVScheme()
        {
        }

        std::string name() const
        {
            return m_name;
        }

        void set_name(const std::string& name)
        {
            m_name = name;
        }

        inline DerivedScheme& derived_cast() & noexcept
        {
            return *static_cast<DerivedScheme*>(this);
        }

        inline const DerivedScheme& derived_cast() const& noexcept
        {
            return *static_cast<const DerivedScheme*>(this);
        }

        inline DerivedScheme derived_cast() && noexcept
        {
            return *static_cast<DerivedScheme*>(this);
        }

        virtual ~FVScheme()
        {
        }

        template <class Coeffs>
        inline static double cell_coeff(const Coeffs& coeffs,
                                        std::size_t cell_number_in_stencil,
                                        [[maybe_unused]] std::size_t field_i,
                                        [[maybe_unused]] std::size_t field_j)
        {
            if constexpr (field_size == 1 && output_field_size == 1)
            {
                return coeffs[cell_number_in_stencil];
            }
            else
            {
                return coeffs[cell_number_in_stencil](field_i, field_j);
            }
        }

        //-------------------------------------------------------------//
        //      Configuration of the BC stencils and equations         //
        //-------------------------------------------------------------//

        auto get_directional_stencil(const DirectionVector<dim>& direction) const
        {
            auto dir_stencils = directional_stencils<dim, bdry_neighbourhood_width>();
            for (std::size_t d = 0; d < 2 * dim; ++d)
            {
                if (direction == dir_stencils[d].direction)
                {
                    return dir_stencils[d];
                }
            }
            assert(false);
            return dir_stencils[0];
        }

        virtual directional_bdry_config_t dirichlet_config(const DirectionVector<dim>& direction) const
        {
            using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;
            directional_bdry_config_t config;

            config.directional_stencil = get_directional_stencil(direction);

            if constexpr (bdry_neighbourhood_width == 1)
            {
                static constexpr std::size_t cell          = 0;
                static constexpr std::size_t interior_cell = 1;
                static constexpr std::size_t ghost         = 2;

                // We have (u_ghost + u_cell)/2 = dirichlet_value, so the coefficient equation is
                //                        [  1/2    1/2 ] = dirichlet_value
                config.equations[0].ghost_index        = ghost;
                config.equations[0].get_stencil_coeffs = [&](double)
                {
                    std::array<coeffs_t, bdry_stencil_size> coeffs;
                    coeffs[cell]          = 0.5 * eye<coeffs_t>();
                    coeffs[ghost]         = 0.5 * eye<coeffs_t>();
                    coeffs[interior_cell] = zeros<coeffs_t>();
                    return coeffs;
                };
                config.equations[0].get_rhs_coeffs = [&](double)
                {
                    coeffs_t coeffs = eye<coeffs_t>();
                    return coeffs;
                };
            }

            return config;
        }

        virtual directional_bdry_config_t neumann_config(const DirectionVector<dim>& direction) const
        {
            using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;
            directional_bdry_config_t config;

            config.directional_stencil = get_directional_stencil(direction);

            if constexpr (bdry_neighbourhood_width == 1)
            {
                static constexpr std::size_t cell          = 0;
                static constexpr std::size_t interior_cell = 1;
                static constexpr std::size_t ghost         = 2;

                // The outward flux is (u_ghost - u_cell)/h = neumann_value, so the coefficient equation is
                //                    [ 1/h  -1/h ] = neumann_value
                config.equations[0].ghost_index        = ghost;
                config.equations[0].get_stencil_coeffs = [&](double)
                {
                    std::array<coeffs_t, bdry_stencil_size> coeffs;
                    coeffs[cell]          = -eye<coeffs_t>();
                    coeffs[ghost]         = eye<coeffs_t>();
                    coeffs[interior_cell] = zeros<coeffs_t>();
                    return coeffs;
                };
                config.equations[0].get_rhs_coeffs = [&](double h)
                {
                    coeffs_t coeffs = h * eye<coeffs_t>();
                    return coeffs;
                };
            }

            return config;
        }

        /**
         * @brief Is the matrix symmetric?
         */
        virtual bool matrix_is_symmetric(const Field&) const
        {
            return false;
        }

        /**
         * @brief Is the matrix symmetric positive-definite?
         */
        virtual bool matrix_is_spd(const Field&) const
        {
            return false;
        }
    };

} // end namespace samurai
