#pragma once
#include "../../bc.hpp"
#include "../../boundary.hpp"
#include "../../field.hpp"
#include "../../static_algorithm.hpp"
#include "utils.hpp"

namespace samurai
{
    enum DirichletEnforcement : int
    {
        Equation,
        Elimination
    };

    template <std::size_t neighbourhood_width_ = 1, DirichletEnforcement dirichlet_enfcmt_ = Equation>
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
    template <class Field, std::size_t output_field_size_, class bdry_cfg_>
    class FVScheme
    {
      public:

        using Mesh                                            = typename Field::mesh_t;
        using mesh_id_t                                       = typename Mesh::mesh_id_t;
        using interval_t                                      = typename Mesh::interval_t;
        using field_value_type                                = typename Field::value_type; // double
        using bdry_cfg                                        = bdry_cfg_;
        static constexpr std::size_t dim                      = Field::dim;
        static constexpr std::size_t field_size               = Field::size;
        static constexpr std::size_t output_field_size        = output_field_size_;
        static constexpr std::size_t bdry_neighbourhood_width = bdry_cfg::neighbourhood_width;
        static constexpr std::size_t bdry_stencil_size        = bdry_cfg::stencil_size;
        static constexpr std::size_t nb_bdry_ghosts           = bdry_cfg::nb_ghosts;

        using dirichlet_t = Dirichlet<Field>;
        using neumann_t   = Neumann<Field>;

        using directional_bdry_config_t = DirectionalBoundaryConfig<Field, output_field_size, bdry_stencil_size, nb_bdry_ghosts>;
        using bdry_stencil_coeffs_t     = typename directional_bdry_config_t::bdry_equation_config_t::stencil_coeffs_t;

      private:

        std::string m_name  = "(unnamed)";
        bool m_is_symmetric = false;
        bool m_is_spd       = false;

        std::array<directional_bdry_config_t, 2 * dim> m_dirichlet_config;
        std::array<directional_bdry_config_t, 2 * dim> m_neumann_config;

      public:

        FVScheme()
        {
            init_dirichlet_config();
            init_neumann_config();
        }

        std::string name() const
        {
            return m_name;
        }

        void set_name(const std::string& name)
        {
            m_name = name;
        }

        virtual ~FVScheme()
        {
            m_name += " (deleted)";
        }

        inline double bdry_cell_coeff(const bdry_stencil_coeffs_t& coeffs,
                                      std::size_t cell_number_in_stencil,
                                      [[maybe_unused]] std::size_t field_i,
                                      [[maybe_unused]] std::size_t field_j) const
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

        auto& dirichlet_config()
        {
            return m_dirichlet_config;
        }

        auto& neumann_config()
        {
            return m_neumann_config;
        }

        auto& dirichlet_config(const DirectionVector<dim>& direction) const
        {
            for (std::size_t d = 0; d < 2 * dim; ++d)
            {
                if (direction == m_dirichlet_config[d].directional_stencil.direction)
                {
                    return m_dirichlet_config[d];
                }
            }
            std::cerr << "No Dirichlet config found for direction " << direction << std::endl;
            assert(false);
            return m_dirichlet_config[0];
        }

        void init_dirichlet_config()
        {
            using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;

            auto dir_stencils = directional_stencils<dim, bdry_neighbourhood_width>();
            for (std::size_t d = 0; d < 2 * dim; ++d)
            {
                auto& config = m_dirichlet_config[d];

                config.directional_stencil = dir_stencils[d];
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
            }
        }

        auto& neumann_config(const DirectionVector<dim>& direction) const
        {
            for (std::size_t d = 0; d < 2 * dim; ++d)
            {
                if (direction == m_neumann_config[d].directional_stencil.direction)
                {
                    return m_neumann_config[d];
                }
            }
            std::cerr << "No Neumann config found for direction " << direction << std::endl;
            assert(false);
            return m_neumann_config[0];
        }

        void init_neumann_config()
        {
            using coeffs_t = typename directional_bdry_config_t::bdry_equation_config_t::equation_coeffs_t::coeffs_t;

            auto dir_stencils = directional_stencils<dim, bdry_neighbourhood_width>();
            for (std::size_t d = 0; d < 2 * dim; ++d)
            {
                auto& config = m_neumann_config[d];

                config.directional_stencil = dir_stencils[d];
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
            }
        }

        /**
         * @brief Is the operator symmetric?
         */
        bool is_symmetric() const
        {
            return m_is_symmetric;
        }

        /**
         * @brief Is the operator symmetric positive-definite?
         */
        bool is_spd() const
        {
            return m_is_spd;
        }

        void is_symmetric(bool value)
        {
            m_is_symmetric = value;
        }

        void is_spd(bool value)
        {
            m_is_spd = value;
        }
    };

} // end namespace samurai
