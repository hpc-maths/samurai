// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <iostream>
#include <samurai/amr/mesh.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>

static char help[] = "Solution of the Poisson problem in the domain [0,1]^d.\n"
                     "Geometric multigrid using the samurai meshes.\n"
                     "\n"
                     "-------- General\n"
                     "\n"
                     "--level        <int>           Level used to set the problem size\n"
                     "--tc           <enum>          Test case:\n"
                     "                                   poly  - The solution is a polynomial "
                     "function.\n"
                     "                                           Homogeneous Dirichlet b.c.\n"
                     "                                   exp   - The solution is an exponential "
                     "function.\n"
                     "                                           Non-homogeneous Dirichlet "
                     "b.c.\n"
                     "--save_sol     [0|1]           Save solution (default 0)\n"
                     "--save_mesh    [0|1]           Save mesh (default 0)\n"
                     "--path         <string>        Output path\n"
                     "--filename     <string>        Solution file name\n"
                     "\n"
                     "-------- Samurai Multigrid ('-pc_type mg' to activate)\n"
                     "\n"
                     "--samg_smooth       <enum>     Smoother used in the samurai multigrid:\n"
                     "                                   sgs   - symmetric Gauss-Seidel\n"
                     "                                   gs    - Gauss-Seidel (pre: lexico., "
                     "post: antilexico.)\n"
                     "                                   petsc - defined by Petsc options "
                     "(default: Chebytchev polynomials)\n"
                     "--samg_transfer_ops [1:4]      Samurai multigrid transfer operators "
                     "(default: 1):\n"
                     "                                   1 - P assembled, R assembled\n"
                     "                                   2 - P assembled, R = P^T\n"
                     "                                   3 - P mat-free, R mat-free (via "
                     "double*)\n"
                     "                                   4 - P mat-free, R mat-free (via "
                     "Fields)\n"
                     "--samg_pred_order   [0|1]      Prediction order used in the prolongation "
                     "operator\n"
                     "\n"
                     "-------- Useful Petsc options\n"
                     "\n"
                     "-pc_type [mg|gamg|hypre...]    Sets the preconditioner ('mg' for the "
                     "samurai multigrid)\n"
                     "-ksp_monitor ascii             Prints the residual at each iteration\n"
                     "-ksp_view ascii                View the solver's parametrization\n"
                     "-ksp_rtol          <double>    Sets the solver tolerance\n"
                     "-ksp_max_it        <int>       Sets the maximum number of iterations\n"
                     "-pc_mg_levels      <int>       Sets the number of multigrid levels\n"
                     "-mg_levels_up_pc_sor_its <int> Sets the number of post-smoothing "
                     "iterations\n"
                     "-log_view -pc_mg_log           Monitors the multigrid performance\n"
                     "\n";

static constexpr double pi = M_PI;

template <class Mesh>
Mesh create_uniform_mesh(std::size_t level)
{
    using Box = samurai::Box<double, Mesh::dim>;

    Box box;
    if constexpr (Mesh::dim == 1)
    {
        box = Box({0}, {1});
    }
    else if constexpr (Mesh::dim == 2)
    {
        box = Box({0, 0}, {1, 1});
    }
    else if constexpr (Mesh::dim == 3)
    {
        box = Box({0, 0, 0}, {1, 1, 1});
    }
    std::size_t start_level, min_level, max_level;
    start_level = level;
    min_level   = level;
    max_level   = level;

    return Mesh(box, start_level, min_level, max_level); // amr::Mesh
    // return Mesh(box, /*start_level,*/ min_level, max_level); // MRMesh
}

template<class Field, std::size_t dim=Field::dim, class cfg=samurai::petsc::StarStencilFV<dim, Field::size*dim, 1>>
class GradientFV_old : public samurai::petsc::CellBasedScheme<cfg, Field>
{
  public:

    using local_matrix_t = typename samurai::petsc::CellBasedScheme<cfg, Field>::local_matrix_t;

    GradientFV_old(Field& u) : 
        samurai::petsc::CellBasedScheme<cfg, Field>(u, samurai::star_stencil<dim>(), coefficients)
    {}

    static std::array<local_matrix_t, 5> coefficients(double h)
    {
        static constexpr unsigned int L = 0; // left
        static constexpr unsigned int C = 1; // center
        static constexpr unsigned int R = 2; // right
        static constexpr unsigned int B = 3; // bottom
        static constexpr unsigned int T = 4; // top

        static constexpr unsigned int x = 0;
        static constexpr unsigned int y = 1;

        // We have:
        // Grad_x(u) = 1/2 * [ (u_{R} - u_{C})/h + (u_{C} - u_{L})/h ]
        //           = 1/(2h) * (u_{R} - u_{L})
        // Grad_y(u) = 1/2 * [ (u_{T} - u_{C})/h + (u_{C} - u_{B})/h ]
        //           = 1/(2h) * (u_{T} - u_{B})
        //
        // The coefficient array is:
        //                           L  C  R  B  T
        //     Grad_x --> 1/(2h) * |-1|  | 1|  |  |
        //     Grad_y --> 1/(2h) * |  |  |  |-1| 1|
        //
        std::array<local_matrix_t, 5> coeffs;
        double one_over_2h = 1 / (2 * h);

        xt::view(coeffs[L], x) = -one_over_2h;
        xt::view(coeffs[C], x) = 0;
        xt::view(coeffs[R], x) = one_over_2h;
        xt::view(coeffs[B], x) = 0;
        xt::view(coeffs[T], x) = 0;

        xt::view(coeffs[L], y) = 0;
        xt::view(coeffs[C], y) = 0;
        xt::view(coeffs[R], y) = 0;
        xt::view(coeffs[B], y) = -one_over_2h;
        xt::view(coeffs[T], y) = one_over_2h;

        return coeffs;
    }
};

template<class Field>
auto make_gradient_FV_old(Field& f)
{
    return GradientFV_old<Field>(f);
}

template<class Field, std::size_t dim=Field::dim, class cfg=samurai::petsc::StarStencilFV<dim, 1, 1>>
class MinusDivergenceFV_old : public samurai::petsc::CellBasedScheme<cfg, Field>
{
  public:

    using local_matrix_t = typename samurai::petsc::CellBasedScheme<cfg, Field>::local_matrix_t;

    MinusDivergenceFV_old(Field& u) : 
        samurai::petsc::CellBasedScheme<cfg, Field>(u, samurai::star_stencil<dim>(), coefficients)
    {}

    static std::array<local_matrix_t, 5> coefficients(double h)
    {
        static constexpr unsigned int L = 0; // left
        static constexpr unsigned int C = 1; // center
        static constexpr unsigned int R = 2; // right
        static constexpr unsigned int B = 3; // bottom
        static constexpr unsigned int T = 4; // top

        static constexpr unsigned int x = 0;
        static constexpr unsigned int y = 1;

        // Let F be a vector field (Fx, Fy), such as a gradient for instance.
        // We have:
        // Div(F) =              (Fx_{L} + Fx_{R})/2             +        (Fy_{B} + Fy_{T})/2
        //        = 1/(2h) * [(u_{C} - u_{L}) + (u_{R} - u_{C}) + (u_{C} - u_{B}) + (u_{T} - u_{C})]
        //        = 1/(2h) * [u_{R} - u_{L} + u_{T} - u_{B}]
        //
        // The coefficient array is:
        //                             L     C     R     B     T
        //                           Fx Fy Fx Fy Fx Fy Fx Fy Fx Fy
        //     Div     --> 1/(2h) * |-1  0| 0  0| 1  0| 0 -1| 0  1|
        //
        std::array<local_matrix_t, 5> coeffs;
        double one_over_2h = 1 / (2 * h);

        coeffs[L][x] = -one_over_2h * (-1);
        coeffs[L][y] = 0;

        coeffs[C][x] = 0;
        coeffs[C][y] = 0;

        coeffs[R][x] = one_over_2h * (-1);
        coeffs[R][y] = 0;

        coeffs[B][x] = 0;
        coeffs[B][y] = -one_over_2h * (-1);

        coeffs[T][x] = 0;
        coeffs[T][y] = one_over_2h * (-1);

        return coeffs;
    }
};

template<class Field>
auto make_minus_divergence_FV_old(Field& f)
{
    return MinusDivergenceFV_old<Field>(f);
}


/*****************************************************************/

/**
 * If u is a scalar field in dimension 2, then
 *      Grad(u) = [d(u)/dx]
 *                [d(u)/dy].
 * On each cell, we adopt a cell-centered approximation:
 *         d(u)/dx = 1/2 [(u^R - u)/h + (u - u^L)/h]   where (L, R) = (Left, Right)
 *         d(u)/dy = 1/2 [(u^T - u)/h + (u - u^B)/h]   where (B, T) = (Bottom, Top).
 * We denote by Fx^f = d(u)/dx * n_f the outer normal flux through the face f, i.e.
 *         Fx^R = (u^R - u)/h,      Fx^L = (u^L - u)/h,
 *         Fy^T = (u^T - u)/h,      Fy^B = (u^B - u)/h.
 * The approximations become
 *         d(u)/dx = 1/2 (Fx^R - Fx^L)        (S)
 *         d(u)/dy = 1/2 (Fy^T - Fy^B).
 * 
 * Implementation:
 * 
 * 1. The computation of the normal fluxes between cell 1 and cell 2 (in the direction dir=x or y) is given by
 *         Fx = (u^2 - u^1)/h = -1/h * u^1 + 1/h * u^2    if dir=x
 *         Fy = (u^2 - u^1)/h = -1/h * u^1 + 1/h * u^2    if dir=y
 * 
 *    So   F = [ -1/h |  1/h ] whatever the direction
 *             |______|______|
 *              cell 1 cell 2
 * 
 * 2. On each couple (cell 1, cell 2), we compute Fx^R(1) or Fx^T(1) (according to the direction) and consider that
 *         Fx^L(1) = -Fx^R(2) 
 *         Fy^B(1) = -Fy^T(2).
 *    The gradient scheme (S) becomes
 *         Grad(u)(cell 1) = [1/2 (Fx^R(1) + Fx^L(2))]
 *                           [1/2 (Fy^T(1) + Fy^B(2))], where 2 denotes the neighbour in the appropriate direction.
 *    So the contribution of a flux F (R or T) computed on cell 1 is 
 *         for cell 1: [1/2 F] if dir=x,  [    0] if dir=y
 *                     [    0]            [1/2 F]
 *         for cell 2: [1/2 F] if dir=x,  [    0] if dir=y
 *                     [    0]            [1/2 F]
*/
template<class Field, std::size_t dim=Field::dim, class cfg=samurai::petsc::FluxBasedAssemblyConfig<dim, 2>>
class GradientFV : public samurai::petsc::FluxBasedScheme<cfg, Field>
{
public:
    using flux_computation_t = typename samurai::petsc::FluxBasedScheme<cfg, Field>::flux_computation_t;
    using coeff_matrix_t = typename flux_computation_t::coeff_matrix_t;

    GradientFV(Field& u) : 
        samurai::petsc::FluxBasedScheme<cfg, Field>(u, scheme_coefficients())
    {
        static_assert(Field::size == 1, "The field put in the gradient operator must be a scalar field.");
    }

    static auto flux_coefficients(double h)
    {
        std::array<double, 2> coeffs;
        coeffs[0] = -1/h;
        coeffs[1] =  1/h;
        return coeffs;
    }

    template<std::size_t d>
    static auto half_flux_in_direction(std::array<double, 2>& flux_coeffs, double, double)
    {
        std::array<coeff_matrix_t, 2> coeffs;
        xt::view(coeffs[0], d) = 0.5 * flux_coeffs[0];
        xt::view(coeffs[1], d) = 0.5 * flux_coeffs[1];
        for (std::size_t d_ = 0; d_ < dim; ++d_)
        {
            if (d_ != d)
            {
                xt::view(coeffs[0], d_) = 0;
                xt::view(coeffs[1], d_) = 0;
            }
        }
        return coeffs;
    }

    // Grad_x(u) = 1/2 * [ Fx(L) + Fx(R) ]
    // Grad_y(u) = 1/2 * [ Fx(B) + Fx(T) ]
    static auto scheme_coefficients()
    {
        static_assert(dim <= 3, "GradientFV.scheme_coefficients() not implemented for dim > 3.");
        std::array<flux_computation_t, dim> fluxes;
        auto directions = samurai::positive_cartesian_directions<dim>();
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto& flux = fluxes[d];
            flux.direction = xt::view(directions, d);
            flux.computational_stencil = samurai::in_out_stencil<dim>(flux.direction);
            flux.get_flux_coeffs = flux_coefficients;
            // Set coeffs only in the direction d.
            // If d=0 (direction x), then set Grad_x only.
            // If d=1 (direction y), then set Grad_y only.
            if (d == 0)
            {
                flux.get_cell1_coeffs = half_flux_in_direction<0>;
                flux.get_cell2_coeffs = half_flux_in_direction<0>;
            }
            if constexpr (dim >= 2)
            {
                if (d == 1)
                {
                    flux.get_cell1_coeffs = half_flux_in_direction<1>;
                    flux.get_cell2_coeffs = half_flux_in_direction<1>;
                }
            }
            if constexpr (dim >= 3)
            {
                if (d == 2)
                {
                    flux.get_cell1_coeffs = half_flux_in_direction<2>;
                    flux.get_cell2_coeffs = half_flux_in_direction<2>;
                }
            }
        }
        return fluxes;
    }
};

template<class Field>
auto make_gradient_FV(Field& f)
{
    return GradientFV<Field>(f);
}

/**
 * If u is a field of size 2, e.g. the velocity --> u = (u_x, u_y), then
 *         Div(u) = d(u_x)/dx + d(u_y)/dy.
 * On each cell, we adopt a cell-centered approximation:
 *         d(u_x)/dx = 1/2 [(u_x^R - u_x)/h + (u_x - u_x^L)/h]   where (L, R) = (Left, Right)
 *         d(u_y)/dy = 1/2 [(u_y^T - u_y)/h + (u_y - u_x^B)/h]   where (B, T) = (Bottom, Top).
 * We denote by Fx^f = d(u_x)/dx * n_f the outer normal flux through the face f, i.e.
 *         Fx^R = (u_x^R - u_x)/h,      Fx^L = (u_x^L - u_x)/h,
 *         Fy^T = (u_y^T - u_y)/h,      Fy^B = (u_y^B - u_y)/h.
 * The approximations become
 *         d(u_x)/dx = 1/2 (Fx^R - Fx^L)
 *         d(u_y)/dy = 1/2 (Fy^T - Fy^B).
 * and finally,
 *         Div(u) = 1/2 (Fx^R - Fx^L) + 1/2 (Fy^T - Fy^B)      (S)
 * 
 * Implementation:
 * 
 * 1. The computation of the normal fluxes between cell 1 and cell 2 (in the direction d=x or y) is given by
 *         Fx = (u_x^2 - u_x^1)/h = -1/h * u_x^1 + 1/h * u_x^2 +  0 * u_y^1 +   0 * u_y^2    if d=x
 *         Fy = (u_y^2 - u_y^1)/h =    0 * u_x^1 +   0 * u_x^2 -1/h * u_y^1 + 1/h * u_x^2    if d=y
 * 
 *    So   F = [-1/h   0 | 1/h   0  ] if d=x
 *         F = [  0  -1/h|  0   1/h ] if d=y
 *             |_________|__________|
 *               cell 1     cell 2
 * 
 * 2. On each couple (cell 1, cell 2), we compute Fx^R(1) or Fx^T(1) (according to the direction) and consider that
 *         Fx^L(1) = -Fx^R(2) 
 *         Fy^B(1) = -Fy^T(2).
 *    The divergence scheme (S) becomes
 *         Div(u)(cell 1) = 1/2 (Fx^R(1) + Fx^L(2)) + 1/2 (Fy^T(1) + Fy^B(2)), where 2 denotes the neighbour in the appropriate direction.
 *    So the contribution of a flux F (R or T) computed on cell 1 is 
 *         for cell 1: 1/2 F
 *         for cell 2: 1/2 F
*/
template<class Field, std::size_t dim=Field::dim, class cfg=samurai::petsc::FluxBasedAssemblyConfig<1, 2>>
class MinusDivergenceFV : public samurai::petsc::FluxBasedScheme<cfg, Field>
{
public:
    using flux_computation_t = typename samurai::petsc::FluxBasedScheme<cfg, Field>::flux_computation_t;
    using flux_matrix_t  = typename flux_computation_t::flux_matrix_t;
    using coeff_matrix_t = typename flux_computation_t::coeff_matrix_t;
    static constexpr std::size_t field_size = Field::size;

    MinusDivergenceFV(Field& u) : 
        samurai::petsc::FluxBasedScheme<cfg, Field>(u, scheme_coefficients())
    {
        static_assert(dim == field_size, "The field put into the divergence operator must have a size equal to the space dimension.");
    }

    /**
     * If u is a field of size 2, e.g. the velocity --> u = (u_x, u_y)
     * then we define the following normal flux from cell 1 to cell 2 (which can be direction x or direction y):
     *          Fx := (u_x^2 - u_x^1)/h
     *          Fy := (u_y^2 - u_y^1)/h
     * So we return the array of coefficients
     * 
     *          [-1/h -1/h][ 1/h  1/h]
     *             |    |     |    |
     *            Fx   Fy    Fx   Fy
     *          |_________||_________|
     *            cell 1      cell 2
    */

    template<std::size_t d>
    static auto flux_coefficients(double h)
    {
        std::array<flux_matrix_t, 2> flux_coeffs;
        if constexpr (field_size == 1)
        {
            flux_coeffs[0] = -1/h;
            flux_coeffs[1] =  1/h;
        }
        else
        {
            flux_coeffs[0].fill(0);
            flux_coeffs[1].fill(0);
            flux_coeffs[0](d) = -1/h;
            flux_coeffs[1](d) =  1/h;
        }
        return flux_coeffs;
    }

    static auto minus_half_flux_in_direction(std::array<flux_matrix_t, 2>& flux_coeffs, double, double)
    {
        std::array<coeff_matrix_t, 2> coeffs;
        coeffs[0] = -0.5 * flux_coeffs[0];
        coeffs[1] = -0.5 * flux_coeffs[1];
        return coeffs;
    }


    // Div(F) =  (Fx_{L} + Fx_{R}) / 2  +  (Fy_{B} + Fy_{T}) / 2
    static auto scheme_coefficients()
    {
        static_assert(dim <= 3, "MinusDivergenceFV.scheme_coefficients() not implemented for dim > 3.");
        std::array<flux_computation_t, dim> fluxes;
        auto directions = samurai::positive_cartesian_directions<dim>();
        for (std::size_t d = 0; d < dim; ++d)
        {
            auto& flux = fluxes[d];
            flux.direction = xt::view(directions, d);
            flux.computational_stencil = samurai::in_out_stencil<dim>(flux.direction);
            if (d == 0)
            {
                flux.get_flux_coeffs = flux_coefficients<0>;
            }
            if constexpr (dim >= 2)
            {
                if (d == 1)
                {
                    flux.get_flux_coeffs = flux_coefficients<1>;
                }
            }
            if constexpr (dim >= 3)
            {
                if (d == 2)
                {
                    flux.get_flux_coeffs = flux_coefficients<2>;
                }
            }
            flux.get_cell1_coeffs = minus_half_flux_in_direction;
            flux.get_cell2_coeffs = minus_half_flux_in_direction;
        }
        return fluxes;
    }
};

template<class Field>
auto make_minus_divergence_FV(Field& f)
{
    return MinusDivergenceFV<Field>(f);
}

/***********************************************************************/



int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 2;
    using Config              = samurai::amr::Config<dim>;
    using Mesh                = samurai::amr::Mesh<Config>;
    // using Config = samurai::MRConfig<dim>;
    // using Mesh = samurai::MRMesh<Config>;
    constexpr bool is_soa = true;

    //------------------//
    // Petsc initialize //
    //------------------//

    PetscInitialize(&argc, &argv, 0, help);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

    //----------------//
    //   Parameters   //
    //----------------//

    // Default values
    PetscInt level          = 2;
    PetscBool save_solution = PETSC_FALSE;
    PetscBool save_mesh     = PETSC_FALSE;
    fs::path path           = fs::current_path();
    std::string filename    = "velocity";

    // Get user options
    PetscOptionsGetInt(NULL, NULL, "--level", &level, NULL);

    PetscOptionsGetBool(NULL, NULL, "--save_sol", &save_solution, NULL);
    PetscOptionsGetBool(NULL, NULL, "--save_mesh", &save_mesh, NULL);

    PetscBool path_is_set = PETSC_FALSE;
    std::string path_str(100, '\0');
    PetscOptionsGetString(NULL, NULL, "--path", path_str.data(), path_str.size(), &path_is_set);
    if (path_is_set)
    {
        path = path_str.substr(0, path_str.find('\0'));
        if (!fs::exists(path))
        {
            fs::create_directory(path);
        }
    }

    PetscBool filename_is_set = PETSC_FALSE;
    std::string filename_str(100, '\0');
    PetscOptionsGetString(NULL, NULL, "--filename", filename_str.data(), filename_str.size(), &filename_is_set);
    if (path_is_set)
    {
        filename = filename_str.substr(0, filename_str.find('\0'));
    }

    //----------------//
    // Create problem //
    //----------------//

    // 2 equations: -Lap(v) + Grad(p) = f
    //              -Div(v)           = 0
    // where v = velocity
    //       p = pressure

    Mesh mesh = create_uniform_mesh<Mesh>(static_cast<std::size_t>(level));

    // Unknowns
    auto velocity = samurai::make_field<double, dim, is_soa>("velocity", mesh);
    auto pressure = samurai::make_field<double, 1, is_soa>("pressure", mesh);

    // Boundary conditions
    velocity
        .set_dirichlet(
            [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                double v_x    = 1 / (pi * pi) * sin(pi * (x + y));
                double v_y    = -v_x;
                return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
            })
        .everywhere();

    pressure
        .set_neumann(
            [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                int normal    = (x == 0 || y == 0) ? -1 : 1;
                return normal * (1 / pi) * cos(pi * (x + y));
            })
        .everywhere();

    // Block operator
    auto diff_v      = samurai::petsc::make_diffusion_FV(velocity);
    auto grad_p      = make_gradient_FV(pressure);
    auto minus_div_v = make_minus_divergence_FV(velocity);
    auto zero_p      = samurai::petsc::make_zero_operator_FV<1>(pressure);

    auto stokes = samurai::petsc::make_block_operator<2, 2>(diff_v, grad_p, minus_div_v, zero_p);

    // Right-hand side
    auto f = samurai::make_field<double, dim, is_soa>(
        "f",
        mesh,
        [](const auto& coord)
        {
            const auto& x = coord[0];
            const auto& y = coord[1];
            double f_x    = 2 * sin(pi * (x + y)) + (1 / pi) * cos(pi * (x + y));
            double f_y    = -2 * sin(pi * (x + y)) + (1 / pi) * cos(pi * (x + y));
            return xt::xtensor_fixed<double, xt::xshape<dim>>{f_x, f_y};
        },
        0);
    auto zero = samurai::make_field<double, 1, is_soa>("zero", mesh);
    zero.fill(0);

    //-------------------//
    //   Linear solver   //
    //-------------------//

    std::cout << "Solving Stokes system..." << std::endl;
    auto block_solver = samurai::petsc::make_solver(stokes);

    //
    // Configuration of the PETSc solver for the Stokes problem
    //

    // 1. Set the use of a Schur complement preconditioner eliminating the
    // velocity
    KSP ksp = block_solver.Ksp();
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCFIELDSPLIT); // (equiv. '-pc_type fieldsplit')
    PCFieldSplitSetType(pc,
                        PC_COMPOSITE_SCHUR); // Schur complement preconditioner (equiv.
                                             // '-pc_fieldsplit_type schur')
    PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELFP,
                            PETSC_NULL);                             // (equiv. '-pc_fieldsplit_schur_precondition selfp')
    PCFieldSplitSetSchurFactType(pc, PC_FIELDSPLIT_SCHUR_FACT_FULL); // (equiv.
                                                                     // '-pc_fieldsplit_schur_fact_type
                                                                     // full')

    // 2. Configure the sub-solvers
    block_solver.setup(); // must be called before using PCFieldSplitSchurGetSubKSP(),
                          // because the matrices are needed.
    KSP* sub_ksp;
    PCFieldSplitSchurGetSubKSP(pc, nullptr, &sub_ksp);
    KSP velocity_ksp = sub_ksp[0];
    KSP schur_ksp    = sub_ksp[1];
    // Set LU by default for the diffusion block. Consider using 'hypre' for
    // large problems, using the option '-fieldsplit_velocity_pc_type hypre'.
    PC velocity_pc;
    KSPGetPC(velocity_ksp, &velocity_pc);
    PCSetType(velocity_pc,
              PCLU);                 // (equiv. '-fieldsplit_velocity_pc_type lu' or 'hypre')
    KSPSetFromOptions(velocity_ksp); // overwrite by user value if needed
    // If a tolerance is set by the user ('-ksp-rtol XXX'), then we set that
    // tolerance to all the sub-solvers
    PetscReal ksp_rtol;
    KSPGetTolerances(ksp, &ksp_rtol, PETSC_NULL, PETSC_NULL, PETSC_NULL);
    KSPSetTolerances(velocity_ksp, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT,
                     PETSC_DEFAULT); // (equiv. '-fieldsplit_velocity_ksp_rtol XXX')
    KSPSetTolerances(schur_ksp, ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT,
                     PETSC_DEFAULT); // (equiv. '-fieldsplit_pressure_ksp_rtol XXX')

    //
    // Solve the system
    //
    block_solver.solve(f, zero);
    std::cout << block_solver.iterations() << " iterations" << std::endl << std::endl;

    //--------------------//
    //       Error        //
    //--------------------//

    std::cout.precision(2);

    double error = diff_v.L2Error(velocity,
                                  [](auto& coord)
                                  {
                                      const auto& x = coord[0];
                                      const auto& y = coord[1];
                                      auto v_x      = 1 / (pi * pi) * sin(pi * (x + y));
                                      auto v_y      = -v_x;
                                      return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
                                  });

    std::cout << "L2-error on the velocity: " << std::scientific << error << std::endl;

    //--------------------//
    //   Save solution    //
    //--------------------//

    if (save_solution)
    {
        std::cout << "Saving solution..." << std::endl;

        samurai::save(path, filename, mesh, velocity);
        samurai::save(path, "pressure", mesh, pressure);

        auto exact_velocity = samurai::make_field<double, dim, is_soa>(
            "exact_velocity",
            mesh,
            [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                auto v_x      = 1 / (pi * pi) * sin(pi * (x + y));
                auto v_y      = -v_x;
                return xt::xtensor_fixed<double, xt::xshape<dim>>{v_x, v_y};
            },
            0);
        samurai::save(path, "exact_velocity", mesh, exact_velocity);

        /*auto err = samurai::make_field<double, dim, is_soa>("error", mesh);
        for_each_cell(err.mesh(), [&](const auto& cell)
            {
                err[cell] = exact_velocity[cell] - velocity[cell];
            });
        samurai::save(path, "error_velocity", mesh, err);*/

        auto exact_pressure = samurai::make_field<double, 1, is_soa>(
            "exact_pressure",
            mesh,
            [](const auto& coord)
            {
                const auto& x = coord[0];
                const auto& y = coord[1];
                return 1 / (pi * pi) * sin(pi * (x + y));
            },
            0);
        samurai::save(path, "exact_pressure", mesh, exact_pressure);
    }

    //--------------------//
    //     Finalize       //
    //--------------------//

    // Destroy Petsc objects
    block_solver.destroy_petsc_objects();
    PetscFinalize();

    return 0;
}