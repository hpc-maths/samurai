#include <array>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/node.hpp>

#include <benchmark/benchmark.h>

template <class Field>
void init(Field& u)
{
    auto& mesh = u.mesh();
    u.resize();

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               auto center           = cell.center();
                               const double radius   = .2;
                               const double x_center = 0.3;
                               // const double y_center = 0.3;
                               if (std::abs(center[0] - x_center) <= radius * radius)
                               {
                                   u[cell] = 1;
                               }
                               else
                               {
                                   u[cell] = 0;
                               }
                           });
}

template <class Field>
void flux_correction(double dt, const std::array<double, 2>& a, const Field& u, Field& unp1)
{
    using mesh_t              = typename Field::mesh_t;
    using mesh_id_t           = typename mesh_t::mesh_id_t;
    using interval_t          = typename mesh_t::interval_t;
    constexpr std::size_t dim = Field::dim;

    auto mesh = u.mesh();

    for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {
            {-1, 0}
        };

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                                .on(level);

        subset_right(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = mesh.cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  + dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).right_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j).right_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j + 1).right_flux(a, u));
            });

        stencil = {
            {1, 0}
        };

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = mesh.cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  - dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).left_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j).left_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j + 1).left_flux(a, u));
            });

        stencil = {
            {0, -1}
        };

        auto subset_up = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil), mesh[mesh_id_t::cells][level])
                             .on(level);

        subset_up(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = mesh.cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  + dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).up_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j + 1).up_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j + 1).up_flux(a, u));
            });

        stencil = {
            {0, 1}
        };

        auto subset_down = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_down(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = mesh.cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  - dt / dx
                                        * (samurai::upwind_op<dim, interval_t>(level, i, j).down_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i, 2 * j).down_flux(a, u)
                                           - .5 * samurai::upwind_op<dim, interval_t>(level + 1, 2 * i + 1, 2 * j).down_flux(a, u));
            });
    }
}

template <std::size_t min_level, std::size_t max_level>
void ADVECTION_2D(benchmark::State& state)
{
    for (auto _ : state)
    {
        state.PauseTiming();

        double Tf = 0.1;

        constexpr std::size_t dim = 2;
        using Config              = samurai::MRConfig<dim>;

        // Simulation parameters
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
        std::array<double, dim> a{
            {1, 0}
        };
        double cfl = 0.5;
        double t   = 0.;
        std::string restart_file;

        // Multiresolution parameters
        double mr_epsilon    = 2.e-4; // Threshold used by multiresolution
        double mr_regularity = 1.;    // Regularity guess for multiresolution
        // bool correction      = false;

        const samurai::Box<double, dim> box(min_corner, max_corner);
        samurai::MRMesh<Config> mesh;
        auto u = samurai::make_scalar_field<double>("u", mesh);

        mesh = {box, min_level, max_level};
        init(u);

        samurai::make_bc<samurai::Neumann<1>>(u, 0.);

        double dt = cfl * mesh.cell_length(max_level);

        auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

        auto MRadaptation = samurai::make_MRAdapt(u);
        MRadaptation(mr_epsilon, mr_regularity);

#ifdef SAMURAI_WITH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        state.ResumeTiming();
        int ITER_STEPS = 10;
        for (int i = 0; i < ITER_STEPS; i++)
        {
            MRadaptation(mr_epsilon, mr_regularity);
            t += dt;
            if (t > Tf)
            {
                dt += Tf - t;
                t = Tf;
            }
            samurai::update_ghost_mr(u);
            unp1.resize();
            unp1 = u - dt * samurai::upwind(a, u);
            // if (correction)
            // {
            //     flux_correction(dt, a, u, unp1);
            // }
            std::swap(u.array(), unp1.array());
        }
#ifdef SAMURAI_WITH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
}

// BENCHMARK(ADVECTION_2D);

// MRA with min_level = 5
BENCHMARK_TEMPLATE(ADVECTION_2D, 5, 8)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 5, 10)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 5, 12)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 5, 14)->Unit(benchmark::kMillisecond)->Iterations(1);

// MRA with max_level - min-level = 2
BENCHMARK_TEMPLATE(ADVECTION_2D, 6, 8)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 8, 10)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 10, 12)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 12, 14)->Unit(benchmark::kMillisecond)->Iterations(1);

// Uniform
BENCHMARK_TEMPLATE(ADVECTION_2D, 6, 6)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 8, 8)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 10, 10)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 12, 12)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(ADVECTION_2D, 14, 14)->Unit(benchmark::kMillisecond)->Iterations(1);

/** SOURCE : https://gist.github.com/mdavezac/eb16de7e8fc08e522ff0d420516094f5
 **/

// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter
{
  public:

    NullReporter()
    {
    }

    virtual bool ReportContext(const Context&)
    {
        return true;
    }

    virtual void ReportRuns(const std::vector<Run>&)
    {
    }

    virtual void Finalize()
    {
    }
};

int main(int argc, char** argv)
{
#ifdef SAMURAI_WITH_MPI
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif

    ::benchmark::Initialize(&argc, argv);
    if (rank == 0)
    {
        ::benchmark::RunSpecifiedBenchmarks();
    }
    else
    {
        NullReporter null;
        ::benchmark::RunSpecifiedBenchmarks(&null);
    }

#ifdef SAMURAI_WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
