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
#include <samurai/schemes/fv.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/node.hpp>

#include <benchmark/benchmark.h>

template <std::size_t min_level, std::size_t max_level>
void LINEAR_CONVECTION(benchmark::State& state)
{
    for (auto _ : state)
    {
        state.PauseTiming();

        static constexpr std::size_t dim = 2;
        using Config                     = samurai::MRConfig<dim, 3>;
        using Box                        = samurai::Box<double, dim>;
        using point_t                    = typename Box::point_t;

        //--------------------//
        // Program parameters //
        //--------------------//

        // Simulation parameters
        double left_box  = -1;
        double right_box = 1;

        // Time integration
        double Tf  = 3;
        double dt  = 0;
        double cfl = 0.95;
        double t   = 0.;
        std::string restart_file;

        // Multiresolution parameters
        double mr_epsilon    = 1e-4; // Threshold used by multiresolution
        double mr_regularity = 1.;   // Regularity guess for multiresolution

        //--------------------//
        // Problem definition //
        //--------------------//

        point_t box_corner1, box_corner2;
        box_corner1.fill(left_box);
        box_corner2.fill(right_box);
        Box box(box_corner1, box_corner2);
        std::array<bool, dim> periodic;
        periodic.fill(true);
        samurai::MRMesh<Config> mesh;
        auto u = samurai::make_scalar_field<double>("u", mesh);

        mesh = {box, min_level, max_level, periodic};
        // Initial solution
        u = samurai::make_scalar_field<double>("u",
                                               mesh,
                                               [](const auto& coords)
                                               {
                                                   if constexpr (dim == 1)
                                                   {
                                                       const auto& x = coords(0);
                                                       return (x >= -0.8 && x <= -0.3) ? 1. : 0.;
                                                   }
                                                   else
                                                   {
                                                       const auto& x = coords(0);
                                                       const auto& y = coords(1);
                                                       return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                                   }
                                               });

        auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
        // Intermediary fields for the RK3 scheme
        auto u1 = samurai::make_scalar_field<double>("u1", mesh);
        auto u2 = samurai::make_scalar_field<double>("u2", mesh);

        unp1.fill(0);
        u1.fill(0);
        u2.fill(0);

        // Convection operator
        samurai::VelocityVector<dim> velocity;
        velocity.fill(1);
        if constexpr (dim == 2)
        {
            velocity(1) = -1;
        }
        auto conv = samurai::make_convection_weno5<decltype(u)>(velocity);

        //--------------------//
        //   Time iteration   //
        //--------------------//

        if (dt == 0)
        {
            double dx             = mesh.cell_length(max_level);
            auto a                = xt::abs(velocity);
            double sum_velocities = xt::sum(xt::abs(velocity))();
            dt                    = cfl * dx / sum_velocities;
        }

        auto MRadaptation = samurai::make_MRAdapt(u);
        MRadaptation(mr_epsilon, mr_regularity);

#ifdef SAMURAI_WITH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        state.ResumeTiming();
        int ITER_STEPS = 10;
        for (int i = 0; i < ITER_STEPS; i++)
        {
            // Move to next timestep
            t += dt;
            if (t > Tf)
            {
                dt += Tf - t;
                t = Tf;
            }

            // Mesh adaptation
            MRadaptation(mr_epsilon, mr_regularity);
            samurai::update_ghost_mr(u);
            unp1.resize();
            u1.resize();
            u2.resize();
            u1.fill(0);
            u2.fill(0);

            // unp1 = u - dt * conv(u);

            // TVD-RK3 (SSPRK3)
            u1 = u - dt * conv(u);
            samurai::update_ghost_mr(u1);
            u2 = 3. / 4 * u + 1. / 4 * (u1 - dt * conv(u1));
            samurai::update_ghost_mr(u2);
            unp1 = 1. / 3 * u + 2. / 3 * (u2 - dt * conv(u2));

            // u <-- unp1
            std::swap(u.array(), unp1.array());
        }
#ifdef SAMURAI_WITH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
}

// BENCHMARK(ADVECTION_2D);

// MRA with min_level = 5
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 5, 8)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 5, 10)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 5, 12)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 5, 14)->Unit(benchmark::kMillisecond)->Iterations(1);

// MRA with max_level - min-level = 2
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 6, 8)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 8, 10)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 10, 12)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 12, 14)->Unit(benchmark::kMillisecond)->Iterations(1);

// Uniform
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 6, 6)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 8, 8)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 10, 10)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 12, 12)->Unit(benchmark::kMillisecond)->Iterations(1);
BENCHMARK_TEMPLATE(LINEAR_CONVECTION, 14, 14)->Unit(benchmark::kMillisecond)->Iterations(1);

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
