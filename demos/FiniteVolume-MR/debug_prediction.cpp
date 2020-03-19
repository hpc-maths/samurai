#include <mure/mure.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "evolve_mesh.hpp"


template <class Config>
auto init(mure::Mesh<Config> &mesh)
{
    mure::BC<2> bc{ {{ {mure::BCType::neumann, 0},
                       {mure::BCType::neumann, 0},
                       {mure::BCType::neumann, 0},
                       {mure::BCType::neumann, 0}
                    }} };

    mure::Field<Config> u{"u", mesh, bc};
    u.array().fill(0);


    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        double radius = .1;
        //u[cell] = sin(2.0 * pow(center[0], 4.5)) + cos(2.87*center[1]);
        u[cell] = 2.0 * center[0] + 2.87*center[1];

    });

    return u;
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 2;
    using Config = mure::MRConfig<dim>;
    using interval_t = typename Config::interval_t;

    std::size_t min_level = 6, max_level = 6;
    mure::Box<double, dim> box({0, 0}, {1, 1});
    mure::Mesh<Config> mesh{box, min_level, max_level};

    auto u = init(mesh);

    spdlog::set_level(spdlog::level::warn);


    mure::mr_projection(u);
    {
        std::stringstream s;
        s << "debug_prediction_after_projection_datum";
        auto h5file = mure::Hdf5(s.str().data());
        h5file.add_mesh(mesh);
        mure::Field<Config> level_{"level", mesh};
        mesh.for_each_cell([&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
        //h5file.add_field(u);
        h5file.add_field(level_);
        h5file.add_field_by_level(mesh, u);

    }

    mure::mr_prediction_for_debug(u, max_level);

    {
        std::stringstream s;
        s << "debug_prediction_after_prediction_datum";
        auto h5file = mure::Hdf5(s.str().data());
        h5file.add_mesh(mesh);
        mure::Field<Config> level_{"level", mesh};
        mesh.for_each_cell([&](auto &cell) { level_[cell] = static_cast<double>(cell.level); });
        //h5file.add_field(u);
        h5file.add_field(level_);
        h5file.add_field_by_level(mesh, u);

    }

    u.update_bc();


    return 0;
}
