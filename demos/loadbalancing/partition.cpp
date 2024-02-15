#include <iostream>

// CLI
#include <CLI/CLI.hpp>

// samurai 
#include <samurai/samurai.hpp>
#include <samurai/timers.hpp>
#include <samurai/field.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/morton.hpp>
#include <samurai/hilbert.hpp>

int main( int argc, char * argv[] ){

    samurai::initialize(argc, argv);
    Timers myTimers;

    constexpr std::size_t dim = 2;
    std::size_t minLevel = 3, maxLevel = 8;
    double mr_regularity = 1.0, mr_epsilon = 1e-4;
    xt::xtensor_fixed<double, xt::xshape<dim>> minCorner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> maxCorner = {1., 1.};
    std::string filename = "load_balancing";

    CLI::App app{"Load balancing test"};
    app.add_option("--min-corner", minCorner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", maxCorner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", minLevel, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", maxLevel, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity,"The regularity criteria used by the multiresolution to "
                   "adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    
    CLI11_PARSE(app, argc, argv);

    std::size_t start_level   = minLevel;

    myTimers.start("InitMesh");
    samurai::Box<double, dim> box( minCorner, maxCorner );
    samurai::CellArray<dim> mesh;

    mesh[start_level] = {start_level, box};

    auto level = samurai::make_field<int, 1>( "level", mesh );
    auto rank  = samurai::make_field<int, 1>( "rank", mesh );

    samurai::for_each_cell( mesh, [&](const auto & cell ){
        level[cell] = static_cast<int>( cell.level );
        rank [cell] = 0;
    });
    myTimers.stop("InitMesh");

    samurai::save( filename, mesh, level, rank );

    myTimers.print();
    samurai::finalize();

    return EXIT_SUCCESS;
}