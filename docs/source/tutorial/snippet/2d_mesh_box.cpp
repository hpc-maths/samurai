#include <samurai/print.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/io/hdf5.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    std::size_t start_level   = 3;

    samurai::Box<double, dim> box({-1, -1}, {1, 1});
    samurai::CellArray<dim> ca_box;

    ca_box[start_level] = {start_level, box};

    samurai::io::print("{}\n", fmt::streamed(ca_box));

    samurai::save("2d_mesh_box", ca_box);
    return 0;
}
