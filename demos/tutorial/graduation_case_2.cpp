#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xrandom.hpp>

#include <mure/box.hpp>
#include <mure/cell_array.hpp>
#include <mure/field.hpp>
#include <mure/hdf5.hpp>

#include <experimental/random>

auto generate_mesh(std::size_t min_level, std::size_t max_level, std::size_t nsamples = 100)
{
    constexpr std::size_t dim = 2;

    mure::CellList<dim> cl;
    cl[0][{0}].add_point(0);

    for(std::size_t s = 0; s < nsamples; ++s)
    {
        auto level = std::experimental::randint(min_level, max_level);
        auto x = std::experimental::randint(0, (1<<level) - 1);
        auto y = std::experimental::randint(0, (1<<level) - 1);

        cl[level][{y}].add_point(x);
    }

    return mure::CellArray<dim>(cl, true);
}

int main()
{
    constexpr std::size_t dim = 2;
    std::size_t min_level = 1;
    std::size_t max_level = 7;
    auto ca = generate_mesh(min_level, max_level);

    std::experimental::reseed(42);

    mure::save("mesh_before", ca);
    std::size_t ite = 0;
    while(true)
    {
        std::cout << "Iteration for remove intersection: " << ite++ << "\n";
        auto tag = mure::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for(std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            for(std::size_t level_below = min_level; level_below < level; ++level_below)
            {
                auto set = mure::intersection(ca[level], ca[level_below]).on(level_below);
                set([&](const auto& i, const auto& index)
                {
                    tag(level_below, i, index[0]) = true;
                });
            }
        }

        mure::CellList<dim> cl;
        mure::for_each_cell(ca, [&](auto cell)
        {
            auto i = cell.indices[0];
            auto j = cell.indices[1];
            if (tag[cell])
            {
                cl[cell.level + 1][{2*j}].add_interval({2*i, 2*i+2});
                cl[cell.level + 1][{2*j + 1}].add_interval({2*i, 2*i+2});
            }
            else
            {
                cl[cell.level][{j}].add_point(i);
            }
        });
        mure::CellArray<dim> new_ca = {cl, true};

        if(new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }
    mure::save("mesh_after", ca);

    // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};
    ite = 0;
    while(true)
    {
        std::cout << "Iteration for graduation: " << ite++ << "\n";
        auto tag = mure::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for(std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for(std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                for(std::size_t i = 0; i < stencil.shape()[0]; ++i)
                {
                    auto s = xt::view(stencil, i);
                    auto set = mure::intersection(mure::translate(ca[level], s), ca[level_below]).on(level_below);
                    set([&](const auto& i, const auto& index)
                    {
                        tag(level_below, i, index[0]) = true;
                    });
                }
            }
        }

        mure::CellList<dim> cl;
        mure::for_each_cell(ca, [&](auto cell)
        {
            auto i = cell.indices[0];
            auto j = cell.indices[1];
            if (tag[cell])
            {
                cl[cell.level + 1][{2*j}].add_interval({2*i, 2*i+2});
                cl[cell.level + 1][{2*j + 1}].add_interval({2*i, 2*i+2});
            }
            else
            {
                cl[cell.level][{j}].add_point(i);
            }
        });
        mure::CellArray<dim> new_ca = {cl, true};

        if(new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }
    mure::save("mesh_graduated", ca);

    return 0;
}