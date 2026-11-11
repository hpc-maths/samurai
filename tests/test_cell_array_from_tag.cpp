#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    // The straightforward version of cells_changed_by_tag and update_cell_array_from_tag: cell by cell,
    // then set expressions. The optimised code must give the same cell arrays.
    namespace reference
    {
        template <std::size_t dim, class TInterval, size_t max_size, class Tag>
        TagChanges<CellArray<dim, TInterval, max_size>>
        cells_changed_by_tag(const CellArray<dim, TInterval, max_size>& old_ca, const Tag& tag)
        {
            using size_type        = unsigned int;
            using value_t          = typename TInterval::value_t;
            using unsigned_value_t = typename std::make_unsigned_t<value_t>;
            using ca_type          = CellArray<dim, TInterval, max_size>;
            using coord_type       = typename ca_type::lca_type::coord_type;

            const auto& mesh         = tag.mesh();
            const size_t start_level = old_ca.min_level();
            const size_t end_level   = old_ca.max_level() + 1;

            TagChanges<ca_type> changes;
            std::vector<TInterval> add_p_interval;
            std::vector<coord_type> add_p_inner_stencil;
            std::vector<size_t> add_p_idx;

            for (size_t level = start_level; level < end_level; ++level)
            {
                const auto begin = old_ca[level].cbegin();
                const auto end   = old_ca[level].cend();
                for (auto it = begin; it != end; ++it)
                {
                    const auto& x_interval = *it;
                    const auto& yz         = it.index();
                    const bool is_yz_even  = dim == 1 or xt::all(xt::equal(yz % 2, 0));

                    for (value_t x = x_interval.start; x < x_interval.end; ++x)
                    {
                        const size_type itag         = static_cast<size_type>(x_interval.index) + static_cast<unsigned_value_t>(x);
                        const bool refine            = tag[itag] & static_cast<std::uint8_t>(CellFlag::refine);
                        const bool coarsenAndNotKeep = tag[itag] & static_cast<std::uint8_t>(CellFlag::coarsen)
                                                   and not(tag[itag] & static_cast<std::uint8_t>(CellFlag::keep));
                        if (refine and level < mesh.max_level())
                        {
                            changes.remove_p[level].add_point_back(x, yz);
                            if constexpr (dim == 1)
                            {
                                changes.add_p[level + 1].add_interval_back({2 * x, 2 * x + 2}, {});
                            }
                            else
                            {
                                nestedLoop<dim - 1, 0, dim - 2>(0,
                                                                2,
                                                                [&](const auto& inner_stencil)
                                                                {
                                                                    add_p_interval.push_back({2 * x, 2 * x + 2});
                                                                    if constexpr (dim > 2)
                                                                    {
                                                                        add_p_inner_stencil.emplace_back(2 * yz + inner_stencil);
                                                                        add_p_idx.push_back(add_p_interval.size() - 1);
                                                                    }
                                                                });
                            }
                        }
                        else if (coarsenAndNotKeep and level > mesh.min_level())
                        {
                            if (x % 2 == 0 and is_yz_even)
                            {
                                changes.add_m[level - 1].add_point_back(x >> 1, yz >> 1);
                            }
                            changes.remove_m[level].add_point_back(x, yz);
                        }
                    }
                    if (dim != 1 and (it + 1 == end or (it + 1).index()[dim - 2] != yz[dim - 2]))
                    {
                        add_list_of_interval_back(add_p_interval, coord_type(2 * yz), add_p_inner_stencil, add_p_idx, changes.add_p[level + 1]);
                        add_p_interval.clear();
                        add_p_inner_stencil.clear();
                        add_p_idx.clear();
                    }
                }
            }
            return changes;
        }

        template <std::size_t dim, class TInterval, size_t max_size, class Tag>
        CellArray<dim, TInterval, max_size> update_cell_array_from_tag(const CellArray<dim, TInterval, max_size>& old_ca, const Tag& tag)
        {
            const auto& mesh                                          = tag.mesh();
            const auto [ca_add_m, ca_remove_m, ca_add_p, ca_remove_p] = reference::cells_changed_by_tag(old_ca, tag);

            CellArray<dim, TInterval, max_size> new_ca;
            for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
            {
                auto set = difference(union_(old_ca[level], ca_add_m[level], ca_add_p[level]),
                                      union_(ca_remove_m[level], ca_remove_p[level]));
                set(
                    [&](const auto& x_interval, const auto& yz)
                    {
                        new_ca[level].add_interval_back(x_interval, yz);
                    });
            }
            return new_ca;
        }
    } // namespace reference

    namespace
    {
        // Random tags with every bit combination, either cell by cell or in runs of one value, so that both the
        // cell-by-cell and the 8-at-a-time reads of the tags see changes at every position. Kind 3 only refines
        // or coarsens, to build the next mesh.
        template <class Tag>
        void random_tags(Tag& tag, std::mt19937& gen, int kind)
        {
            std::uniform_int_distribution<int> any_flags(0, 15);
            std::uniform_int_distribution<int> run_length(1, 40);
            std::uniform_int_distribution<int> percent(0, 99);
            using value_t = std::decay_t<decltype(tag[0])>;

            std::size_t i = 0;
            while (i < tag.array().size())
            {
                int flags       = any_flags(gen);
                std::size_t len = 1;
                if (kind == 1) // runs
                {
                    len = static_cast<std::size_t>(run_length(gen));
                }
                else if (kind == 2) // mostly keep, a few changes
                {
                    len   = static_cast<std::size_t>(run_length(gen));
                    flags = percent(gen) < 85 ? static_cast<int>(CellFlag::keep) : flags;
                }
                else if (kind == 3) // runs of refine, coarsen or keep
                {
                    len                       = static_cast<std::size_t>(run_length(gen));
                    const int p               = percent(gen);
                    constexpr CellFlag pick[] = {CellFlag::refine, CellFlag::coarsen, CellFlag::keep};
                    flags                     = static_cast<int>(pick[p < 35 ? 0 : (p < 55 ? 1 : 2)]);
                }
                for (std::size_t j = 0; j < len and i < tag.array().size(); ++j, ++i)
                {
                    tag[i] = static_cast<value_t>(flags);
                }
            }
        }
    }

    template <typename T>
    class cell_array_from_tag_test : public ::testing::Test
    {
    };

    using cell_array_from_tag_test_types = ::testing::
        Types<std::integral_constant<std::size_t, 1>, std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;

    TYPED_TEST_SUITE(cell_array_from_tag_test, cell_array_from_tag_test_types, );

    // Adapts a mesh with random tags several times. At each step the new cell array and the four sets of
    // cells_changed_by_tag are compared to the reference version, for one-byte and int tags.
    TYPED_TEST(cell_array_from_tag_test, same_as_reference)
    {
        static constexpr std::size_t dim  = TypeParam::value;
        constexpr std::size_t min_level   = 2;
        constexpr std::size_t max_level   = dim == 1 ? 10 : (dim == 2 ? 7 : 5);
        constexpr std::size_t start_level = 3;
        constexpr int steps               = 12;

        using box_t   = Box<double, dim>;
        auto mesh_cfg = mesh_config<dim>().min_level(min_level).max_level(max_level).disable_args_parse();
        // a box around the origin, for negative coordinates
        auto mesh     = mra::make_mesh(box_t{-xt::ones<double>({dim}), xt::ones<double>({dim})}, mesh_cfg);
        using mesh_t  = decltype(mesh);
        using mesh_id = typename mesh_t::mesh_id_t;

        // start from the domain at a coarse level, the random refinements below build the levels
        {
            typename mesh_t::ca_type coarse_ca;
            coarse_ca[start_level] = typename mesh_t::lca_type(self(mesh.domain()).on(start_level));
            mesh_t coarse{coarse_ca, mesh};
            mesh.swap(coarse);
        }

        std::mt19937 gen(42 + static_cast<unsigned>(dim));
        for (int step = 0; step < steps; ++step)
        {
            const auto& cells = mesh[mesh_id::cells];
            for (int kind = 0; kind < 3; ++kind)
            {
                auto tag = make_scalar_field<std::uint8_t>("tag", mesh);
                random_tags(tag, gen, kind);

                EXPECT_TRUE(update_cell_array_from_tag(cells, tag) == reference::update_cell_array_from_tag(cells, tag))
                    << "step " << step << ", kind " << kind;
                const auto changes = cells_changed_by_tag(cells, tag);
                const auto ref     = reference::cells_changed_by_tag(cells, tag);
                EXPECT_TRUE(changes.add_m == ref.add_m);
                EXPECT_TRUE(changes.remove_m == ref.remove_m);
                EXPECT_TRUE(changes.add_p == ref.add_p);
                EXPECT_TRUE(changes.remove_p == ref.remove_p);

                auto int_tag = make_scalar_field<int>("int_tag", mesh);
                random_tags(int_tag, gen, kind);
                EXPECT_TRUE(update_cell_array_from_tag(cells, int_tag) == reference::update_cell_array_from_tag(cells, int_tag))
                    << "int tags, step " << step << ", kind " << kind;
            }

            // all keep: the cells do not change
            auto keep = make_scalar_field<std::uint8_t>("keep", mesh, static_cast<std::uint8_t>(CellFlag::keep));
            EXPECT_TRUE(update_cell_array_from_tag(cells, keep) == reference::update_cell_array_from_tag(cells, keep));

            // next mesh: refine and coarsen at random, graded, so that the levels mix
            auto tag = make_scalar_field<std::uint8_t>("tag", mesh);
            random_tags(tag, gen, 3);
            auto new_ca = reference::update_cell_array_from_tag(cells, tag);
            make_graduation(new_ca,
                            mesh.domain_pyramid(),
                            mesh.mpi_neighbourhood(),
                            mesh.periodicity(),
                            mesh.graduation_width(),
                            mesh.max_stencil_radius());
            mesh_t new_mesh{new_ca, mesh};
            mesh.swap(new_mesh);
        }
    }
}
