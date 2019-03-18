#pragma once

#include <xtensor/xfixed.hpp>

#include "cell_list.hpp"

namespace mure
{
    template<class MRConfig, class value_t=double>
    class Field;

    template<class MRConfig>
    class Projection
    {
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

    public:
        Projection(std::size_t level,
                  xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                  interval_t interval)
            : level{level}, i{interval}
        {
            if (dim > 1)
                j = index[1];
            if (dim > 2)
                k = index[2];
        }

        void apply(Field<MRConfig>& field) const
        {
            apply_impl(field, std::integral_constant<std::size_t, dim>{});
        }

    private:
        std::size_t level;
        interval_t i;
        coord_index_t j, k;

        void apply_impl(Field<MRConfig>& field, std::integral_constant<std::size_t, 1>) const
        {
            field(level, i) = .5*(field(level+1, 2*i) + field(level+1, 2*i + 1));
        }

        void apply_impl(Field<MRConfig>& field, std::integral_constant<std::size_t, 2>) const
        {
            field(level, i, j) = .25*(field(level+1, 2*i    , 2*j) + field(level+1, 2*i    , 2*j+1)
                                    + field(level+1, 2*i + 1, 2*j) + field(level+1, 2*i + 1, 2*j+1));
        }

        void apply_impl(Field<MRConfig>& field, std::integral_constant<std::size_t, 3>) const
        {
            field(level - 1, i, j, k) = .125*(field(level, 2*i    , 2*j    , 2*k    )
                                            + field(level, 2*i + 1, 2*j    , 2*k    )
                                            + field(level, 2*i    , 2*j + 1, 2*k    )
                                            + field(level, 2*i + 1, 2*j + 1, 2*k    )
                                            + field(level, 2*i    , 2*j + 1, 2*k + 1)
                                            + field(level, 2*i + 1, 2*j + 1, 2*k + 1));
        }
    };

    template<class MRConfig>
    class Maximum
    {
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

    public:
        Maximum(std::size_t level,
                  xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                  interval_t interval)
            : level{level}, i{interval}
        {
            if (dim > 1)
                j = index[1];
            if (dim > 2)
                k = index[2];
        }

        void apply(Field<MRConfig, bool>& field) const
        {
            apply_impl(field, std::integral_constant<std::size_t, dim>{});
        }

    private:
        std::size_t level;
        interval_t i;
        coord_index_t j, k;

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 1>) const
        {
            field(level, i) = field(level+1, 2*i) | field(level+1, 2*i + 1);
        }

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 2>) const
        {
            // xt::xtensor<bool, 1> mask = field(level+1, 2*i    , 2*j    ) |
            //                             field(level+1, 2*i + 1, 2*j    ) |
            //                             field(level+1, 2*i    , 2*j + 1) |
            //                             field(level+1, 2*i + 1, 2*j + 1);

            // xt::masked_view(field(level+1, 2*i    , 2*j    ), mask) = true;
            // xt::masked_view(field(level+1, 2*i + 1, 2*j    ), mask) = true;
            // xt::masked_view(field(level+1, 2*i    , 2*j + 1), mask) = true;
            // xt::masked_view(field(level+1, 2*i + 1, 2*j + 1), mask) = true;

            field(level, i, j) = field(level+1, 2*i    , 2*j    ) |
                                 field(level+1, 2*i + 1, 2*j    ) |
                                 field(level+1, 2*i    , 2*j + 1) |
                                 field(level+1, 2*i + 1, 2*j + 1);
        }

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 3>) const
        {
            field(level - 1, i, j, k) = field(level, 2*i    , 2*j    , 2*k    ) |
                                        field(level, 2*i + 1, 2*j    , 2*k    ) |
                                        field(level, 2*i    , 2*j + 1, 2*k    ) |
                                        field(level, 2*i + 1, 2*j + 1, 2*k    ) |
                                        field(level, 2*i    , 2*j    , 2*k + 1) |
                                        field(level, 2*i + 1, 2*j    , 2*k + 1) |
                                        field(level, 2*i    , 2*j + 1, 2*k + 1) |
                                        field(level, 2*i + 1, 2*j + 1, 2*k + 1);
        }
    };

    // template<class MRConfig>
    // class Coarsen
    // {
    //     using index_t = typename MRConfig::index_t;
    //     using coord_index_t = typename MRConfig::coord_index_t;
    //     using interval_t = typename MRConfig::interval_t;
    //     constexpr static auto dim = MRConfig::dim;

    // public:
    //     Coarsen(std::size_t level,
    //               xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
    //               interval_t interval)
    //         : level{level}, i{interval}
    //     {
    //         if (dim > 1)
    //             j = index[1];
    //         if (dim > 2)
    //             k = index[2];
    //     }

    //     void apply(CellList<MRConfig>& cell_list, Field<MRConfig, bool> const& keep) const
    //     {
    //         apply_impl(cell_list, keep, std::integral_constant<std::size_t, dim>{});
    //     }

    // private:
    //     std::size_t level;
    //     interval_t i;
    //     coord_index_t j, k;

    //     void apply_impl(CellList<MRConfig>& cell_list, Field<MRConfig, bool> const& field, std::integral_constant<std::size_t, 1>) const
    //     {
    //         xt::xtensor<bool, 1> mask = field(level, 2*i) | field(level, 2*i + 1);
            
    //         xt::masked_view(field(level-1, i), !mask) = true;
    //         // xt::masked_view(field(level, 2*i), mask) = true;
    //         // xt::masked_view(field(level, 2*i + 1), mask) = true;
    //     }

    //     void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 2>) const
    //     {
    //         xt::xtensor<bool, 1> mask = field(level, 2*i    , 2*j    ) |
    //                                     field(level, 2*i + 1, 2*j    ) |
    //                                     field(level, 2*i    , 2*j + 1) |
    //                                     field(level, 2*i + 1, 2*j + 1);

    //         xt::masked_view(field(level-1, i, j), !mask) = true;

    //         // xt::masked_view(field(level, 2*i    , 2*j    ), mask) = true;
    //         // xt::masked_view(field(level, 2*i + 1, 2*j    ), mask) = true;
    //         // xt::masked_view(field(level, 2*i    , 2*j + 1), mask) = true;
    //         // xt::masked_view(field(level, 2*i + 1, 2*j + 1), mask) = true;
    //     }

    //     void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 3>) const
    //     {
    //         field(level - 1, i, j, k) = xt::maximum(field(level, 2*i    , 2*j    , 2*k    ),
    //                                                 field(level, 2*i + 1, 2*j    , 2*k    ),
    //                                                 field(level, 2*i    , 2*j + 1, 2*k    ),
    //                                                 field(level, 2*i + 1, 2*j + 1, 2*k    ),
    //                                                 field(level, 2*i    , 2*j    , 2*k + 1),
    //                                                 field(level, 2*i + 1, 2*j    , 2*k + 1),
    //                                                 field(level, 2*i    , 2*j + 1, 2*k + 1),
    //                                                 field(level, 2*i + 1, 2*j + 1, 2*k + 1));
    //     }
    // };

    template<class MRConfig>
    class Test
    {
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

    public:
        Test(std::size_t level,
                  xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                  interval_t interval)
            : level{level}, i{interval}
        {
            if (dim > 1)
                j = index[1];
            if (dim > 2)
                k = index[2];
        }

        void apply(Field<MRConfig, bool>& field) const
        {
            apply_impl(field, std::integral_constant<std::size_t, dim>{});
        }

    private:
        std::size_t level;
        interval_t i;
        coord_index_t j, k;

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 1>) const
        {
            xt::xtensor<bool, 1> mask = field(level, 2*i) | field(level, 2*i + 1);
            
            xt::masked_view(field(level-1, i), !mask) = true;
            // xt::masked_view(field(level, 2*i), mask) = true;
            // xt::masked_view(field(level, 2*i + 1), mask) = true;
        }

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 2>) const
        {
            xt::xtensor<bool, 1> mask = field(level, 2*i    , 2*j    ) |
                                        field(level, 2*i + 1, 2*j    ) |
                                        field(level, 2*i    , 2*j + 1) |
                                        field(level, 2*i + 1, 2*j + 1);

            xt::masked_view(field(level-1, i, j), !mask) = true;

            // xt::masked_view(field(level-1, i, j), mask) = false;
            // xt::masked_view(field(level, 2*i    , 2*j    ), mask) = true;
            // xt::masked_view(field(level, 2*i + 1, 2*j    ), mask) = true;
            // xt::masked_view(field(level, 2*i    , 2*j + 1), mask) = true;
            // xt::masked_view(field(level, 2*i + 1, 2*j + 1), mask) = true;
        }

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 3>) const
        {
            field(level - 1, i, j, k) = xt::maximum(field(level, 2*i    , 2*j    , 2*k    ),
                                                    field(level, 2*i + 1, 2*j    , 2*k    ),
                                                    field(level, 2*i    , 2*j + 1, 2*k    ),
                                                    field(level, 2*i + 1, 2*j + 1, 2*k    ),
                                                    field(level, 2*i    , 2*j    , 2*k + 1),
                                                    field(level, 2*i + 1, 2*j    , 2*k + 1),
                                                    field(level, 2*i    , 2*j + 1, 2*k + 1),
                                                    field(level, 2*i + 1, 2*j + 1, 2*k + 1));
        }
    };

    template<class MRConfig>
    class Clean
    {
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

    public:
        Clean(std::size_t level,
                  xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                  interval_t interval)
            : level{level}, i{interval}
        {
            if (dim > 1)
                j = index[1];
            if (dim > 2)
                k = index[2];
        }

        void apply(Field<MRConfig, bool>& field) const
        {
            apply_impl(field, std::integral_constant<std::size_t, dim>{});
        }

    private:
        std::size_t level;
        interval_t i;
        coord_index_t j, k;

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 1>) const
        {
            field(level, i) = false;
        }

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 2>) const
        {
            field(level, i, j) = false;
        }

        void apply_impl(Field<MRConfig, bool>& field, std::integral_constant<std::size_t, 3>) const
        {
            field(level - 1, i, j, k) = xt::maximum(field(level, 2*i    , 2*j    , 2*k    ),
                                                    field(level, 2*i + 1, 2*j    , 2*k    ),
                                                    field(level, 2*i    , 2*j + 1, 2*k    ),
                                                    field(level, 2*i + 1, 2*j + 1, 2*k    ),
                                                    field(level, 2*i    , 2*j    , 2*k + 1),
                                                    field(level, 2*i + 1, 2*j    , 2*k + 1),
                                                    field(level, 2*i    , 2*j + 1, 2*k + 1),
                                                    field(level, 2*i + 1, 2*j + 1, 2*k + 1));
        }
    };

    template<class MRConfig>
    class Graded_op
    {
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

    public:
        Graded_op(std::size_t level,
                xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                xt::xtensor_fixed<interval_t, xt::xshape<dim>> interval)
            : level{level}, i{interval[0]}, interval{interval}
        {
            if (dim > 1)
                j = index[1];
            if (dim > 2)
                k = index[2];
        }

        void apply(Field<MRConfig, bool>& field) const
        {
            apply_impl(field, std::integral_constant<std::size_t, dim>{});
        }

    private:
        std::size_t level;
        interval_t i;
        coord_index_t j, k;
        xt::xtensor_fixed<interval_t, xt::xshape<dim>> interval;

        void apply_impl(Field<MRConfig, bool>& field,
                        std::integral_constant<std::size_t, 1>) const
        {
            field(level, i+1) |= field(level, i);
            field(level, i-1) |= field(level, i);
            if (!(i.start&1))
                field(level, i-2) |= field(level, i);
            if (!(i.end&1))
                field(level, i+2) |= field(level, i);
        }

        void apply_impl(Field<MRConfig, bool>& field,
                        std::integral_constant<std::size_t, 2>) const
        {
            int ii_start = -1, ii_end = 1;
            int jj_start = -1, jj_end = 1;
            // if (!(i.start&1)) ii_start--;
            // if (!(i.end&1)) ii_end++;
            // if (!(interval[1].start&1) and j == interval[1].start) jj_start--;
            // if (!(interval[1].end&1)  and j == interval[1].end-1) jj_end++;

            for(int jj=jj_start; jj<=jj_end; ++jj)
                for(int ii=ii_start; ii<=ii_end; ++ii)
                    field(level, i + ii, j + jj) |= field(level, i, j);
            // field(level, i+1, j) |= field(level, i, j);
            // field(level, i-1, j) |= field(level, i, j);
            // field(level, i-1, j+1) |= field(level, i, j);
            // field(level, i  , j+1) |= field(level, i, j);
            // field(level, i+1, j+1) |= field(level, i, j);
            // field(level, i-1, j-1) |= field(level, i, j);
            // field(level, i  , j-1) |= field(level, i, j);
            // field(level, i+1, j-1) |= field(level, i, j);
            // if (!(i.start&1))
            // {
            //     field(level, i-2, j+1) |= field(level, i, j);
            //     field(level, i-2, j) |= field(level, i, j);
            //     field(level, i-2, j-1) |= field(level, i, j);
            // }                                
            // if (!(i.end&1))
            // {
            //     field(level, i+2, j+1) |= field(level, i, j);
            //     field(level, i+2, j) |= field(level, i, j);
            //     field(level, i+2, j-1) |= field(level, i, j);
            // }
            // if (!(interval[1].start&1) and j == interval[1].start)
            // {
            //     field(level, i+1, j-2) |= field(level, i, j);
            //     field(level, i, j-2) |= field(level, i, j);
            //     field(level, i-1, j-2) |= field(level, i, j);
            //     if (!(i.start&1))
            //         field(level, i-2, j-2) |= field(level, i, j);
            //     if (!(i.end&1))
            //         field(level, i+2, j-2) |= field(level, i, j);
            // }
            // if (!(interval[1].end&1) and j == interval[1].end-1)
            // {
            //     field(level, i+1, j+2) |= field(level, i, j);
            //     field(level, i, j+2) |= field(level, i, j);
            //     field(level, i-1, j+2) |= field(level, i, j);
            //     if (!(i.start&1))
            //         field(level, i-2, j+2) |= field(level, i, j);
            //     if (!(i.end&1))
            //         field(level, i+2, j+2) |= field(level, i, j);
            // }
        }

        void apply_impl(Field<MRConfig, bool>& field,
                        std::integral_constant<std::size_t, 3>) const
        {
            if (xt::any(field(level, i, j, k) < 1))
                field(level, i, j, k) = true;
        }
    };

    template<class MRConfig>
    class Copy
    {
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;

    public:
        Copy(std::size_t level,
                xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                interval_t interval)
            : level{level}, i{interval}
        {
            if (dim > 1)
                j = index[1];
            if (dim > 2)
                k = index[2];
        }

        void apply(Field<MRConfig>& dest, Field<MRConfig> const& src) const
        {
            apply_impl(dest, src, std::integral_constant<std::size_t, dim>{});
        }

    private:
        std::size_t level;
        interval_t i;
        coord_index_t j, k;

        void apply_impl(Field<MRConfig>& dest, Field<MRConfig> const& src, std::integral_constant<std::size_t, 1>) const
        {
            dest(level, i) = src(level, i);
        }

        void apply_impl(Field<MRConfig>& dest, Field<MRConfig> const& src, std::integral_constant<std::size_t, 2>) const
        {
            dest(level, i, j) = src(level, i, j);
        }

        void apply_impl(Field<MRConfig>& dest, Field<MRConfig> const& src, std::integral_constant<std::size_t, 3>) const
        {
            dest(level, i, j, k) = src(level, i, j, k);
        }
    };

    template<class MRConfig>
    class Detail_op
    {
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using interval_t = typename MRConfig::interval_t;
        constexpr static auto dim = MRConfig::dim;
        constexpr static auto max_refinement_level = MRConfig::max_refinement_level;

    public:
        Detail_op(std::size_t level,
                  xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> index,
                  interval_t interval)
            : level{level}, i{interval}
        {
            if (dim > 1)
                j = index[1];
            if (dim > 2)
                k = index[2];
        }

        void compute_detail(Field<MRConfig>& detail, Field<MRConfig> const& field) const
        {
            compute_detail_impl(detail, field, std::integral_constant<std::size_t, dim>{});
        }

        void compute_max_detail(xt::xtensor_fixed<double, xt::xshape<max_refinement_level+1>> &max_detail,
                                Field<MRConfig> const& detail) const
        {
            compute_max_detail_impl(max_detail, detail, std::integral_constant<std::size_t, dim>{});
        }

        void to_coarsen(Field<MRConfig, bool>& keep,
                        Field<MRConfig> const& detail,
                        xt::xtensor_fixed<double, xt::xshape<max_refinement_level+1>> const& max_detail,
                        double eps) const
        {
            to_coarsen_impl(keep, detail, max_detail, eps, std::integral_constant<std::size_t, dim>{});
        }

    private:
        std::size_t level;
        interval_t i;
        coord_index_t j, k;

        void compute_detail_impl(Field<MRConfig>& detail,
                                 Field<MRConfig> const& field,
                                 std::integral_constant<std::size_t, 1>) const
        {
            detail(level + 1, 2*i) = field(level + 1, 2*i) - (field(level, i) - 1./8*(field(level, i + 1) - field(level, i - 1)));
            detail(level + 1, 2*i + 1) = field(level + 1, 2*i + 1) - (field(level, i) + 1./8*(field(level, i + 1) - field(level, i - 1)));
        }

        void compute_detail_impl(Field<MRConfig>& detail,
                                 Field<MRConfig> const& field,
                                 std::integral_constant<std::size_t, 2>) const
        {
            detail(level + 1, 2*i, 2*j) = field(level + 1, 2*i, 2*j)
                                        - (field(level, i, j) - 1./8*(field(level, i + 1, j) - field(level, i - 1, j))
                                                              - 1./8*(field(level, i, j + 1) - field(level, i, j - 1))
                                                              - 1./64*(field(level, i + 1, j + 1) - field(level, i - 1, j + 1)
                                                                      +field(level, i - 1, j - 1) - field(level, i + 1, j - 1))
                                                              );

            detail(level + 1, 2*i, 2*j + 1) = field(level + 1, 2*i, 2*j + 1)
                                        - (field(level, i, j) - 1./8*(field(level, i + 1, j) - field(level, i - 1, j))
                                                              + 1./8*(field(level, i, j + 1) - field(level, i, j - 1))
                                                              + 1./64*(field(level, i + 1, j + 1) - field(level, i - 1, j + 1)
                                                                      +field(level, i - 1, j - 1) - field(level, i + 1, j - 1)) 
                                                              );

            detail(level + 1, 2*i + 1, 2*j) = field(level + 1, 2*i + 1, 2*j)
                                        - (field(level, i, j) + 1./8*(field(level, i + 1, j) - field(level, i - 1, j))
                                                              - 1./8*(field(level, i, j + 1) - field(level, i, j - 1))
                                                              + 1./64*(field(level, i + 1, j + 1) - field(level, i - 1, j + 1)
                                                                      +field(level, i - 1, j - 1) - field(level, i + 1, j - 1)) 
                                                              );

            detail(level + 1, 2*i + 1, 2*j + 1) = field(level + 1, 2*i + 1, 2*j + 1)
                                        - (field(level, i, j) + 1./8*(field(level, i + 1, j) - field(level, i - 1, j))
                                                              + 1./8*(field(level, i, j + 1) - field(level, i, j - 1))
                                                              - 1./64*(field(level, i + 1, j + 1) - field(level, i - 1, j + 1)
                                                                      +field(level, i - 1, j - 1) - field(level, i + 1, j - 1)) 
                                                              );
        }

        void compute_max_detail_impl(xt::xtensor_fixed<double, xt::xshape<max_refinement_level+1>> &max_detail,
                                     Field<MRConfig> const& detail,
                                     std::integral_constant<std::size_t, 1>) const
        {
            auto ii = 2*i;
            ii.step = 1;
            max_detail[level+1] = std::max(max_detail[level+1], xt::amax(xt::abs(detail(level + 1, ii)))[0]);
        }

        void compute_max_detail_impl(xt::xtensor_fixed<double, xt::xshape<max_refinement_level+1>> &max_detail,
                                     Field<MRConfig> const& detail,
                                     std::integral_constant<std::size_t, 2>) const
        {
            auto ii = 2*i;
            ii.step = 1;
            max_detail[level+1] = std::max(max_detail[level+1],
                                           xt::amax(xt::maximum(xt::abs(detail(level + 1, ii, 2*j)),
                                                                xt::abs(detail(level + 1, ii, 2*j+1))))[0]);
        }

        void to_coarsen_impl(Field<MRConfig, bool>& keep,
                             Field<MRConfig> const& detail,
                             xt::xtensor_fixed<double, xt::xshape<max_refinement_level+1>> const& max_detail,
                             double eps,
                             std::integral_constant<std::size_t, 1>) const
        {
            auto mask = (.5*(xt::abs(detail(level + 1, 2*i)) + xt::abs(detail(level + 1, 2*i+1)))/max_detail[level + 1]) < eps;
            // auto mask = xt::abs(detail(level + 1, ii)) < eps;
            xt::masked_view(keep(level + 1, 2*i), mask) = false;
            xt::masked_view(keep(level + 1, 2*i + 1), mask) = false;
        }

        void to_coarsen_impl(Field<MRConfig, bool>& keep,
                             Field<MRConfig> const& detail,
                             xt::xtensor_fixed<double, xt::xshape<max_refinement_level+1>> const& max_detail,
                             double eps,
                             std::integral_constant<std::size_t, 2>) const
        {
            auto mask = (0.25*(xt::abs(detail(level + 1, 2*i    , 2*j    )) + 
                               xt::abs(detail(level + 1, 2*i + 1, 2*j    )) + 
                               xt::abs(detail(level + 1, 2*i    , 2*j + 1)) + 
                               xt::abs(detail(level + 1, 2*i + 1, 2*j + 1)))/max_detail[level + 1]) < eps;
            xt::masked_view(keep(level + 1, 2*i    , 2*j    ), mask) = false;
            xt::masked_view(keep(level + 1, 2*i + 1, 2*j    ), mask) = false;
            xt::masked_view(keep(level + 1, 2*i    , 2*j + 1), mask) = false;
            xt::masked_view(keep(level + 1, 2*i + 1, 2*j + 1), mask) = false;
        }
    };

}