#pragma once
// #include "../../../petsc/fv/flux_based_scheme_assembly.hpp"
#include "../explicit_FV_scheme.hpp"
#include "flux_based_scheme__lin_hom.hpp"

namespace samurai
{
    /**
     * LINEAR HOMOGENEOUS explicit schemes
     */
    template <class cfg, class bdry_cfg>
        requires(cfg::scheme_type == SchemeType::LinearHomogeneous)
    class Explicit<FluxBasedScheme<cfg, bdry_cfg>> : public ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>
    {
        using base_class = ExplicitFVScheme<FluxBasedScheme<cfg, bdry_cfg>>;

        using scheme_t       = typename base_class::scheme_t;
        using input_field_t  = typename base_class::input_field_t;
        using output_field_t = typename base_class::output_field_t;
        using value_t        = typename input_field_t::value_type;
        using size_type      = typename base_class::size_type;
        using base_class::scheme;

        static constexpr size_type n_comp         = input_field_t::n_comp;
        static constexpr size_type output_n_comp  = scheme_t::output_n_comp;
        static constexpr std::size_t stencil_size = cfg::stencil_size;

      public:

        using base_class::apply;

        explicit Explicit(scheme_t& s)
            : base_class(s)
        {
        }

      private:

        /**
         * Number of coarse cells adjacent to the fine interval @p i at a level jump.
         */
        template <class Interval>
        static std::size_t n_coarse_cells(const Interval& i)
        {
            return static_cast<std::size_t>(((i.end - 1) >> 1) - (i.start >> 1) + 1);
        }

        /**
         * At a level jump, gives each coarse cell the contributions of its fine neighbours in the
         * fine interval @p i: coarse_add(k, c) is called with c = contribution(ii) (or the sum of two
         * of them) for the fine cells ii whose parent is the k-th coarse cell of the interval.
         *
         * The fine interval is not always aligned with the coarse cells: with MPI, and especially
         * after load balancing, a subdomain may own only one of the two fine children of a coarse
         * cell. The interval can then start on a right child (odd start) and have an odd size.
         */
        template <class Interval, class CoarseAdd, class Contribution>
        SAMURAI_INLINE static void add_fine_to_coarse(const Interval& i, CoarseAdd&& coarse_add, Contribution&& contribution)
        {
            using offset_t = typename Interval::value_t;

            const offset_t first   = i.start & 1; // 1 if the interval starts on a right child
            const offset_t n_fine  = static_cast<offset_t>(i.size());
            const offset_t n_pairs = (n_fine - first) / 2;

            if (first != 0)
            {
                coarse_add(0, contribution(0));
            }
            // clang-format off
            #pragma omp simd
            for (offset_t p = 0; p < n_pairs; ++p)
            {
                coarse_add(first + p, contribution(first + 2*p) + contribution(first + 2*p + 1));
            }
            // clang-format on
            if ((n_fine - first) % 2 != 0)
            {
                coarse_add(first + n_pairs, contribution(n_fine - 1));
            }
        }

        template <class InterfaceType, class StencilType, class Coeffs>
        void _apply_contribution_in_sequential_context(output_field_t& output_field,
                                                       input_field_t& input_field,
                                                       InterfaceType& interface,
                                                       StencilType& stencil,
                                                       Coeffs& left_cell_coeffs,
                                                       Coeffs& right_cell_coeffs) const
        {
            const auto& i    = interface.interval();
            auto& left_cell  = interface.cells()[0];
            auto& right_cell = interface.cells()[1];

            auto left_cell_index_init  = left_cell.index;
            auto right_cell_index_init = right_cell.index;

            using index_t = decltype(left_cell_index_init);

            for (size_type field_i = 0; field_i < output_n_comp; ++field_i)
            {
                for (size_type field_j = 0; field_j < n_comp; ++field_j)
                {
                    for (std::size_t c = 0; c < stencil_size; ++c)
                    {
                        index_t comput_index_init = stencil.cells()[c].index;

                        auto left_cell_coeff  = this->scheme().cell_coeff(left_cell_coeffs, c, field_i, field_j);
                        auto right_cell_coeff = this->scheme().cell_coeff(right_cell_coeffs, c, field_i, field_j);

                        // clang-format off
                        if (left_cell.level == right_cell.level || i.size() == 1) // same level, or a single fine cell (always the case for a jump in the x-direction)
                        {
                            #pragma omp simd
                            for (index_t ii = 0; ii < static_cast<index_t>(i.size()); ++ii)
                            {
                                field_value(output_field, left_cell_index_init + ii, field_i) += left_cell_coeff * field_value(input_field, comput_index_init + ii, field_j);
                            }
                            #pragma omp simd
                            for (index_t ii = 0; ii < static_cast<index_t>(i.size()); ++ii)
                            {
                                field_value(output_field, right_cell_index_init + ii, field_i) += right_cell_coeff * field_value(input_field, comput_index_init + ii, field_j);
                            }
                        }
                        else if (left_cell.level < right_cell.level)
                        {
                            // Level jump: the left cells are the coarse ones, the interval i runs over the fine cells.
                            add_fine_to_coarse(i,
                                               [&](auto k, auto contrib) { field_value(output_field, left_cell_index_init + k, field_i) += contrib; },
                                               [&](auto ii) { return left_cell_coeff * field_value(input_field, comput_index_init + ii, field_j); });
                            #pragma omp simd
                            for (index_t ii = 0; ii < static_cast<index_t>(i.size()); ++ii) // iteration on the fine cells
                            {
                                field_value(output_field, right_cell_index_init + ii, field_i) += right_cell_coeff * field_value(input_field, comput_index_init + ii, field_j);
                            }
                        }
                        else // if (left_cell.level > right_cell.level)
                        {
                            // Same as above, the other way around.
                            #pragma omp simd
                            for (index_t ii = 0; ii < static_cast<index_t>(i.size()); ++ii)
                            {
                                field_value(output_field, left_cell_index_init + ii, field_i) += left_cell_coeff * field_value(input_field, comput_index_init + ii, field_j);
                            }
                            add_fine_to_coarse(i,
                                               [&](auto k, auto contrib) { field_value(output_field, right_cell_index_init + k, field_i) += contrib; },
                                               [&](auto ii) { return right_cell_coeff * field_value(input_field, comput_index_init + ii, field_j); });
                        }
                        // clang-format on
                    }
                }
            }
        }

        template <class InterfaceType, class StencilType, class Coeffs>
        void _apply_contribution_in_parallel_context(output_field_t& output_field,
                                                     input_field_t& input_field,
                                                     InterfaceType& interface,
                                                     StencilType& stencil,
                                                     Coeffs& left_cell_coeffs,
                                                     Coeffs& right_cell_coeffs) const
        {
            const auto& i    = interface.interval();
            auto& left_cell  = interface.cells()[0];
            auto& right_cell = interface.cells()[1];

            auto left_cell_index_init  = left_cell.index;
            auto right_cell_index_init = right_cell.index;

            auto n_left_cells  = (left_cell.level >= right_cell.level) ? i.size() : n_coarse_cells(i);
            auto n_right_cells = (left_cell.level <= right_cell.level) ? i.size() : n_coarse_cells(i);

            using index_t = decltype(left_cell_index_init);

            std::vector<value_t> left_contributions(n_left_cells, 0);
            std::vector<value_t> right_contributions(n_right_cells, 0);

            for (std::size_t field_i = 0; field_i < output_n_comp; ++field_i)
            {
                // We first accumulate the contributions in a SIMD fashion into local vectors,
                // and then we add the results to the field in an atomic fashion.

                std::fill(left_contributions.begin(), left_contributions.end(), 0);
                std::fill(right_contributions.begin(), right_contributions.end(), 0);

                for (std::size_t field_j = 0; field_j < n_comp; ++field_j)
                {
                    for (std::size_t c = 0; c < stencil_size; ++c)
                    {
                        index_t comput_index_init = stencil.cells()[c].index;

                        auto left_cell_coeff  = this->scheme().cell_coeff(left_cell_coeffs, c, field_i, field_j);
                        auto right_cell_coeff = this->scheme().cell_coeff(right_cell_coeffs, c, field_i, field_j);

                        // clang-format off
                        if (left_cell.level == right_cell.level || i.size() == 1) // same level, or a single fine cell (always the case for a jump in the x-direction)
                        {
                            #pragma omp simd
                            for (index_t ii = 0; ii < static_cast<index_t>(i.size()); ++ii)
                            {
                                left_contributions[static_cast<std::size_t>(ii)] += left_cell_coeff * field_value(input_field, comput_index_init + ii, field_j);
                            }
                            #pragma omp simd
                            for (index_t ii = 0; ii < static_cast<index_t>(i.size()); ++ii)
                            {
                                right_contributions[static_cast<std::size_t>(ii)] += right_cell_coeff * field_value(input_field, comput_index_init + ii, field_j);
                            }
                        }
                        else if (left_cell.level < right_cell.level)
                        {
                            // Level jump: the left cells are the coarse ones, the interval i runs over the fine cells.
                            add_fine_to_coarse(i,
                                               [&](auto k, auto contrib) { left_contributions[static_cast<std::size_t>(k)] += contrib; },
                                               [&](auto ii) { return left_cell_coeff * field_value(input_field, comput_index_init + ii, field_j); });
                            #pragma omp simd
                            for (index_t ii = 0; ii < static_cast<index_t>(i.size()); ++ii) // iteration on the fine cells
                            {
                                right_contributions[static_cast<std::size_t>(ii)] += right_cell_coeff * field_value(input_field, comput_index_init + ii, field_j);
                            }
                        }
                        else // if (left_cell.level > right_cell.level)
                        {
                            // Same as above, the other way around.
                            #pragma omp simd
                            for (index_t ii = 0; ii < static_cast<index_t>(i.size()); ++ii)
                            {
                                left_contributions[static_cast<std::size_t>(ii)] += left_cell_coeff * field_value(input_field, comput_index_init + ii, field_j);
                            }
                            add_fine_to_coarse(i,
                                               [&](auto k, auto contrib) { right_contributions[static_cast<std::size_t>(k)] += contrib; },
                                               [&](auto ii) { return right_cell_coeff * field_value(input_field, comput_index_init + ii, field_j); });
                        }
                        // clang-format on
                    }
                }

                // Here, the non-SIMD loops of atomic instructions are more efficient than opening a critical section and
                // execute SIMD loops.

                // clang-format off
                for (index_t ii = 0; ii < static_cast<index_t>(left_contributions.size()); ++ii)
                {
                    #pragma omp atomic update
                    field_value(output_field, left_cell_index_init + ii, field_i) += left_contributions[static_cast<std::size_t>(ii)];
                }
                for (index_t ii = 0; ii < static_cast<index_t>(right_contributions.size()); ++ii)
                {
                    #pragma omp atomic update
                    field_value(output_field, right_cell_index_init + ii, field_i) += right_contributions[static_cast<std::size_t>(ii)];
                }
                // clang-format on
            }
        }

      public:

        void apply(std::size_t d, output_field_t& output_field, input_field_t& input_field) override
        {
            assert(input_field.ghosts_updated());

            scheme().apply_directional_bc(input_field, d);

            /**
             * Implementation by matrix-vector multiplication
             */
            // Mat A;
            // auto assembly = petsc::make_assembly(scheme());
            // assembly.create_matrix(A);
            // assembly.assemble_matrix(A);
            // Vec vec_f   = petsc::create_petsc_vector_from(f);
            // Vec vec_res = petsc::create_petsc_vector_from(output_field);
            // MatMult(A, vec_f, vec_res);

            // Interior interfaces
            scheme().template for_each_interior_interface_and_coeffs<Run::Parallel, Get::Intervals>(
                d,
                input_field,
                [&](auto& interface, auto& stencil, auto& left_cell_coeffs, auto& right_cell_coeffs)
                {
#ifdef SAMURAI_WITH_OPENMP
                    if (omp_get_max_threads() > 1)
                    {
                        _apply_contribution_in_parallel_context(output_field, input_field, interface, stencil, left_cell_coeffs, right_cell_coeffs);
                    }
                    else
                    {
                        _apply_contribution_in_sequential_context(output_field,
                                                                  input_field,
                                                                  interface,
                                                                  stencil,
                                                                  left_cell_coeffs,
                                                                  right_cell_coeffs);
                    }
#else
                    _apply_contribution_in_sequential_context(output_field, input_field, interface, stencil, left_cell_coeffs, right_cell_coeffs);
#endif
                });

            // Boundary interfaces
            if (scheme().include_boundary_fluxes())
            {
                scheme().template for_each_boundary_interface_and_coeffs<Run::Parallel, Get::Intervals>(
                    d,
                    input_field,
                    [&](auto& cell, auto& stencil, auto& coeffs)
                    {
                        for (size_type field_i = 0; field_i < output_n_comp; ++field_i)
                        {
                            for (size_type field_j = 0; field_j < n_comp; ++field_j)
                            {
                                for (std::size_t c = 0; c < stencil_size; ++c)
                                {
#ifdef SAMURAI_CHECK_NAN
                                    if (std::isnan(field_value(input_field, stencil.cells()[c], field_j)))
                                    {
                                        std::cerr
                                            << "NaN detected when computing the flux on the boundary interfaces: " << stencil.cells()[c]
                                            << std::endl;
                                        assert(false);
                                    }
#endif
                                    auto coeff = this->scheme().cell_coeff(coeffs, c, field_i, field_j);
                                    // field_value(output_field, cell, field_i) += coeff * field_value(input_field, stencil[c], field_j);

                                    auto cell_index_init   = cell.cells()[0].index;
                                    auto comput_index_init = stencil.cells()[c].index;

                                    using index_t = decltype(cell_index_init);

                                // clang-format off

                                #pragma omp simd
                                for (index_t ii = 0; ii < static_cast<index_t>(stencil.interval().size()); ++ii)
                                {
                                    field_value(output_field, cell_index_init + ii, field_i) += coeff * field_value(input_field, comput_index_init + ii, field_j);
                                }
                                    // clang-format on
                                }
                            }
                        }
                    });
            }
        }
    };

} // end namespace samurai
