#pragma once
#include "flux_based_scheme.hpp"
#include "cell_based_scheme.hpp"

namespace samurai 
{
    namespace petsc
    {
        /**
         * Multiplicatiion by a scalar value of the flux-based scheme
        */
        template<class Scheme>
        class Scalar_x_FluxBasedScheme : public FluxBasedScheme<typename Scheme::cfg_t, typename Scheme::field_t>
        {
        public:
            using cfg_t = typename Scheme::cfg_t;
            using field_t = typename Scheme::field_t;
            using Mesh = typename field_t::mesh_t;
            using flux_computation_t = typename FluxBasedScheme<cfg_t, field_t>::flux_computation_t;
            static constexpr std::size_t dim = field_t::dim;
        private:
            const Scheme& m_scheme;
            const double m_scalar;

        public:
            Scalar_x_FluxBasedScheme(const Scheme& scheme, double scalar) : 
                FluxBasedScheme<cfg_t, field_t>(scheme.unknown(), scalar_x_coefficients(scheme)),
                m_scheme(scheme),
                m_scalar(scalar)
            {}

        private:
            auto scalar_x_coefficients(const Scheme& scheme)
            {
                const std::array<flux_computation_t, dim>& scheme_fluxes = scheme.scheme_coefficients();
                std::array<flux_computation_t, dim> scalar_x_fluxes;
                
                for (std::size_t d = 0; d < dim; ++d)
                {
                    auto& scheme_flux = scheme_fluxes[d];
                    auto& scalar_x_flux = scalar_x_fluxes[d];

                    scalar_x_flux.direction             = scheme_flux.direction;
                    scalar_x_flux.computational_stencil = scheme_flux.computational_stencil;
                    scalar_x_flux.get_flux_coeffs       = scheme_flux.get_flux_coeffs;
                    scalar_x_flux.get_cell1_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        auto coeffs = scheme_flux.get_cell1_coeffs(flux_coeffs, h_face, h_cell);
                        for (auto& coeff : coeffs)
                        {
                            coeff *= m_scalar;
                        }
                        return coeffs;
                    };
                    scalar_x_flux.get_cell2_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        auto coeffs = scheme_flux.get_cell2_coeffs(flux_coeffs, h_face, h_cell);
                        for (auto& coeff : coeffs)
                        {
                            coeff *= m_scalar;
                        }
                        return coeffs;
                    };
                }
                return scalar_x_fluxes;
            }
        };

        template<class Scheme>
        auto operator*(double scalar, const Scheme& scheme)
        {
            return Scalar_x_FluxBasedScheme<Scheme>(scheme, scalar);
        }



        /**
         * Addition of two flux-based schemes
        */
        template<class Scheme1, class Scheme2>
        class Sum_FluxBasedScheme : public FluxBasedScheme<typename Scheme1::cfg_t, typename Scheme1::field_t>
        {
        public:
            using cfg_t = typename Scheme1::cfg_t;
            using field_t = typename Scheme1::field_t;
            using Mesh = typename field_t::mesh_t;
            using flux_computation_t = typename FluxBasedScheme<cfg_t, field_t>::flux_computation_t;
            static constexpr std::size_t dim = field_t::dim;
        private:
            const Scheme1& m_scheme1;
            const Scheme2& m_scheme2;

        public:
            Sum_FluxBasedScheme(const Scheme1& scheme1, const Scheme2& scheme2) : 
                FluxBasedScheme<cfg_t, field_t>(scheme1.unknown(), sum_coefficients(scheme1, scheme2)),
                m_scheme1(scheme1),
                m_scheme2(scheme2)
            {
                static_assert(std::is_same<typename Scheme1::field_t, typename Scheme2::field_t>::value, "Invalid '+' operation: incompatible field types.");

                if (&scheme1.unknown() != &scheme2.unknown())
                {
                    std::cerr << "Invalid '+' operation: both schemes must be associated to the same unknown." << std::endl;
                    assert(&scheme1.unknown() == &scheme2.unknown());
                }
            }

        private:
            auto sum_coefficients(const Scheme1& scheme1, const Scheme2& scheme2)
            {
                static_assert(Scheme1::cfg_t::comput_stencil_size == Scheme2::cfg_t::comput_stencil_size);

                auto& scheme1_fluxes = scheme1.scheme_coefficients();
                auto& scheme2_fluxes = scheme2.scheme_coefficients();
                std::array<flux_computation_t, dim> sum_fluxes = scheme1_fluxes;
                
                for (std::size_t d = 0; d < dim; ++d)
                {
                    const auto& scheme1_flux = scheme1_fluxes[d];
                    const auto& scheme2_flux = scheme2_fluxes[d];
                    auto& sum_flux = sum_fluxes[d];
                    sum_flux.direction             = scheme2_flux.direction;
                    sum_flux.computational_stencil = scheme2_flux.computational_stencil;
                    sum_flux.get_flux_coeffs       = scheme2_flux.get_flux_coeffs;
                    sum_flux.get_cell1_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        auto coeffs1 = scheme1_flux.get_cell1_coeffs(flux_coeffs, h_face, h_cell);
                        auto coeffs2 = scheme2_flux.get_cell1_coeffs(flux_coeffs, h_face, h_cell);
                        decltype(coeffs1) coeffs;
                        for (std::size_t i=0; i<cfg_t::comput_stencil_size; ++i)
                        {
                            coeffs[i] = coeffs1[i] + coeffs2[i];
                        }
                        return coeffs;
                    };
                    sum_flux.get_cell2_coeffs = [&](auto& flux_coeffs, double h_face, double h_cell)
                    {
                        auto coeffs1 = scheme1_flux.get_cell2_coeffs(flux_coeffs, h_face, h_cell);
                        auto coeffs2 = scheme2_flux.get_cell2_coeffs(flux_coeffs, h_face, h_cell);
                        decltype(coeffs1) coeffs;
                        for (std::size_t i=0; i<cfg_t::comput_stencil_size; ++i)
                        {
                            coeffs[i] = coeffs1[i] + coeffs2[i];
                        }
                        return coeffs;
                    };
                }
                return sum_fluxes;
            }
        };

        template <typename Scheme1, typename Scheme2, std::enable_if_t<is_FluxBasedScheme<Scheme1>, bool> = true, std::enable_if_t<is_FluxBasedScheme<Scheme2>, bool> = true>
        auto operator + (const Scheme1& s1, const Scheme2& s2 )
        {
            return Sum_FluxBasedScheme<Scheme1, Scheme2>(s1, s2);
        }






        /**
         * Addition of a flux-based scheme and a cell-based scheme.
         * The cell-based scheme is assembled first, then the flux-based scheme.
         * The boundary conditions are taken from the flux-based scheme.
        */
        template<class FluxScheme, class CellScheme>
        class FluxBasedScheme_Sum_CellBasedScheme : public MatrixAssembly
        {
        private:
            FluxScheme& m_flux_scheme;
            CellScheme& m_cell_scheme;
        public:
            FluxBasedScheme_Sum_CellBasedScheme(FluxScheme& flux_scheme, CellScheme& cell_scheme) :
                MatrixAssembly(),
                m_flux_scheme(flux_scheme),
                m_cell_scheme(cell_scheme)
            {
                if (&flux_scheme.unknown() != &cell_scheme.unknown())
                {
                    std::cerr << "Invalid '+' operation: both schemes must be associated to the same unknown." << std::endl;
                    assert(&flux_scheme.unknown() == &cell_scheme.unknown());
                }
            }

            auto& unknown() const
            {
                return m_flux_scheme.unknown();
            }

            PetscInt matrix_rows() const override
            {
                if (m_flux_scheme.matrix_rows() != m_cell_scheme.matrix_rows())
                {
                    std::cerr << "Invalid '+' operation: both schemes must generate the same number of matrix rows." << std::endl;
                }
                return m_flux_scheme.matrix_rows();
            }

            PetscInt matrix_cols() const override
            {
                if (m_flux_scheme.matrix_cols() != m_cell_scheme.matrix_cols())
                {
                    std::cerr << "Invalid '+' operation: both schemes must generate the same number of matrix columns." << std::endl;
                }
                return m_flux_scheme.matrix_cols();
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                // !!!!!!!!!!!!!!!!!!!
                // We hope that flux_scheme allocates enough non-zeros for both...!
                m_flux_scheme.sparsity_pattern_scheme(nnz);
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                // !!!!!!!!!!!!!!!!!!!
                // We hope that flux_scheme allocates enough non-zeros for both...!
                m_flux_scheme.sparsity_pattern_boundary(nnz);
            }
            
            void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const override
            {
                m_flux_scheme.sparsity_pattern_projection(nnz);
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& nnz) const override
            {
                m_flux_scheme.sparsity_pattern_prediction(nnz);
            }

            bool matrix_is_spd() const override
            { 
                return m_flux_scheme.matrix_is_spd() && m_cell_scheme.matrix_is_spd();
            }

            void assemble_scheme(Mat& A) override
            {
                // First the cell-based scheme because it uses INSERT_VALUES
                m_cell_scheme.assemble_scheme(A);

                // Flush to use ADD_VALUES instead of INSERT_VALUES
                MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

                // Then the flux-based scheme
                m_flux_scheme.assemble_scheme(A);
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                // We hope that flux_scheme and cell_scheme implement the boundary conditions in the same fashion,
                // and arbitrarily choose flux_scheme.
                m_flux_scheme.assemble_boundary_conditions(A);
            }

            void assemble_projection(Mat& A) override
            {
                // We hope that flux_scheme and cell_scheme implement the projection operator in the same fashion,
                // and arbitrarily choose flux_scheme.
                m_flux_scheme.assemble_projection(A);
            }

            void assemble_prediction(Mat& A) override
            {
                // We hope that flux_scheme and cell_scheme implement the prediction operator in the same fashion,
                // and arbitrarily choose flux_scheme.
                m_flux_scheme.assemble_prediction(A);
            }

            void add_1_on_diag_for_useless_ghosts(Mat& A) override
            {
                m_flux_scheme.add_1_on_diag_for_useless_ghosts(A);
            }

            void enforce_bc(Vec& b) const
            {
                m_flux_scheme.enforce_bc(b);
            }

            void add_0_for_useless_ghosts(Vec& b)
            {
                m_flux_scheme.add_0_for_useless_ghosts(b);
            }

            void enforce_projection_prediction(Vec& b) const
            {
                m_flux_scheme.enforce_projection_prediction(b);
            }
        };

        // Operator +
        template <typename FluxScheme, typename CellScheme, std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true>
        auto operator + (FluxScheme& flux_scheme, CellScheme& cell_scheme)
        {
            return FluxBasedScheme_Sum_CellBasedScheme<FluxScheme, CellScheme>(flux_scheme, cell_scheme);
        }

        // Operator + with reference rvalue
        template <typename FluxScheme, typename CellScheme, std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true>
        auto operator + (FluxScheme&& flux_scheme, CellScheme& cell_scheme)
        {
            return flux_scheme + cell_scheme;
        }

        // Operator + in the reverse order
        template <typename CellScheme, typename FluxScheme, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true, std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true>
        auto operator + (CellScheme& cell_scheme, FluxScheme& flux_scheme)
        {
            return flux_scheme + cell_scheme;
        }

        // Operator + in the reverse order with reference rvalue
        template <typename CellScheme, typename FluxScheme, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true, std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true>
        auto operator + (CellScheme& cell_scheme, FluxScheme&& flux_scheme)
        {
            return flux_scheme + cell_scheme;
        }

        // Operator + with reference rvalue
        /*template <typename CellScheme, typename FluxScheme, std::enable_if_t<is_CellBasedScheme<CellScheme>, bool> = true, std::enable_if_t<is_FluxBasedScheme<FluxScheme>, bool> = true>
        auto operator + (CellScheme&& cell_scheme, FluxScheme&& flux_scheme)
        {
            return flux_scheme + cell_scheme;
        }*/


    } // end namespace petsc
} // end namespace samurai