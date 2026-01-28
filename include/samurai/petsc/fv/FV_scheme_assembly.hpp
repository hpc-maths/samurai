#pragma once
#include "../../algorithm/update.hpp"
#include "../../boundary.hpp"
#include "../../numeric/prediction.hpp"
#include "../../schemes/fv/FV_scheme.hpp"
#include "../../schemes/fv/scheme_operators.hpp"
#include "../compute_cell_ownership.hpp"
#include "../global_numbering.hpp"
#include "../matrix_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <BlockAssemblyType assembly_type_, std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockAssembly;

        /**
         * Finite Volume scheme.
         * This is the base class of CellBasedSchemeAssembly and FluxBasedSchemeAssembly.
         * It contains the management of
         *     - the boundary conditions
         *     - the projection/prediction ghosts
         *     - the unused ghosts
         */
        template <class Scheme>
        class FVSchemeAssembly : public MatrixAssembly
        {
            template <class Scheme2>
            friend class FVSchemeAssembly;

          protected:

            using MatrixAssembly::m_block_col_shift;
            using MatrixAssembly::m_block_row_shift;

            using MatrixAssembly::m_ghosts_col_shift;
            using MatrixAssembly::m_ghosts_row_shift;

          public:

            using cfg_t                                            = typename Scheme::cfg_t;
            using bdry_cfg_t                                       = typename Scheme::bdry_cfg;
            using field_t                                          = typename Scheme::field_t;
            using input_field_t                                    = field_t;
            using output_field_t                                   = typename cfg_t::output_field_t;
            using mesh_t                                           = typename field_t::mesh_t;
            using mesh_id_t                                        = typename mesh_t::mesh_id_t;
            using interval_t                                       = typename mesh_t::interval_t;
            using mesh_interval_t                                  = typename mesh_t::mesh_interval_t;
            using field_value_type                                 = typename field_t::value_type; // double
            using coord_index_t                                    = typename interval_t::coord_index_t;
            using index_t                                          = typename interval_t::index_t;
            static constexpr std::size_t dim                       = field_t::dim;
            static constexpr std::size_t input_n_comp              = field_t::n_comp;
            static constexpr std::size_t output_n_comp             = output_field_t::n_comp;
            static constexpr std::size_t prediction_stencil_radius = mesh_t::config::prediction_stencil_radius;
            static constexpr std::size_t bdry_neighbourhood_width  = bdry_cfg_t::neighbourhood_width;
            static constexpr std::size_t bdry_stencil_size         = bdry_cfg_t::stencil_size;
            static constexpr std::size_t nb_bdry_ghosts            = bdry_cfg_t::nb_ghosts;
            using cell_t                                           = Cell<dim, interval_t>;

            using dirichlet_t = DirichletImpl<nb_bdry_ghosts, field_t>;
            using neumann_t   = NeumannImpl<nb_bdry_ghosts, field_t>;

            using directional_bdry_config_t = DirectionalBoundaryConfig<field_t, output_n_comp, bdry_stencil_size, nb_bdry_ghosts>;

#ifdef SAMURAI_WITH_MPI
            static constexpr bool ghost_elimination_enabled = false; // ghost elimination and MPI are not compatible yet
#else
            static constexpr bool ghost_elimination_enabled = true;
#endif

          protected:

            Scheme m_scheme;
            field_t* m_unknown = nullptr;

            Numbering* m_row_numbering = nullptr;
            Numbering* m_col_numbering = nullptr;
            bool m_owns_numbering      = false;

            std::vector<bool> m_is_row_empty;

            // Ghost recursion
            using cell_coeff_pair_t     = std::pair<index_t, double>;
            using CellLinearCombination = std::vector<cell_coeff_pair_t>;
            using recursion_t           = std::map<index_t, CellLinearCombination>;
            recursion_t m_ghost_recursion;

          public:

            explicit FVSchemeAssembly(const Scheme& scheme)
                : m_scheme(scheme)
            {
                this->set_name(scheme.name());
            }

            ~FVSchemeAssembly() override
            {
                if (m_owns_numbering)
                {
                    assert(m_row_numbering == m_col_numbering);
                    delete m_row_numbering;
                    m_row_numbering = nullptr;
                    m_col_numbering = nullptr;
                }
            }

            auto& scheme()
            {
                return m_scheme;
            }

            const auto& scheme() const
            {
                return m_scheme;
            }

            void set_scheme(const Scheme& s)
            {
                m_scheme = s;
            }

            const auto& row_numbering() const
            {
                assert(m_row_numbering != nullptr);
                return *m_row_numbering;
            }

            auto& row_numbering()
            {
                assert(m_row_numbering != nullptr);
                return *m_row_numbering;
            }

            const auto& col_numbering() const
            {
                assert(m_col_numbering != nullptr);
                return *m_col_numbering;
            }

            auto& col_numbering()
            {
                assert(m_col_numbering != nullptr);
                return *m_col_numbering;
            }

            void set_row_numbering(Numbering& numbering)
            {
                m_row_numbering = &numbering;
            }

            void set_col_numbering(Numbering& numbering)
            {
                m_col_numbering = &numbering;
            }

            /**
             * This function is called in case of stand-alone assembly (e.g., Poisson equation).
             */
            void setup() override
            {
                if (!m_unknown)
                {
                    std::cerr << "Undefined unknown for operator " << scheme().name() << ". Assembly initialization failed!" << std::endl;
                    assert(false);
                    exit(EXIT_FAILURE);
                }

                mesh().cell_ownership().is_computed = false;
                compute_cell_ownership(mesh());
                compute_numbering();
#ifdef SAMURAI_WITH_MPI
                if (!m_is_block_in_monolithic_matrix)
                {
                    m_ghosts_row_shift = owned_matrix_rows();
                    m_ghosts_col_shift = owned_matrix_cols();
                    compute_local_to_global_rows(m_row_numbering->local_to_global_mapping);
                    m_col_numbering = m_row_numbering;
                }
#endif
                m_is_row_empty.resize(static_cast<std::size_t>(owned_matrix_rows()));
                std::fill(m_is_row_empty.begin(), m_is_row_empty.end(), true);

                if constexpr (ghost_elimination_enabled)
                {
                    m_ghost_recursion = ghost_recursion();
                }
            }

            /**
             * This function is called in case of sum_assembly (e.g., heat equation).
             */
            template <class OtherScheme>
            void setup(const FVSchemeAssembly<OtherScheme>& other)
            {
                compute_cell_ownership(mesh());
                m_row_numbering    = other.m_row_numbering;
                m_col_numbering    = other.m_col_numbering;
                m_ghosts_row_shift = owned_matrix_rows();
                m_ghosts_col_shift = owned_matrix_cols();

                m_is_row_empty.resize(static_cast<std::size_t>(owned_matrix_rows()));
                std::fill(m_is_row_empty.begin(), m_is_row_empty.end(), true);

                if constexpr (ghost_elimination_enabled)
                {
                    // m_ghost_recursion = other.m_ghost_recursion;
                    m_ghost_recursion = ghost_recursion(); // not optimized...
                }
            }

            /**
             * This function is called in case of monolithic block_assembly (e.g., Stokes equation).
             */
            template <std::size_t rows_, std::size_t cols_, class... Operators>
            void setup(BlockAssembly<BlockAssemblyType::Monolithic, rows_, cols_, Operators...>& block_assembly)
            {
                compute_cell_ownership(mesh());
                m_row_numbering = &block_assembly.numbering();
                m_col_numbering = m_row_numbering;

                m_is_row_empty.resize(static_cast<std::size_t>(owned_matrix_rows()));
                std::fill(m_is_row_empty.begin(), m_is_row_empty.end(), true);

                if constexpr (ghost_elimination_enabled)
                {
                    // m_ghost_recursion = block_assembly.first_block().m_ghost_recursion;
                    m_ghost_recursion = ghost_recursion(); // not optimized...
                }
            }

            /**
             * This function is called in case of nested block_assembly (e.g., Stokes equation).
             */
            template <std::size_t rows_, std::size_t cols_, class... Operators>
            void setup(BlockAssembly<BlockAssemblyType::NestedMatrices, rows_, cols_, Operators...>& /*block_assembly*/)
            {
                compute_cell_ownership(mesh());

                m_ghosts_row_shift = owned_matrix_rows();
                m_ghosts_col_shift = owned_matrix_cols();

                m_is_row_empty.resize(static_cast<std::size_t>(owned_matrix_rows()));
                std::fill(m_is_row_empty.begin(), m_is_row_empty.end(), true);

                if constexpr (ghost_elimination_enabled)
                {
                    m_ghost_recursion = ghost_recursion(); // not optimized...
                }
            }

            const std::vector<PetscInt>& local_to_global_rows() const override
            {
                assert(!has_duplicates(m_row_numbering->local_to_global_mapping)
                       && "local to global mapping (rows) has duplicate global indices");
                return m_row_numbering->local_to_global_mapping;
            }

            const std::vector<PetscInt>& local_to_global_cols() const override
            {
                assert(!has_duplicates(m_col_numbering->local_to_global_mapping)
                       && "local to global mapping (cols) has duplicate global indices");
                return m_col_numbering->local_to_global_mapping;
            }

            /**
             * This function is called in case of stand-alone assembly (e.g., Poisson equation) or nested block_assembly (e.g., Stokes
             * equation).
             */
            void compute_numbering()
            {
                if (m_row_numbering == nullptr)
                {
                    assert(m_col_numbering == nullptr);
                    m_row_numbering  = new Numbering();
                    m_owns_numbering = true;
                }
#ifdef SAMURAI_WITH_MPI
                assert(input_n_comp == output_n_comp && "we compute the numbering only for square matrices or blocks");

                PetscInt rank_row_shift = compute_rank_shift(owned_matrix_rows());

                m_row_numbering->resize(local_matrix_rows());
                compute_global_numbering<output_n_comp>(mesh(),
                                                        *m_row_numbering,
                                                        rank_row_shift,
                                                        0 /*block_row_shift*/,
                                                        owned_matrix_rows() /*block_ghosts_shift*/);
#endif
                m_col_numbering = m_row_numbering;
            }

            /**
             * This function is called in case of monolithic block_assembly (e.g., Stokes equation).
             */
            void compute_block_numbering()
            {
                assert(input_n_comp == output_n_comp && "we compute the numbering only for square matrices or blocks");

                // if (mpi::communicator().rank() == 1)
                // {
                //     sleep(1); // to have ordered output
                // }

                // std::cout << "[" << mpi::communicator().rank() << "] Computing global numbering for block '" << name() << "'\n";
                compute_global_numbering<output_n_comp>(mesh(), *m_row_numbering, m_rank_row_shift, m_block_row_shift, m_ghosts_row_shift);
            }

            void compute_local_to_global_rows(std::vector<PetscInt>& local_to_global_mapping)
            {
#ifdef SAMURAI_WITH_MPI
                if (args::print_petsc_numbering)
                {
                    mpi::communicator().barrier();
                    sleep(static_cast<unsigned int>(mpi::communicator().rank()));
                    std::cout << "[" << mpi::communicator().rank() << "] Computing local to global rows for matrix '" << name() << "'\n";
                }
#endif
                for (std::size_t cell_index = 0; cell_index < cell_ownership().n_local_cells; ++cell_index)
                {
                    for (unsigned int c = 0; c < static_cast<std::size_t>(output_n_comp); ++c)
                    {
                        auto local_index  = local_row_index(static_cast<PetscInt>(cell_index), c);
                        auto global_index = global_row_index(static_cast<PetscInt>(cell_index), c);

                        assert(static_cast<std::size_t>(local_index) < local_to_global_mapping.size());
                        local_to_global_mapping[static_cast<std::size_t>(local_index)] = global_index;

                        // std::cout << "[" << mpi::communicator().rank() << "] (owned by "
                        //           << m_row_numbering->ownership[static_cast<std::size_t>(cell_index)] << ") L" << local_index << " G"
                        //           << global_index << "\n";
                    }
                }
            }

            void compute_local_to_global_rows()
            {
                compute_local_to_global_rows(m_row_numbering->local_to_global_mapping);
            }

          private:

            auto ghost_recursion()
            {
                static constexpr PetscInt number_of_children = (1 << dim);

                recursion_t recursion;

                for_each_projection_ghost_and_children_cells<std::size_t>(
                    mesh(),
                    [&](auto, std::size_t ghost, const std::array<std::size_t, static_cast<std::size_t>(number_of_children)>& children)
                    {
                        CellLinearCombination linear_comb;
                        for (auto child : children)
                        {
                            linear_comb.emplace_back(child, 1. / number_of_children);
                        }
                        recursion.insert({ghost, linear_comb});
                    });

                for_each_prediction_ghost(mesh(),
                                          [&](auto& ghost)
                                          {
                                              recursion.insert({ghost.index, prediction_linear_combination(ghost)});
                                          });

                bool changes = true;
                while (changes)
                {
                    changes = false;
                    for (auto& pair : recursion)
                    {
                        // auto ghost        = pair.first;
                        auto& linear_comb = pair.second;

                        CellLinearCombination new_linear_comb;

                        // for (auto& c : linear_comb)
                        for (auto& [cell, coeff] : linear_comb)
                        {
                            auto it = recursion.find(cell);
                            if (it == recursion.end()) // it's a cell
                            {
                                new_linear_comb.emplace_back(cell, coeff);
                            }
                            else // it's a ghost
                            {
                                auto& ghost_lin_comb = it->second;
                                for (auto& [cell2, coeff2] : ghost_lin_comb)
                                {
                                    new_linear_comb.emplace_back(cell2, coeff * coeff2);
                                }
                                changes = true;
                            }
                        }

                        linear_comb = new_linear_comb;
                    }
                }

                return recursion;
            }

          protected:

            SAMURAI_INLINE bool is_locally_owned([[maybe_unused]] std::size_t cell_index) const
            {
#ifdef SAMURAI_WITH_MPI
                return cell_ownership().owner_rank[cell_index] == mpi::communicator().rank();
#else
                return true;
#endif
            }

            SAMURAI_INLINE bool is_locally_owned(const cell_t& cell) const
            {
                return is_locally_owned(static_cast<std::size_t>(cell.index));
            }

          public:

            void set_unknown(field_t& unknown)
            {
                m_unknown = &unknown;
            }

            auto& unknown() const
            {
                assert(m_unknown && "undefined unknown");
                return *m_unknown;
            }

            auto unknown_ptr() const
            {
                return m_unknown;
            }

            bool undefined_unknown() const
            {
                return !m_unknown;
            }

            const auto& mesh() const
            {
                return unknown().mesh();
            }

            auto& mesh()
            {
                return unknown().mesh();
            }

            const auto& cell_ownership() const
            {
                return mesh().cell_ownership();
            }

            PetscInt owned_matrix_rows() const override
            {
                return static_cast<PetscInt>(cell_ownership().n_owned_cells * output_n_comp);
            }

            PetscInt owned_matrix_cols() const override
            {
                return static_cast<PetscInt>(cell_ownership().n_owned_cells * input_n_comp);
            }

            PetscInt local_matrix_rows() const override
            {
                return static_cast<PetscInt>(cell_ownership().n_local_cells * output_n_comp);
            }

            PetscInt local_matrix_cols() const override
            {
                return static_cast<PetscInt>(cell_ownership().n_local_cells * input_n_comp);
            }

            SAMURAI_INLINE PetscInt local_col_index(PetscInt cell_index, unsigned int field_j) const
            {
#ifdef SAMURAI_WITH_MPI
                auto shift = is_locally_owned(static_cast<std::size_t>(cell_index)) ? m_block_col_shift : m_ghosts_col_shift;
                auto index = m_col_numbering->unknown_index<input_n_comp>(cell_ownership(),
                                                                          shift,
                                                                          static_cast<std::size_t>(cell_index),
                                                                          static_cast<int>(field_j));
                return m_col_numbering->local_indices[index];
#else
                return m_col_numbering->unknown_index<input_n_comp, PetscInt>(cell_ownership(),
                                                                              m_block_col_shift,
                                                                              static_cast<std::size_t>(cell_index),
                                                                              static_cast<int>(field_j));
#endif
            }

            SAMURAI_INLINE PetscInt global_col_index(PetscInt cell_index, unsigned int field_j) const
            {
#ifdef SAMURAI_WITH_MPI
                auto shift = is_locally_owned(static_cast<std::size_t>(cell_index)) ? m_block_col_shift : m_ghosts_col_shift;
                auto index = m_col_numbering->unknown_index<input_n_comp>(cell_ownership(),
                                                                          shift,
                                                                          static_cast<std::size_t>(cell_index),
                                                                          static_cast<int>(field_j));
                return m_col_numbering->global_indices[index];
#else
                return local_col_index(cell_index, field_j);
#endif
            }

            SAMURAI_INLINE PetscInt local_row_index(PetscInt cell_index, unsigned int field_i) const
            {
#ifdef SAMURAI_WITH_MPI
                auto shift = is_locally_owned(static_cast<std::size_t>(cell_index)) ? m_block_row_shift : m_ghosts_row_shift;
                auto index = m_row_numbering->unknown_index<output_n_comp>(cell_ownership(),
                                                                           shift,
                                                                           static_cast<std::size_t>(cell_index),
                                                                           static_cast<int>(field_i));
                return m_row_numbering->local_indices[index];
#else
                return m_row_numbering->unknown_index<output_n_comp, PetscInt>(cell_ownership(),
                                                                               m_block_row_shift,
                                                                               static_cast<std::size_t>(cell_index),
                                                                               static_cast<int>(field_i));
#endif
            }

            SAMURAI_INLINE PetscInt global_row_index(PetscInt cell_index, unsigned int field_i) const
            {
#ifdef SAMURAI_WITH_MPI
                auto shift = is_locally_owned(static_cast<std::size_t>(cell_index)) ? m_block_row_shift : m_ghosts_row_shift;
                auto index = m_row_numbering->unknown_index<output_n_comp>(cell_ownership(),
                                                                           shift,
                                                                           static_cast<std::size_t>(cell_index),
                                                                           static_cast<int>(field_i));
                return m_row_numbering->global_indices[index];
#else
                return local_row_index(cell_index, field_i);
#endif
            }

            SAMURAI_INLINE PetscInt global_col_index(std::size_t cell_index, unsigned int field_j) const
            {
                return global_col_index(static_cast<PetscInt>(cell_index), field_j);
            }

            SAMURAI_INLINE PetscInt global_row_index(std::size_t cell_index, unsigned int field_i) const
            {
                return global_row_index(static_cast<PetscInt>(cell_index), field_i);
            }

            SAMURAI_INLINE PetscInt global_col_index(const cell_t& cell, unsigned int field_j) const
            {
                return global_col_index(static_cast<PetscInt>(cell.index), field_j);
            }

            SAMURAI_INLINE PetscInt global_row_index(const cell_t& cell, unsigned int field_i) const
            {
                return global_row_index(static_cast<PetscInt>(cell.index), field_i);
            }

            SAMURAI_INLINE PetscInt local_col_index(std::size_t cell_index, unsigned int field_j) const
            {
                return local_col_index(static_cast<PetscInt>(cell_index), field_j);
            }

            SAMURAI_INLINE PetscInt local_row_index(std::size_t cell_index, unsigned int field_i) const
            {
                return local_row_index(static_cast<PetscInt>(cell_index), field_i);
            }

            SAMURAI_INLINE PetscInt local_col_index(const cell_t& cell, unsigned int field_j) const
            {
                return local_col_index(static_cast<PetscInt>(cell.index), field_j);
            }

            SAMURAI_INLINE PetscInt local_row_index(const cell_t& cell, unsigned int field_i) const
            {
                return local_row_index(static_cast<PetscInt>(cell.index), field_i);
            }

            template <class Coeffs>
            SAMURAI_INLINE double
            rhs_coeff(const Coeffs& coeffs, [[maybe_unused]] unsigned int field_i, [[maybe_unused]] unsigned int field_j) const
            {
                if constexpr (field_t::is_scalar && output_n_comp == 1)
                {
                    return coeffs;
                }
                else
                {
                    return coeffs(field_i, field_j);
                }
            }

            //-------------------------------------------------------------//
            //                           Vectors                           //
            //-------------------------------------------------------------//

          public:

            Vec create_solution_vector(const input_field_t& field) const
            {
#ifdef SAMURAI_WITH_MPI
                Vec v = create_petsc_vector(owned_matrix_cols());
                // VecSetLocalToGlobalMapping(v, m_local_to_global_cols);
                copy_unknown(field, v);
#else
                Vec v = create_petsc_vector_from(field);
#endif
                PetscObjectSetName(reinterpret_cast<PetscObject>(v), field.name().data());
                return v;
            }

            void copy_unknown(const input_field_t& field, Vec& v) const
            {
                double* v_data;
                VecGetArray(v, &v_data);
                for (std::size_t cell_index = 0; cell_index < cell_ownership().n_local_cells; ++cell_index)
                {
                    if (is_locally_owned(cell_index))
                    {
                        for (unsigned int j = 0; j < input_n_comp; ++j)
                        {
                            auto vec_row    = local_col_index(cell_index, j);
                            v_data[vec_row] = field_value(field, cell_index, j);
                        }
                    }
                }
                VecRestoreArray(v, &v_data);
            }

            void copy_unknown(const Vec& v, input_field_t& field) const
            {
                const double* v_data;
                VecGetArrayRead(v, &v_data);
                for (std::size_t cell_index = 0; cell_index < cell_ownership().n_local_cells; ++cell_index)
                {
                    if (is_locally_owned(cell_index))
                    {
                        for (unsigned int j = 0; j < input_n_comp; ++j)
                        {
                            auto vec_row                      = local_col_index(cell_index, j);
                            field_value(field, cell_index, j) = v_data[vec_row];
                        }
                    }
                }
                field.ghosts_updated() = false;
                VecRestoreArrayRead(v, &v_data);
            }

            Vec create_rhs_vector(const output_field_t& field) const
            {
#ifdef SAMURAI_WITH_MPI
                Vec v = create_petsc_vector(owned_matrix_rows());
                // VecSetLocalToGlobalMapping(v, m_local_to_global_rows);
                copy_rhs(field, v);
#else
                Vec v = create_petsc_vector_from(field);
#endif
                PetscObjectSetName(reinterpret_cast<PetscObject>(v), field.name().data());
                return v;
            }

            void copy_rhs(const output_field_t& field, Vec& v) const
            {
                double* v_data;
                VecGetArray(v, &v_data);
                for (std::size_t cell_index = 0; cell_index < cell_ownership().n_local_cells; ++cell_index)
                {
                    if (is_locally_owned(cell_index))
                    {
                        for (unsigned int i = 0; i < output_n_comp; ++i)
                        {
                            auto vec_row    = local_row_index(cell_index, i);
                            v_data[vec_row] = field_value(field, cell_index, i);
                        }
                    }
                }
                VecRestoreArray(v, &v_data);
            }

            void copy_rhs(const Vec& v, output_field_t& field) const
            {
                const double* v_data;
                VecGetArrayRead(v, &v_data);
                for (std::size_t cell_index = 0; cell_index < cell_ownership().n_local_cells; ++cell_index)
                {
                    if (is_locally_owned(cell_index))
                    {
                        for (unsigned int i = 0; i < output_n_comp; ++i)
                        {
                            auto vec_row                      = local_row_index(cell_index, i);
                            field_value(field, cell_index, i) = v_data[vec_row];
                        }
                    }
                }
                VecRestoreArrayRead(v, &v_data);
            }

            template <class int_type>
            SAMURAI_INLINE void set_is_row_not_empty(int_type local_row_number)
            {
                assert(local_row_number - m_block_row_shift >= 0);
                m_is_row_empty[static_cast<std::size_t>(local_row_number - m_block_row_shift)] = false;
            }

          protected:

            template <class Func>
            void iterate_on_boundary(Func&& apply) const
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (mesh().is_periodic(d))
                    {
                        continue;
                    }

                    if (cfg_t::stencil_size > 1 && unknown().get_bc().empty())
                    {
                        std::cerr << "Failure to assemble the boundary conditions in the operator '" << this->name()
                                  << "': no boundary condition attached to the field '" << unknown().name() << "'." << std::endl;
                        assert(false);
                        continue;
                    }

                    // Iterate over the boundary conditions set by the user
                    for (auto& bc : unknown().get_bc())
                    {
                        auto bc_region                  = bc->get_region();
                        auto& directions                = bc_region.first;
                        auto& boundary_cells_directions = bc_region.second;
                        // Iterate over the directions in that region
                        for (std::size_t d_region = 0; d_region < directions.size(); ++d_region)
                        {
                            auto& towards_out = directions[d_region];

                            int number_of_one = xt::sum(xt::abs(towards_out))[0];
                            if (number_of_one == 1 && find_direction_index(towards_out) == d)
                            {
                                auto& boundary_cells   = boundary_cells_directions[d_region];
                                dirichlet_t* dirichlet = dynamic_cast<dirichlet_t*>(bc.get());
                                neumann_t* neumann     = dynamic_cast<neumann_t*>(bc.get());
                                if (dirichlet)
                                {
                                    auto config = scheme().dirichlet_config(towards_out);
                                    for_each_stencil_on_boundary(mesh(),
                                                                 boundary_cells,
                                                                 config.directional_stencil.stencil,
                                                                 config.equations,
                                                                 [&](auto& cells, auto& equations)
                                                                 {
                                                                     apply(cells, equations, towards_out, dirichlet);
                                                                 });
                                }
                                else if (neumann)
                                {
                                    auto config = scheme().neumann_config(towards_out);
                                    for_each_stencil_on_boundary(mesh(),
                                                                 boundary_cells,
                                                                 config.directional_stencil.stencil,
                                                                 config.equations,
                                                                 [&](auto& cells, auto& equations)
                                                                 {
                                                                     apply(cells, equations, towards_out, neumann);
                                                                 });
                                }
                                else
                                {
                                    std::cerr << "Unknown boundary condition type. Only Dirichlet and Neumann are implemented at the moment."
                                              << std::endl;
                                }
                            }
                        }
                    }
                }
            }

            template <class Func>
            void iterate_on_periodic_ghosts(Func&& apply) const
            {
                mesh_interval_t ghost_mi;
                mesh_interval_t cell_mi;

                std::size_t min_level = mesh()[mesh_id_t::reference].min_level();
                std::size_t max_level = mesh()[mesh_id_t::reference].max_level();

                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    ghost_mi.level = level;
                    cell_mi.level  = level;
                    iterate_over_periodic_ghosts(
                        level,
                        unknown(),
                        [&](const auto& i_ghosts, const auto& index_ghosts, const auto& i_cells, const auto& index_cells)
                        {
                            ghost_mi.i            = i_ghosts;
                            ghost_mi.index        = index_ghosts;
                            auto ghost_cell_index = get_index_start(mesh(), ghost_mi);
                            cell_mi.i             = i_cells;
                            cell_mi.index         = index_cells;
                            auto cell_cell_index  = get_index_start(mesh(), cell_mi);

                            for (std::size_t ii = 0; ii < i_ghosts.size(); ++ii)
                            {
                                apply(ghost_cell_index, cell_cell_index);

                                ghost_cell_index++;
                                cell_cell_index++;
                            }
                        });
                };
            }

            //-------------------------------------------------------------//
            //        Sparsity pattern of the boundary conditions          //
            //-------------------------------------------------------------//

          public:

            void sparsity_pattern_boundary(std::vector<PetscInt>& d_nnz, [[maybe_unused]] std::vector<PetscInt>& o_nnz) const override
            {
                if (mesh().is_periodic())
                {
                    iterate_on_periodic_ghosts(
                        [&](auto ghost_cell_index, [[maybe_unused]] auto cell_cell_index)
                        {
                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
                                std::size_t row = static_cast<std::size_t>(local_row_index(static_cast<PetscInt>(ghost_cell_index), field_i));
#ifdef SAMURAI_WITH_MPI
                                if (is_locally_owned(static_cast<std::size_t>(cell_cell_index)))
                                    d_nnz[row] = 2;
                                else
                                {
                                    d_nnz[row] = 1;
                                    o_nnz[row] = 1;
                                }
#else
                                d_nnz[row] = 2;
#endif
                            }
                        });
                }

                iterate_on_boundary(
                    [&](auto& cells, auto& equations, auto&, auto*)
                    {
                        sparsity_pattern_bc(d_nnz, o_nnz, cells, equations);
                    });
            }

          protected:

            template <class CellList, class CoeffList>
            void sparsity_pattern_bc(std::vector<PetscInt>& d_nnz,
                                     [[maybe_unused]] std::vector<PetscInt>& o_nnz,
                                     CellList& cells,
                                     std::array<CoeffList, nb_bdry_ghosts>& equations) const
            {
                for (std::size_t e = 0; e < nb_bdry_ghosts; ++e)
                {
                    const auto& eq    = equations[e];
                    const auto& ghost = cells[eq.ghost_index];

#ifdef SAMURAI_WITH_MPI
                    if (!is_locally_owned(ghost))
                    {
                        continue;
                    }

                    for (std::size_t c = 0; c < bdry_stencil_size; ++c)
                    {
                        if (is_locally_owned(cells[c]))
                        {
                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
                                d_nnz[static_cast<std::size_t>(local_row_index(ghost, field_i))] += 1;
                            }
                        }
                        else
                        {
                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
                                o_nnz[static_cast<std::size_t>(local_row_index(ghost, field_i))] += 1;
                            }
                        }
                    }
#else
                    for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                    {
                        d_nnz[static_cast<std::size_t>(local_row_index(ghost, field_i))] = bdry_stencil_size;
                    }
#endif
                }
            }

            //-------------------------------------------------------------//
            //             Assemble the boundary conditions                //
            //-------------------------------------------------------------//

          public:

            void assemble_boundary_conditions(Mat& A) override
            {
                // std::cout << "[" << mpi::communicator().rank() << "] assemble_boundary_conditions of " << this->name() << std::endl;
                if (current_insert_mode() == ADD_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    // std::cout << "[" << mpi::communicator().rank() << "] Flushing assembly to switch to INSERT_VALUES mode\n";
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(INSERT_VALUES);
                    // std::cout << "[" << mpi::communicator().rank() << "] end flush" << std::endl;
                }

                if (mesh().is_periodic())
                {
#ifdef SAMURAI_WITH_MPI
                    if (mpi::communicator().size() > 1)
                    {
                        std::cerr << "Periodic boundary conditions are not supported with MPI." << std::endl;
                        exit(EXIT_FAILURE);
                    }
#endif
                    assemble_periodic_bc(A);
                }

                iterate_on_boundary(
                    [&](auto& cells, auto& equations, auto&, auto*)
                    {
                        assemble_bc(A, cells, equations);
                    });
            }

            template <class CellList, class CoeffList>
            void assemble_bc(Mat& A, CellList& cells, std::array<CoeffList, nb_bdry_ghosts>& equations)
            {
                for (std::size_t e = 0; e < nb_bdry_ghosts; ++e)
                {
                    auto eq           = equations[e];
                    const auto& ghost = cells[eq.ghost_index];

#ifdef SAMURAI_WITH_MPI
                    if (!is_locally_owned(ghost))
                    {
                        continue;
                    }
#endif
                    for (std::size_t c = 0; c < bdry_stencil_size; ++c)
                    {
                        for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                        {
                            PetscInt equation_row = local_row_index(ghost, field_i);
                            PetscInt col          = local_col_index(cells[c], field_i);

                            double coeff = scheme().bdry_cell_coeff(eq.stencil_coeffs, c, field_i, field_i);

                            if (coeff != 0)
                            {
                                // std::cout << fmt::format("[{}] BC  A[L{},L{}] = A[G{},G{}] = {} (ghost {}, cell {})\n",
                                //                          mpi::communicator().rank(),
                                //                          equation_row,
                                //                          col,
                                //                          global_row_index(ghost, field_i),
                                //                          global_col_index(cells[c], field_i),
                                //                          coeff,
                                //                          ghost.index,
                                //                          cells[c].index);
                                MatSetValueLocal(A, equation_row, col, coeff, INSERT_VALUES);
                                set_is_row_not_empty(equation_row);
                            }
                        }
                    }
                }
            }

          private:

            SAMURAI_INLINE void assemble_periodic_bc(Mat& A)
            {
                std::vector<bool> is_periodic_row_empty(mesh().nb_cells(), true);

                iterate_on_periodic_ghosts(
                    [&](auto ghost_cell_index, auto cell_cell_index)
                    {
                        if (is_periodic_row_empty[static_cast<std::size_t>(ghost_cell_index)]) // to avoid multiple insertions when a
                                                                                               // ghost is periodic in several
                                                                                               // directions (external corners)
                        {
                            if (!is_locally_owned(static_cast<std::size_t>(ghost_cell_index))) // cppcheck-suppress knownConditionTrueFalse
                            {
                                return;
                            }
                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
                                PetscInt row = local_row_index(static_cast<PetscInt>(ghost_cell_index), field_i);
                                PetscInt col = local_col_index(static_cast<PetscInt>(cell_cell_index), field_i);

                                MatSetValueLocal(A, row, row, 1., INSERT_VALUES);
                                MatSetValueLocal(A, row, col, -1, INSERT_VALUES);
                                set_is_row_not_empty(row);
                            }
                            is_periodic_row_empty[static_cast<std::size_t>(ghost_cell_index)] = false;
                        }
                    });
            }

          public:

            //-------------------------------------------------------------//
            //   Enforce the boundary conditions on the right-hand side    //
            //-------------------------------------------------------------//

            virtual void enforce_bc(Vec& b) const
            {
                // std::cout << "enforce_bc of " << this->name() << std::endl;
                if (!this->is_block_in_monolithic_matrix())
                {
                    PetscInt b_rows;
                    VecGetLocalSize(b, &b_rows);
                    if (b_rows != this->owned_matrix_cols())
                    {
                        std::cerr << "Operator '" << this->name() << "': the number of rows in vector (" << b_rows
                                  << ") does not equal the number of columns of the matrix (" << this->owned_matrix_cols() << ")"
                                  << std::endl;
                        assert(false);
                        return;
                    }
                }

                double* b_data;
                VecGetArray(b, &b_data);

                iterate_on_boundary(
                    [&](auto& cells, auto& equations, auto& towards_out, auto* bc)
                    {
                        enforce_bc(b_data, cells, equations, bc, towards_out);
                    });

                VecRestoreArray(b, &b_data);
            }

          private:

            template <class CellList, class CoeffList, class BoundaryCondition>
            void enforce_bc(double* b_data,
                            CellList& cells,
                            std::array<CoeffList, nb_bdry_ghosts>& equations,
                            const BoundaryCondition* bc,
                            const DirectionVector<dim>& towards_out) const
            {
                auto& cell          = cells[0];
                auto boundary_point = cell.face_center(towards_out);

                for (std::size_t e = 0; e < nb_bdry_ghosts; ++e)
                {
                    auto eq           = equations[e];
                    const auto& ghost = cells[eq.ghost_index];
                    if (!is_locally_owned(ghost)) // cppcheck-suppress knownConditionTrueFalse
                    {
                        continue;
                    }
                    for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                    {
                        PetscInt equation_row = local_row_index(ghost, field_i);

                        double coeff = rhs_coeff(eq.rhs_coeffs, field_i, field_i);
                        assert(coeff != 0);

                        double bc_value;
                        if constexpr (field_t::is_scalar)
                        {
                            bc_value = bc->value(towards_out, cell, boundary_point);
                        }
                        else
                        {
                            bc_value = bc->value(towards_out, cell, boundary_point)(field_i); // TODO: call get_value() only once
                                                                                              // instead of once per field_i
                        }
                        b_data[equation_row] += coeff * bc_value;
                    }
                }
            }

          public:

            //-------------------------------------------------------------//
            //                      Useless ghosts                         //
            //-------------------------------------------------------------//

            template <class Func>
            void for_each_useless_ghost_row(Func&& f) const
            {
                for (std::size_t i = 0; i < m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        f(m_block_row_shift + static_cast<PetscInt>(i));
                    }
                }
            }

            void insert_value_on_diag_for_useless_ghosts(Mat& A) override
            {
                if (current_insert_mode() == ADD_VALUES)
                {
                    // Must flush to use INSERT_VALUES instead of ADD_VALUES
                    MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
                    set_current_insert_mode(INSERT_VALUES);
                }
                for (std::size_t i = 0; i < m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        auto error = MatSetValueLocal(A,
                                                      m_block_row_shift + static_cast<PetscInt>(i),
                                                      m_block_col_shift + static_cast<PetscInt>(i),
                                                      this->diag_value_for_useless_ghosts(),
                                                      INSERT_VALUES);
                        if (error)
                        {
                            std::cerr << scheme().name() << ": failure to insert diagonal coefficient at ("
                                      << m_block_row_shift + static_cast<PetscInt>(i) << ", " << m_block_col_shift + static_cast<PetscInt>(i)
                                      << "), i.e. (" << i << ", " << i << ") in the block." << std::endl;
                            assert(false);
                            exit(EXIT_FAILURE);
                        }
                        // m_is_row_empty[i] = false;
                    }
                }
            }

            void set_0_for_useless_ghosts(Vec& b) const
            {
                for (std::size_t i = 0; i < m_is_row_empty.size(); i++)
                {
                    if (m_is_row_empty[i])
                    {
                        VecSetValueLocal(b, m_block_row_shift + static_cast<PetscInt>(i), 0, INSERT_VALUES);
                    }
                }
            }

            void set_0_for_all_ghosts(Vec& b) const
            {
                double* b_data;
                VecGetArray(b, &b_data);
                for (std::size_t level = 0; level <= mesh().max_level(); ++level)
                {
                    auto ghosts = difference(mesh()[mesh_id_t::reference][level], mesh()[mesh_id_t::cells][level]);

                    for_each_cell(mesh(),
                                  ghosts,
                                  [&](auto& ghost)
                                  {
                                      if (is_locally_owned(ghost)) // cppcheck-suppress knownConditionTrueFalse
                                      {
                                          for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                                          {
                                              b_data[local_row_index(ghost, field_i)] = 0;
                                          }
                                      }
                                  });
                }
                VecRestoreArray(b, &b_data);
            }

            //-------------------------------------------------------------//
            //                  Projection / prediction                    //
            //-------------------------------------------------------------//

            void sparsity_pattern_projection(std::vector<PetscInt>& d_nnz, [[maybe_unused]] std::vector<PetscInt>& o_nnz) const override
            {
                // ----  Projection stencil size
                // cell + 2^dim children --> 1+2=3 in 1D
                //                           1+4=5 in 2D
                static constexpr std::size_t proj_stencil_size = 1 + (1 << dim);
#ifdef SAMURAI_WITH_MPI
                // assume that half the stencil can be on other processes
                PetscInt o_nnz_value = mpi::communicator().size() == 1 ? 0 : proj_stencil_size / 2;
#endif

                for_each_projection_ghost(mesh(),
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                                              {
                                                  std::size_t row = static_cast<std::size_t>(local_row_index(ghost, field_i));
#ifdef SAMURAI_WITH_MPI
                                                  if (is_locally_owned(ghost))
                                                  {
                                                      d_nnz[row] = proj_stencil_size;
                                                      o_nnz[row] = o_nnz_value;
                                                  }
#else
                                                if constexpr (ghost_elimination_enabled)
                                                {
                                                    d_nnz[row] = static_cast<PetscInt>(m_ghost_recursion.at(ghost.index).size());
                                                }
                                                else
                                                {
                                                    d_nnz[row] = proj_stencil_size;
                                             }
#endif
                                              }
                                          });
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& d_nnz, [[maybe_unused]] std::vector<PetscInt>& o_nnz) const override
            {
                // ----  Prediction stencil size
                // Order 1: cell + hypercube of 3 coarser cells --> 1 + 3= 4 in 1D
                //                                                  1 + 9=10 in 2D
                // Order 2: cell + hypercube of 5 coarser cells --> 1 + 5= 6 in 1D
                //                                                  1 +25=21 in 2D
                static constexpr std::size_t pred_stencil_size = 1 + ce_pow(2 * prediction_stencil_radius + 1, dim);

#ifdef SAMURAI_WITH_MPI
                // assume that the whole stencil can be on another process
                PetscInt o_nnz_value = mpi::communicator().size() == 1 ? 0 : pred_stencil_size;
#endif

                for_each_prediction_ghost(mesh(),
                                          [&](const auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                                              {
                                                  std::size_t row = static_cast<std::size_t>(local_row_index(ghost, field_i));
#ifdef SAMURAI_WITH_MPI
                                                  if (is_locally_owned(ghost))
                                                  {
                                                      d_nnz[row] = pred_stencil_size;
                                                      o_nnz[row] = o_nnz_value;
                                                  }
#else
                                                  if constexpr (ghost_elimination_enabled)
                                                  {
                                                      d_nnz[row] = static_cast<PetscInt>(
                                                          m_ghost_recursion.at(ghost.index).size());
                                                  }
                                                  else
                                                  {
                                                      d_nnz[row] = pred_stencil_size;
                                                  }
#endif
                                              }
                                          });
            }

            virtual void enforce_projection_prediction(Vec& b) const
            {
                // Projection
                for_each_projection_ghost(mesh(),
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                                              {
                                                  VecSetValueLocal(b, local_row_index(ghost, field_i), 0, INSERT_VALUES);
                                              }
                                          });

                // Prediction
                for_each_prediction_ghost(mesh(),
                                          [&](auto& ghost)
                                          {
                                              for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                                              {
                                                  VecSetValueLocal(b, local_row_index(ghost, field_i), 0, INSERT_VALUES);
                                              }
                                          });
            }

          public:

            void assemble_projection([[maybe_unused]] Mat& A) override
            {
                if constexpr (!ghost_elimination_enabled)
                {
                    static constexpr PetscInt number_of_children = (1 << dim);

                    for_each_projection_ghost_and_children_cells<PetscInt>(
                        mesh(),
                        [&](auto level, PetscInt ghost, const std::array<PetscInt, static_cast<std::size_t>(number_of_children)>& children)
                        {
                            if (!is_locally_owned(static_cast<std::size_t>(ghost))) // cppcheck-suppress knownConditionTrueFalse
                            {
                                return;
                            }

                            double h       = mesh().cell_length(level);
                            double scaling = 1. / (h * h);
                            for (unsigned int field_i = 0; field_i < output_n_comp; ++field_i)
                            {
                                PetscInt ghost_index = local_row_index(ghost, field_i);
                                MatSetValueLocal(A, ghost_index, ghost_index, scaling, current_insert_mode());
                                for (unsigned int i = 0; i < number_of_children; ++i)
                                {
                                    auto error = MatSetValueLocal(A,
                                                                  ghost_index,
                                                                  local_col_index(children[i], field_i),
                                                                  -scaling / number_of_children,
                                                                  current_insert_mode());
                                    if (error)
                                    {
                                        std::cerr << scheme().name() << ": failure to insert projection coefficient at (" << ghost_index
                                                  << ", " << local_col_index(children[i], field_i) << ")." << std::endl;
                                        assert(false);
                                        exit(EXIT_FAILURE);
                                    }
                                }
                                set_is_row_not_empty(ghost_index);
                            }
                        });
                }
            }

            void assemble_prediction([[maybe_unused]] Mat& A) override
            {
                if constexpr (!ghost_elimination_enabled)
                {
                    static_assert(dim >= 1 && dim <= 3, "assemble_prediction() is not implemented for this dimension.");
                    if constexpr (dim == 1)
                    {
                        assemble_prediction_1D(A);
                    }
                    else if constexpr (dim == 2)
                    {
                        assemble_prediction_2D(A);
                    }
                    else if constexpr (dim == 3)
                    {
                        assemble_prediction_3D(A);
                    }
                }
            }

            template <class Cell>
            auto prediction_linear_combination(const Cell& ghost)
            {
                static_assert(dim >= 1 && dim <= 3, "prediction_linear_combination() is not implemented for this dimension.");
                if constexpr (dim == 1)
                {
                    return prediction_linear_combination_1D(ghost);
                }
                else if constexpr (dim == 2)
                {
                    return prediction_linear_combination_2D(ghost);
                }
                else if constexpr (dim == 3)
                {
                    return prediction_linear_combination_3D(ghost);
                }
            }

          private:

            void assemble_prediction_1D(Mat& A)
            {
                // double scalar = 1;
                //  if constexpr (is_FluxBasedScheme_v<Scheme>)
                //  {
                //      scalar = scheme().scalar();
                //  }

                samurai::for_each_prediction_ghost(
                    mesh(),
                    [&](auto& ghost)
                    {
                        if (!is_locally_owned(ghost)) // cppcheck-suppress knownConditionTrueFalse
                        {
                            return;
                        }

                        double h       = mesh().cell_length(ghost.level);
                        double scaling = 1. / (h * h);
                        for (unsigned int field_i = 0; field_i < input_n_comp; ++field_i)
                        {
                            PetscInt ghost_index = this->local_row_index(ghost, field_i);
                            MatSetValueLocal(A, ghost_index, ghost_index, scaling, current_insert_mode());

                            auto ii      = ghost.indices(0);
                            auto ig      = ii >> 1;
                            double isign = (ii & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(isign);

                            auto parent_index = this->local_col_index(static_cast<PetscInt>(this->mesh().get_index(ghost.level - 1, ig)),
                                                                      field_i);
                            MatSetValueLocal(A, ghost_index, parent_index, -scaling, current_insert_mode());

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                if (ci != prediction_stencil_radius)
                                {
                                    double value           = -interpx[ci];
                                    auto coarse_cell_index = this->local_col_index(
                                        static_cast<PetscInt>(
                                            this->mesh().get_index(ghost.level - 1,
                                                                   ig + static_cast<coord_index_t>(ci - prediction_stencil_radius))),
                                        field_i);
                                    MatSetValueLocal(A, ghost_index, coarse_cell_index, scaling * value, current_insert_mode());
                                }
                            }
                            set_is_row_not_empty(ghost_index);
                        }
                    });
            }

            template <class Cell>
            auto prediction_linear_combination_1D(const Cell& ghost)
            {
                std::vector<cell_coeff_pair_t> linear_comb;

                auto ii      = ghost.indices(0);
                auto ig      = ii >> 1;
                double isign = (ii & 1) ? -1 : 1;

                auto interpx = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(isign);

                auto parent_index = this->mesh().get_index(ghost.level - 1, ig);
                linear_comb.emplace_back(parent_index, 1.);

                for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                {
                    if (ci != prediction_stencil_radius)
                    {
                        auto coarse_cell_index = this->mesh().get_index(ghost.level - 1,
                                                                        ig + static_cast<coord_index_t>(ci - prediction_stencil_radius));
                        linear_comb.emplace_back(coarse_cell_index, interpx[ci]);
                    }
                }
                return linear_comb;
            }

            void assemble_prediction_2D(Mat& A)
            {
                samurai::for_each_prediction_ghost(
                    mesh(),
                    [&](auto& ghost)
                    {
                        if (!is_locally_owned(ghost)) // cppcheck-suppress knownConditionTrueFalse
                        {
                            return;
                        }

                        double h       = mesh().cell_length(ghost.level);
                        double scaling = 1. / (h * h);
                        for (unsigned int field_i = 0; field_i < input_n_comp; ++field_i)
                        {
                            PetscInt ghost_index = this->local_row_index(ghost, field_i);
                            MatSetValueLocal(A, ghost_index, ghost_index, scaling, current_insert_mode());

                            auto ii      = ghost.indices(0);
                            auto ig      = ii >> 1;
                            auto j       = ghost.indices(1);
                            auto jg      = j >> 1;
                            double isign = (ii & 1) ? -1 : 1;
                            double jsign = (j & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(isign);
                            auto interpy = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(jsign);

                            auto parent_index = this->local_col_index(static_cast<PetscInt>(this->mesh().get_index(ghost.level - 1, ig, jg)),
                                                                      field_i);
                            MatSetValueLocal(A, ghost_index, parent_index, -scaling, current_insert_mode());

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                                {
                                    if (ci != prediction_stencil_radius || cj != prediction_stencil_radius)
                                    {
                                        double value           = -interpx[ci] * interpy[cj];
                                        auto coarse_cell_index = this->local_col_index(
                                            static_cast<PetscInt>(
                                                this->mesh().get_index(ghost.level - 1,
                                                                       ig + static_cast<coord_index_t>(ci - prediction_stencil_radius),
                                                                       jg + static_cast<coord_index_t>(cj - prediction_stencil_radius))),
                                            field_i);
                                        MatSetValueLocal(A, ghost_index, coarse_cell_index, scaling * value, current_insert_mode());
                                    }
                                }
                            }
                            set_is_row_not_empty(ghost_index);
                        }
                    });
            }

            template <class Cell>
            auto prediction_linear_combination_2D(const Cell& ghost)
            {
                std::vector<cell_coeff_pair_t> linear_comb;

                auto ii      = ghost.indices(0);
                auto ig      = ii >> 1;
                auto j       = ghost.indices(1);
                auto jg      = j >> 1;
                double isign = (ii & 1) ? -1 : 1;
                double jsign = (j & 1) ? -1 : 1;

                auto interpx = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(isign);
                auto interpy = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(jsign);

                auto parent_index = this->mesh().get_index(ghost.level - 1, ig, jg);
                linear_comb.emplace_back(parent_index, 1.);

                for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                {
                    for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                    {
                        if (ci != prediction_stencil_radius || cj != prediction_stencil_radius)
                        {
                            auto coarse_cell_index = this->mesh().get_index(ghost.level - 1,
                                                                            ig + static_cast<coord_index_t>(ci - prediction_stencil_radius),
                                                                            jg + static_cast<coord_index_t>(cj - prediction_stencil_radius));
                            linear_comb.emplace_back(coarse_cell_index, interpx[ci] * interpy[cj]);
                        }
                    }
                }
                return linear_comb;
            }

            void assemble_prediction_3D(Mat& A)
            {
                samurai::for_each_prediction_ghost(
                    mesh(),
                    [&](auto& ghost)
                    {
                        if (!is_locally_owned(ghost)) // cppcheck-suppress knownConditionTrueFalse
                        {
                            return;
                        }

                        double h       = mesh().cell_length(ghost.level);
                        double scaling = 1. / (h * h);
                        for (unsigned int field_i = 0; field_i < input_n_comp; ++field_i)
                        {
                            PetscInt ghost_index = this->local_row_index(ghost, field_i);
                            MatSetValueLocal(A, ghost_index, ghost_index, scaling, current_insert_mode());

                            auto ii      = ghost.indices(0);
                            auto ig      = ii >> 1;
                            auto j       = ghost.indices(1);
                            auto jg      = j >> 1;
                            auto k       = ghost.indices(2);
                            auto kg      = k >> 1;
                            double isign = (ii & 1) ? -1 : 1;
                            double jsign = (j & 1) ? -1 : 1;
                            double ksign = (k & 1) ? -1 : 1;

                            auto interpx = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(isign);
                            auto interpy = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(jsign);
                            auto interpz = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(ksign);

                            auto parent_index = this->local_col_index(
                                static_cast<PetscInt>(this->mesh().get_index(ghost.level - 1, ig, jg, kg)),
                                field_i);
                            MatSetValueLocal(A, ghost_index, parent_index, -scaling, current_insert_mode());

                            for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                            {
                                for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                                {
                                    for (std::size_t ck = 0; ck < interpz.size(); ++ck)
                                    {
                                        if (ci != prediction_stencil_radius || cj != prediction_stencil_radius
                                            || ck != prediction_stencil_radius)
                                        {
                                            double value           = -interpx[ci] * interpy[cj] * interpz[ck];
                                            auto coarse_cell_index = this->local_col_index(
                                                static_cast<PetscInt>(this->mesh().get_index(
                                                    ghost.level - 1,
                                                    ig + static_cast<coord_index_t>(ci - prediction_stencil_radius),
                                                    jg + static_cast<coord_index_t>(cj - prediction_stencil_radius),
                                                    kg + static_cast<coord_index_t>(ck - prediction_stencil_radius))),
                                                field_i);
                                            MatSetValueLocal(A, ghost_index, coarse_cell_index, scaling * value, current_insert_mode());
                                        }
                                    }
                                }
                            }
                            set_is_row_not_empty(ghost_index);
                        }
                    });
            }

            template <class Cell>
            auto prediction_linear_combination_3D(const Cell& ghost)
            {
                std::vector<cell_coeff_pair_t> linear_comb;

                auto ii      = ghost.indices(0);
                auto ig      = ii >> 1;
                auto j       = ghost.indices(1);
                auto jg      = j >> 1;
                auto k       = ghost.indices(2);
                auto kg      = k >> 1;
                double isign = (ii & 1) ? -1 : 1;
                double jsign = (j & 1) ? -1 : 1;
                double ksign = (k & 1) ? -1 : 1;

                auto interpx = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(isign);
                auto interpy = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(jsign);
                auto interpz = samurai::interp_coeffs<2 * prediction_stencil_radius + 1>(ksign);

                auto parent_index = this->mesh().get_index(ghost.level - 1, ig, jg, kg);
                linear_comb.emplace_back(parent_index, 1.);

                for (std::size_t ci = 0; ci < interpx.size(); ++ci)
                {
                    for (std::size_t cj = 0; cj < interpy.size(); ++cj)
                    {
                        for (std::size_t ck = 0; ck < interpz.size(); ++ck)
                        {
                            if (ci != prediction_stencil_radius || cj != prediction_stencil_radius || ck != prediction_stencil_radius)
                            {
                                auto coarse_cell_index = this->mesh().get_index(
                                    ghost.level - 1,
                                    ig + static_cast<coord_index_t>(ci - prediction_stencil_radius),
                                    jg + static_cast<coord_index_t>(cj - prediction_stencil_radius),
                                    kg + static_cast<coord_index_t>(ck - prediction_stencil_radius));
                                linear_comb.emplace_back(coarse_cell_index, interpx[ci] * interpy[cj] * interpz[ck]);
                            }
                        }
                    }
                }
                return linear_comb;
            }

          public:

            bool matrix_is_symmetric() const override
            {
                return scheme().is_symmetric() && is_uniform(mesh());
            }

            bool matrix_is_spd() const override
            {
                return scheme().is_spd() && is_uniform(mesh());
            }
        };

    } // end namespace petsc
} // end namespace samurai
