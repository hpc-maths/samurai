#pragma once
#include "petsc_cell_based_scheme_assembly.hpp"

namespace samurai { namespace petsc
{
    /**
     * @class PetscBackwardEuler
    */
    template<class Operator>
    class PetscBackwardEuler : public PetscCellBasedSchemeAssembly<typename Operator::cfg_t, typename Operator::field_t>
    {
    public:
        using cfg = typename Operator::cfg_t;
        using field_t = typename Operator::field_t;
        using Mesh = typename field_t::mesh_t;
        using boundary_condition_t = typename field_t::boundary_condition_t;
    private:
        const Operator& _operator;
        double _dt;

    public:
        PetscBackwardEuler(Operator& op, double dt) : 
            PetscCellBasedSchemeAssembly<cfg, field_t>(op.mesh, op.stencil(), [&](double h) { return coefficients(h); }, op.boundary_conditions()),
            _operator(op),
            _dt(dt)
        {}

    private:
        bool matrix_is_spd() const override
        {
            return _operator.matrix_is_spd();
        }

        std::array<double, cfg::scheme_stencil_size> coefficients(double h)
        {
            auto coeffs = _operator.coefficients(h);
            for (unsigned int i=0; i<cfg::scheme_stencil_size; ++i)
            {
                coeffs[i] *= _dt;
            }
            coeffs[cfg::center_index] += 1;
            return coeffs;
        }
    };
    
}} // end namespace