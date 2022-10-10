#pragma once
#include "coarsening.hpp"

namespace samurai_new
{
    enum TransferOperators : int
    {
        // P mat-free, R mat-free (via Fields)
        MatrixFree_Fields = 1,
        // P mat-free, R mat-free (via double*)
        MatrixFree_Arrays,
        // P assembled, R = P^T
        Assembled_PTranspose,
        // P assembled, R = assembled
        Assembled
    };

    template<class Dsctzr>
    class LevelCtx
    {
    public:
        using Mesh = typename Dsctzr::Mesh;

    private:
        Mesh _mesh;
        Dsctzr _discretizer;
    public:
        int level;
        LevelCtx* finer = nullptr;
        LevelCtx* coarser = nullptr;
        TransferOperators transfer_ops;

        LevelCtx(Dsctzr& d, Mesh& m, TransferOperators to) : 
            _discretizer(d), _mesh(m)
        {
            level = 0;
            transfer_ops = to;
        }

        LevelCtx(LevelCtx& fine_ctx) :
            _mesh(samurai_new::coarsen(fine_ctx.mesh())),
            _discretizer(Dsctzr::create_coarse(fine_ctx.discretizer(), _mesh))
        {
            level = fine_ctx.level + 1;
            transfer_ops = fine_ctx.transfer_ops;
            this->finer = &fine_ctx;
            fine_ctx.coarser = this;
            //samurai_new::coarsen(fine_ctx->mesh(), _mesh);
            //_mesh = _hierarchy->add_coarser(fine_ctx->mesh());
            //_mesh_ptr = &_mesh;
            //_discretizer = Dsctzr::create_coarse(fine_ctx->discretizer(), *_mesh);
            //_discretizer_ptr = &_discretizer;
        }

        /*static void create_coarse(LevelCtx& fine_ctx)
        {
            fine_ctx.coarser = new LevelCtx()
        }*/

        Mesh& mesh() { return _mesh; }
        Dsctzr& discretizer() { return _discretizer; }
        bool is_finest() { return level == 0; }


    };

}