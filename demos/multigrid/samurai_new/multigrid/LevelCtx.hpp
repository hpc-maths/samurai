#pragma once
#include "coarsening.hpp"

namespace samurai_new
{
    enum TransferOperators : int
    {
        // P assembled, R = assembled
        Assembled = 1,
        // P assembled, R = P^T
        Assembled_PTranspose,
        // P mat-free, R mat-free (via double*)
        MatrixFree_Arrays,
        // P mat-free, R mat-free (via Fields)
        MatrixFree_Fields
    };

    template <class Dsctzr>
    class LevelContext
    {
      public:

        using Mesh = typename Dsctzr::Mesh;

      private:

        Mesh _mesh;
        Dsctzr _discretizer;

      public:

        int level;
        LevelContext* finer   = nullptr;
        LevelContext* coarser = nullptr;
        TransferOperators transfer_ops;
        int prediction_order;

        LevelContext(Dsctzr& d, Mesh& m, TransferOperators to, int pred_order)
            : _mesh(m)
            , _discretizer(d)
        {
            level            = 0;
            transfer_ops     = to;
            prediction_order = pred_order;
        }

        LevelContext(LevelContext& fine_ctx)
            : _mesh(samurai_new::coarsen(fine_ctx.mesh()))
            , _discretizer(Dsctzr::create_coarse(fine_ctx.assembly(), _mesh))
        {
            level            = fine_ctx.level + 1;
            transfer_ops     = fine_ctx.transfer_ops;
            prediction_order = fine_ctx.prediction_order;

            this->finer      = &fine_ctx;
            fine_ctx.coarser = this;
        }

        Mesh& mesh()
        {
            return _mesh;
        }

        Dsctzr& assembly()
        {
            return _discretizer;
        }

        bool is_finest()
        {
            return level == 0;
        }
    };

}
