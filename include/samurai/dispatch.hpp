#pragma once

#include <xtl/xmultimethods.hpp>

namespace samurai
{
    namespace mpl = xtl::mpl;

    /*********************
     * static_dispatcher *
     *********************/

    template <class executor, class base_lhs, class lhs_type_list, class return_type = void>
    class unit_static_dispatcher
    {
      private:

        template <class lhs_type, class... Args>
        static return_type invoke_executor(lhs_type& lhs, Args&&... args)
        {
            executor exec;
            return exec.run(lhs, std::forward<Args>(args)...);
        }

        template <class... Args>
        static return_type dispatch_lhs(base_lhs& lhs, mpl::vector<>, Args&&... args)
        {
            executor exec;
            return exec.on_error(lhs, std::forward<Args>(args)...);
        }

        template <class T, class... U, class... Args>
        static return_type dispatch_lhs(base_lhs& lhs, mpl::vector<T, U...>, Args&&... args)
        {
            if (T* p = dynamic_cast<T*>(&lhs))
            {
                return invoke_executor(*p, std::forward<Args>(args)...);
            }
            return dispatch_lhs(lhs, mpl::vector<U...>(), std::forward<Args>(args)...);
        }

      public:

        template <class... Args>
        static return_type dispatch(base_lhs& lhs, Args&&... args)
        {
            return dispatch_lhs(lhs, lhs_type_list(), std::forward<Args>(args)...);
        }
    };
}
