#pragma once

namespace samurai
{
    template <class D>
    class crtp_base
    {
      public:

        using derived_type = D;

        const derived_type& derived_cast() const& noexcept;
        derived_type& derived_cast() & noexcept;
        derived_type derived_cast() && noexcept;

      protected:

        crtp_base()  = default;
        ~crtp_base() = default;

        crtp_base(const crtp_base&)            = default;
        crtp_base& operator=(const crtp_base&) = default;

        crtp_base(crtp_base&&) noexcept            = default;
        crtp_base& operator=(crtp_base&&) noexcept = default;
    };

    template <class D>
    inline auto crtp_base<D>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D>
    inline auto crtp_base<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto crtp_base<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }
}
