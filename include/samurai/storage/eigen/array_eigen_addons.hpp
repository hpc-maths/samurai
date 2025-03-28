const CwiseBinaryOp<internal::scalar_bitwise_or_op<Scalar>, const Derived, const ConstantReturnType> operator|(const Scalar& scalar) const
{
    return CwiseBinaryOp<internal::scalar_bitwise_or_op<Scalar>, const Derived, const ConstantReturnType>(
        derived(),
        Derived::PlainObject::Constant(rows(), cols(), scalar));
}

friend const CwiseBinaryOp<internal::scalar_bitwise_or_op<Scalar>, const ConstantReturnType, Derived>
operator|(const Scalar& scalar, const ArrayBase<Derived>& mat)
{
    return CwiseBinaryOp<internal::scalar_bitwise_or_op<Scalar>, const ConstantReturnType, Derived>(
        Derived::PlainObject::Constant(rows(), cols(), scalar),
        mat.derived());
}

template <typename OtherDerived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator|=(const ArrayBase<OtherDerived>& other)
{
    derived() = derived() | other;
    return derived();
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator|=(const Scalar& s)
{
    derived() = derived() | s;
    return derived();
}

// EIGEN_MAKE_SCALAR_BINARY_OP(operator&, bitwise_and)

const CwiseBinaryOp<internal::scalar_bitwise_and_op<Scalar>, const ConstantReturnType, const Derived> operator&(const Scalar& scalar) const
{
    return CwiseBinaryOp<internal::scalar_bitwise_and_op<Scalar>, const ConstantReturnType, const Derived>(
        Derived::PlainObject::Constant(rows(), cols(), scalar),
        derived());
}

friend const CwiseBinaryOp<internal::scalar_bitwise_and_op<Scalar>, const ConstantReturnType, Derived>
operator&(const Scalar& scalar, const ArrayBase<Derived>& mat)
{
    return CwiseBinaryOp<internal::scalar_bitwise_and_op<Scalar>, const ConstantReturnType, Derived>(Constant(rows(), cols(), scalar),
                                                                                                     mat.derived());
}

template <typename OtherDerived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator&=(const ArrayBase<OtherDerived>& other)
{
    derived() = derived() & other;
    return derived();
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator&=(const Scalar& s)
{
    derived() = derived() & s;
    return derived();
}
