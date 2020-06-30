# add_executable(VF_advection_mr_2d VF_advection_mr_2d.cpp)
# target_link_libraries(VF_advection_mr_2d mure)

# add_executable(VF_burgers_scalar_mr_2d VF_burgers_scalar_mr_2d.cpp)
# target_link_libraries(VF_burgers_scalar_mr_2d mure)

# add_executable(VF_burgers_vect_mr_2d VF_burgers_vect_mr_2d.cpp)
# target_link_libraries(VF_burgers_vect_mr_2d mure)

# add_executable(lbm_d1q2_burgers D1Q2_Burgers.cpp ${MURE_HEADERS})
# target_link_libraries(lbm_d1q2_burgers mure)

# add_executable(D2Q4_advection D2Q4_advection.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_advection mure)


# add_executable(D2Q9_KelvinHelmholtz_Geier D2Q9_KelvinHelmholtz_Geier.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q9_KelvinHelmholtz_Geier mure)

# add_executable(D2Q4_3_Euler D2Q4_3_Euler.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_3_Euler mure)

add_executable(D2Q4_3_Euler_right_boundary_conditions D2Q4_3_Euler_right_boundary_conditions.cpp ${MURE_HEADERS})
target_link_libraries(D2Q4_3_Euler_right_boundary_conditions mure)

# add_executable(D2Q4_advection_right_boundary_conditions D2Q4_advection_right_boundary_conditions.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_advection_right_boundary_conditions mure)

# add_executable(D1Q2_Burgers D1Q2_Burgers.cpp ${MURE_HEADERS})
# target_link_libraries(D1Q2_Burgers mure)