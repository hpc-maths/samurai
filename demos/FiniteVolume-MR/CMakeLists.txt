# add_executable(VF_advection_mr_2d VF_advection_mr_2d.cpp)
# target_link_libraries(VF_advection_mr_2d mure)

# add_executable(VF_burgers_scalar_mr_2d VF_burgers_scalar_mr_2d.cpp)
# target_link_libraries(VF_burgers_scalar_mr_2d mure)

# add_executable(VF_burgers_vect_mr_2d VF_burgers_vect_mr_2d.cpp)
# target_link_libraries(VF_burgers_vect_mr_2d mure)

# add_executable(lbm_d1q2_burgers D1Q2_Burgers.cpp ${MURE_HEADERS})
# target_link_libraries(lbm_d1q2_burgers mure)

# add_executable(lbm_d1q2_advection_diffusion D1Q2_advection_diffusion.cpp ${MURE_HEADERS})
# target_link_libraries(lbm_d1q2_advection_diffusion mure)

# add_executable(D2Q4_advection D2Q4_advection.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_advection mure)

# add_executable(D2Q9_KelvinHelmholtz_Geier D2Q9_KelvinHelmholtz_Geier.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q9_KelvinHelmholtz_Geier mure)

# add_executable(D2Q4_3_Euler D2Q4_3_Euler.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_3_Euler mure)

# add_executable(D2Q4_3_Euler_right_boundary_conditions D2Q4_3_Euler_right_boundary_conditions.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_3_Euler_right_boundary_conditions mure)
# 
# add_executable(D2Q4_3_ShallowWaters_right_boundary_conditions D2Q4_3_ShallowWaters_right_boundary_conditions.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_3_ShallowWaters_right_boundary_conditions mure)
# 
#add_executable(D2Q4twisted_3_ShallowWaters_right_boundary_conditions D2Q4twisted_3_ShallowWaters_right_boundary_conditions.cpp ${MURE_HEADERS})
#target_link_libraries(D2Q4twisted_3_ShallowWaters_right_boundary_conditions mure)
# 
# add_executable(D2Q4_4_Implosion D2Q4_4_Implosion.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_4_Implosion mure)

# add_executable(D2Q4_advection_right_boundary_conditions D2Q4_advection_right_boundary_conditions.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q4_advection_right_boundary_conditions mure)

# add_executable(D1Q2_Burgers D1Q2_Burgers.cpp ${MURE_HEADERS})
# target_link_libraries(D1Q2_Burgers mure)

# add_executable(hole hole.cpp ${MURE_HEADERS})
# target_link_libraries(hole mure)

# add_executable(D2Q9_von_Karman_street D2Q9_von_Karman_street.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q9_von_Karman_street mure)

# add_executable(D2Q9_lid_driven_cavity D2Q9_lid_driven_cavity.cpp ${MURE_HEADERS})
# target_link_libraries(D2Q9_lid_driven_cavity mure)

add_executable(D2Q9_von_Karman_street_meshless D2Q9_von_Karman_street_meshless.cpp ${MURE_HEADERS})
target_link_libraries(D2Q9_von_Karman_street_meshless mure)

add_executable(D2Q9_von_Karman_street_meshless_diffusion_correction D2Q9_von_Karman_street_meshless_diffusion_correction.cpp ${MURE_HEADERS})
target_link_libraries(D2Q9_von_Karman_street_meshless_diffusion_correction mure)

add_executable(D2Q4_3_Emery_Dubois D2Q4_3_Emery_Dubois.cpp ${MURE_HEADERS})
target_link_libraries(D2Q4_3_Emery_Dubois mure)

add_executable(finest_reconstruction_2D finest_reconstruction_2D.cpp ${MURE_HEADERS})
target_link_libraries(finest_reconstruction_2D mure)

