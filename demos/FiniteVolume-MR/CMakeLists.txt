# add_executable(VF_advection_mr_2d VF_advection_mr_2d.cpp)
# target_link_libraries(VF_advection_mr_2d samurai)

# add_executable(VF_burgers_scalar_mr_2d VF_burgers_scalar_mr_2d.cpp)
# target_link_libraries(VF_burgers_scalar_mr_2d samurai)

# add_executable(VF_burgers_vect_mr_2d VF_burgers_vect_mr_2d.cpp)
# target_link_libraries(VF_burgers_vect_mr_2d samurai)

# add_executable(lbm_d1q2_burgers D1Q2_Burgers.cpp ${SAMURAI_HEADERS})
# target_link_libraries(lbm_d1q2_burgers samurai)

# add_executable(lbm_d1q2_advection_diffusion D1Q2_advection_diffusion.cpp ${SAMURAI_HEADERS})
# target_link_libraries(lbm_d1q2_advection_diffusion samurai)

# add_executable(D2Q4_advection D2Q4_advection.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q4_advection samurai)

# add_executable(D2Q9_KelvinHelmholtz_Geier D2Q9_KelvinHelmholtz_Geier.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q9_KelvinHelmholtz_Geier samurai)

# add_executable(D2Q4_3_Euler D2Q4_3_Euler.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q4_3_Euler samurai)

add_executable(D2Q4_3_Euler_right_boundary_conditions D2Q4_3_Euler_right_boundary_conditions.cpp)
target_link_libraries(D2Q4_3_Euler_right_boundary_conditions samurai)
#
# add_executable(D2Q4_3_ShallowWaters_right_boundary_conditions D2Q4_3_ShallowWaters_right_boundary_conditions.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q4_3_ShallowWaters_right_boundary_conditions samurai)
#
#add_executable(D2Q4twisted_3_ShallowWaters_right_boundary_conditions D2Q4twisted_3_ShallowWaters_right_boundary_conditions.cpp ${SAMURAI_HEADERS})
#target_link_libraries(D2Q4twisted_3_ShallowWaters_right_boundary_conditions samurai)
#
# add_executable(D2Q4_4_Implosion D2Q4_4_Implosion.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q4_4_Implosion samurai)

# add_executable(D2Q4_advection_right_boundary_conditions D2Q4_advection_right_boundary_conditions.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q4_advection_right_boundary_conditions samurai)

# add_executable(D1Q3_ShallowWaters D1Q3_ShallowWaters.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D1Q3_ShallowWaters samurai)

# add_executable(D1Q2_Burgers D1Q2_Burgers.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D1Q2_Burgers samurai)

# add_executable(hole hole.cpp ${SAMURAI_HEADERS})
# target_link_libraries(hole samurai)

# add_executable(D2Q9_von_Karman_street D2Q9_von_Karman_street.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q9_von_Karman_street samurai)

# add_executable(D2Q9_lid_driven_cavity D2Q9_lid_driven_cavity.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q9_lid_driven_cavity samurai)

# add_executable(D2Q9_von_Karman_street_meshless D2Q9_von_Karman_street_meshless.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q9_von_Karman_street_meshless samurai)

# add_executable(D2Q9_von_Karman_street_meshless_diffusion_correction D2Q9_von_Karman_street_meshless_diffusion_correction.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q9_von_Karman_street_meshless_diffusion_correction samurai)

# add_executable(D2Q4_3_Emery_Dubois D2Q4_3_Emery_Dubois.cpp ${SAMURAI_HEADERS})
# target_link_libraries(D2Q4_3_Emery_Dubois samurai)

# add_executable(finest_reconstruction_2D finest_reconstruction_2D.cpp ${SAMURAI_HEADERS})
# target_link_libraries(finest_reconstruction_2D samurai)

add_executable(test_D1Q2 test_D1Q2.cpp ${SAMURAI_HEADERS})
target_link_libraries(test_D1Q2 samurai)
