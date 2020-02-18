add_executable(lbm_d1q2_burgers D1Q2_Burgers.cpp ${MURE_HEADERS})
target_link_libraries(lbm_d1q2_burgers mure)

# add_executable(lbm_d2q4_advection D2Q4_advection.cpp)
# target_link_libraries(lbm_d2q4_advection mure)

add_executable(advection_2d advection_2d.cpp)
target_link_libraries(advection_2d mure)

# add_executable(adv_ana_2d test_analytic_advection_2d.cpp)
# target_link_libraries(adv_ana_2d mure)