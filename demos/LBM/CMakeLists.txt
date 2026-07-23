find_package(cxxopts)

# New schemes/lbm formalism (see notes/schemes-lbm-design.md)

add_executable(lbm-new-D1Q2-advection new_D1Q2_advection.cpp)
target_link_libraries(lbm-new-D1Q2-advection samurai)

add_executable(lbm-new-D2Q4-advection new_D2Q4_advection.cpp)
target_link_libraries(lbm-new-D2Q4-advection samurai)

add_executable(lbm-new-D1Q3-shallow-waters-dam new_D1Q3_shallow_waters_dam.cpp)
target_link_libraries(lbm-new-D1Q3-shallow-waters-dam samurai)

add_executable(lbm-new-D2Q4diag-advection new_D2Q4diag_advection.cpp)
target_link_libraries(lbm-new-D2Q4diag-advection samurai)

add_executable(lbm-new-D2Q9-taylor-green new_D2Q9_taylor_green.cpp)
target_link_libraries(lbm-new-D2Q9-taylor-green samurai)

add_executable(lbm-new-D1Q5-shallow-waters-dam new_D1Q5_shallow_waters_dam.cpp)
target_link_libraries(lbm-new-D1Q5-shallow-waters-dam samurai)

add_executable(lbm-new-D1Q222-euler-sod new_D1Q222_euler_sod.cpp)
target_link_libraries(lbm-new-D1Q222-euler-sod samurai)

add_executable(lbm-new-D2Q4444-euler-lax-liu new_D2Q4444_euler_lax_liu.cpp)
target_link_libraries(lbm-new-D2Q4444-euler-lax-liu samurai)

# ---------------------------------------------------------------------------
# Legacy demos (outdated: overleaves, hand-rolled prediction maps, old BC).
# Kept for reference during the migration to schemes/lbm; not built for now.
# ---------------------------------------------------------------------------

# add_executable(lbm-test-D1Q2 test_D1Q2.cpp)
# target_link_libraries(lbm-test-D1Q2 samurai)

# add_executable(lbm-test-D1Q3 test_D1Q3.cpp)
# target_link_libraries(lbm-test-D1Q3 samurai)

# add_executable(lbm-test-D1Q5 test_D1Q5.cpp)
# target_link_libraries(lbm-test-D1Q5 samurai)

# add_executable(lbm-test-D1Q222 test_D1Q222.cpp)
# target_link_libraries(lbm-test-D1Q222 samurai)

# add_executable(lbm-D1Q2-advection-burgers D1Q2_Advection_and_Burgers.cpp)
# target_link_libraries(lbm-D1Q2-advection-burgers samurai)

# add_executable(lbm-D1Q3-shallow-waters-dam D1Q3_Shallow_Waters_Dam.cpp)
# target_link_libraries(lbm-D1Q3-shallow-waters-dam samurai)

# add_executable(lbm-D1Q5-shallow-waters-dam D1Q5_Shallow_Waters_Dam.cpp)
# target_link_libraries(lbm-D1Q5-shallow-waters-dam samurai)

# add_executable(lbm-D1Q222-euler-sod D1Q222_Euler_Sod.cpp)
# target_link_libraries(lbm-D1Q222-euler-sod samurai)

# add_executable(lbm-D2Q4444-euler-lax-liu D2Q4444_Euler_Lax_Liu.cpp)
# target_link_libraries(lbm-D2Q4444-euler-lax-liu samurai)

# add_executable(lbm-D2Q4444-euler-lax-liu-uniform D2Q4444_Euler_Lax_Liu_uniform.cpp)
# target_link_libraries(lbm-D2Q4444-euler-lax-liu-uniform samurai)

# add_executable(lbm-D2Q9-navier-stokes-von-karman-street D2Q9_Navier_Stokes_von_Karman_street.cpp)
# target_link_libraries(lbm-D2Q9-navier-stokes-von-karman-street samurai)

# add_executable(lbm-D2Q4444-euler-implosion D2Q4444_Euler_Implosion.cpp)
# target_link_libraries(lbm-D2Q4444-euler-implosion samurai)

# add_executable(lbm-D2Q5444-euler-rayleigh-taylor D2Q5444_Euler_Rayleigh_Taylor.cpp)
# target_link_libraries(lbm-D2Q5444-euler-rayleigh-taylor samurai)
