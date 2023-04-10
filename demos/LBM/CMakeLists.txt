# 1D Paper - Verification of the error estimates

add_executable(lbm-test-D1Q2 test_D1Q2.cpp)
target_link_libraries(lbm-test-D1Q2 samurai)

add_executable(lbm-test-D1Q3 test_D1Q3.cpp)
target_link_libraries(lbm-test-D1Q3 samurai)

add_executable(lbm-test-D1Q5 test_D1Q5.cpp)
target_link_libraries(lbm-test-D1Q5 samurai)

add_executable(lbm-test-D1Q222 test_D1Q222.cpp)
target_link_libraries(lbm-test-D1Q222 samurai)

# 1D Paper - Plot solutions

add_executable(lbm-D1Q2-advection-burgers D1Q2_Advection_and_Burgers.cpp)
target_link_libraries(lbm-D1Q2-advection-burgers samurai)

add_executable(lbm-D1Q3-shallow-waters-dam D1Q3_Shallow_Waters_Dam.cpp)
target_link_libraries(lbm-D1Q3-shallow-waters-dam samurai)

add_executable(lbm-D1Q5-shallow-waters-dam D1Q5_Shallow_Waters_Dam.cpp)
target_link_libraries(lbm-D1Q5-shallow-waters-dam samurai)

add_executable(lbm-D1Q222-euler-sod D1Q222_Euler_Sod.cpp)
target_link_libraries(lbm-D1Q222-euler-sod samurai)


# 2D Paper

add_executable(lbm-D2Q4444-euler-lax-liu D2Q4444_Euler_Lax_Liu.cpp)
target_link_libraries(lbm-D2Q4444-euler-lax-liu samurai)

add_executable(lbm-D2Q4444-euler-lax-liu-uniform D2Q4444_Euler_Lax_Liu_uniform.cpp)
target_link_libraries(lbm-D2Q4444-euler-lax-liu-uniform samurai)

add_executable(lbm-D2Q9-navier-stokes-von-karman-street D2Q9_Navier_Stokes_von_Karman_street.cpp)
target_link_libraries(lbm-D2Q9-navier-stokes-von-karman-street samurai)

# Additional 2D tests

add_executable(lbm-D2Q4444-euler-implosion D2Q4444_Euler_Implosion.cpp)
target_link_libraries(lbm-D2Q4444-euler-implosion samurai)

add_executable(lbm-D2Q5444-euler-rayleigh-taylor D2Q5444_Euler_Rayleigh_Taylor.cpp)
target_link_libraries(lbm-D2Q5444-euler-rayleigh-taylor samurai)
