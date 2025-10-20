# Changelog

## [0.27.0](https://github.com/hpc-maths/samurai/compare/v0.26.1...v0.27.0) (2025-10-20)


### Features

* automatic update_ghost_mr() ([#370](https://github.com/hpc-maths/samurai/issues/370)) ([de19ff4](https://github.com/hpc-maths/samurai/commit/de19ff480f62c8a44711d1bc4123e9acc4b64c5e))
* periodic BC in implicit ([#375](https://github.com/hpc-maths/samurai/issues/375)) ([5946747](https://github.com/hpc-maths/samurai/commit/59467470f06ef3849ad601498c9d5e71abe68800))
* PETSc dependency managed by samurai ([#365](https://github.com/hpc-maths/samurai/issues/365)) ([9f6036f](https://github.com/hpc-maths/samurai/commit/9f6036f548d3c4f81d798c11f549fcc84e9f49e7))
* schemes require `output_field` instead of `output_n_comp` ([#368](https://github.com/hpc-maths/samurai/issues/368)) ([18142c7](https://github.com/hpc-maths/samurai/commit/18142c7d857e6f2626b913e69082d4bc10f12097))


### Bug Fixes

* Fix rotation stencil ([#391](https://github.com/hpc-maths/samurai/issues/391)) ([9081575](https://github.com/hpc-maths/samurai/commit/9081575ffe1bfd0d1664ab931f1ae20d2ddeab58))
* h_face ([#371](https://github.com/hpc-maths/samurai/issues/371)) ([5742dfb](https://github.com/hpc-maths/samurai/commit/5742dfb5a00eaeb6d485415b03d0d241fdbb7b05))
* save issue in demos ([#384](https://github.com/hpc-maths/samurai/issues/384)) ([61f56b3](https://github.com/hpc-maths/samurai/commit/61f56b33e644a8a8a17bd1360c7f57414fe446b0))
* set HighFive version to 2.10 to fix libdeps error in CI ([#376](https://github.com/hpc-maths/samurai/issues/376)) ([5112594](https://github.com/hpc-maths/samurai/commit/51125942f3136ee114cb52aef95409617d5eabd3))
* timers output formatting and deviation computation ([#381](https://github.com/hpc-maths/samurai/issues/381)) ([2216ff4](https://github.com/hpc-maths/samurai/commit/2216ff4f0a8cd185b5bd77462d0b7f8171b3485b))


### Performance Improvements

* some performance improvement in for_each_interface ([#373](https://github.com/hpc-maths/samurai/issues/373)) ([e4a62ca](https://github.com/hpc-maths/samurai/commit/e4a62ca157f8a88a5253944057d69b501c0ba906))

## [0.26.1](https://github.com/hpc-maths/samurai/compare/v0.26.0...v0.26.1) (2025-07-25)


### Bug Fixes

* user prediction function ([#366](https://github.com/hpc-maths/samurai/issues/366)) ([6019432](https://github.com/hpc-maths/samurai/commit/60194328994dfee39b26222d8473bbbd9563d37c))

## [0.26.0](https://github.com/hpc-maths/samurai/compare/v0.25.1...v0.26.0) (2025-07-24)


### Features

* add finer-level-flux option ([#359](https://github.com/hpc-maths/samurai/issues/359)) ([c35e852](https://github.com/hpc-maths/samurai/commit/c35e852bf29a72e109a531d2c2458d191b722efe))
* add relative detail computation + MRA options ([#362](https://github.com/hpc-maths/samurai/issues/362)) ([a49cde6](https://github.com/hpc-maths/samurai/commit/a49cde6d5433df184f257dc262d346bb5f1cca96))
* Add row_index() and col_index() to operator sum ([#358](https://github.com/hpc-maths/samurai/issues/358)) ([3b981e7](https://github.com/hpc-maths/samurai/commit/3b981e71976595433907ce9215d602180258fe16))
* add the possibility to define a user prediction function ([#355](https://github.com/hpc-maths/samurai/issues/355)) ([870dd9e](https://github.com/hpc-maths/samurai/commit/870dd9ecae4da6008d02358d19156f0c2b359aaa))
* obstacles ([#349](https://github.com/hpc-maths/samurai/issues/349)) ([ff287c0](https://github.com/hpc-maths/samurai/commit/ff287c0e22e39f574c50967b2874bc658b9027f9))
* save debug informations via `--save-debug-fields` ([#352](https://github.com/hpc-maths/samurai/issues/352)) ([2ce85c8](https://github.com/hpc-maths/samurai/commit/2ce85c861805301c4d3f4e5b6a5c05616efc68a7))


### Bug Fixes

* finer level flux for `VectorField` ([#360](https://github.com/hpc-maths/samurai/issues/360)) ([e32a1d3](https://github.com/hpc-maths/samurai/commit/e32a1d3d46019958c53b59761679fc9a128a1646))
* hdf5 ([#356](https://github.com/hpc-maths/samurai/issues/356)) ([30c5bde](https://github.com/hpc-maths/samurai/commit/30c5bdec01bcb2843628ac6b503f59d88f5c3019))
* periodic directions ([#354](https://github.com/hpc-maths/samurai/issues/354)) ([8d07a54](https://github.com/hpc-maths/samurai/commit/8d07a540ecb62f3ea3d23b78271e75496bc51dd5))

## [0.25.1](https://github.com/hpc-maths/samurai/compare/v0.25.0...v0.25.1) (2025-07-08)


### Bug Fixes

* fix MRAdaptation with fields to adapt ([#350](https://github.com/hpc-maths/samurai/issues/350)) ([bbe3fc5](https://github.com/hpc-maths/samurai/commit/bbe3fc5967ce772e80169b2474d9394be5d97f3d))

## [0.25.0](https://github.com/hpc-maths/samurai/compare/v0.24.0...v0.25.0) (2025-07-04)


### Features

* find neighbourhood ([#345](https://github.com/hpc-maths/samurai/issues/345)) ([6dfacf9](https://github.com/hpc-maths/samurai/commit/6dfacf9414dc0a87a23a9ca5183257ff01fb4039))
* parallel flux ([#343](https://github.com/hpc-maths/samurai/issues/343)) ([7fb1273](https://github.com/hpc-maths/samurai/commit/7fb12739bacd0d341589276c064db189411fe448))


### Bug Fixes

* add direction and cell to implicit BC ([#346](https://github.com/hpc-maths/samurai/issues/346)) ([3d47b9d](https://github.com/hpc-maths/samurai/commit/3d47b9df137e87ca0616adf50cb8faecf08007f8))
* fix template deduction for mat_vec function ([#339](https://github.com/hpc-maths/samurai/issues/339)) ([7d95e3f](https://github.com/hpc-maths/samurai/commit/7d95e3f80dc23790c54fc6a2d5a92e3f5029adb1))
* graduation when periodic conditions are used ([#347](https://github.com/hpc-maths/samurai/issues/347)) ([c6fc698](https://github.com/hpc-maths/samurai/commit/c6fc698c7aef09b937d48644c35cf7637cac6bd7))


### Performance Improvements

* projected/predicted boundary conditions ([#327](https://github.com/hpc-maths/samurai/issues/327)) ([178300d](https://github.com/hpc-maths/samurai/commit/178300dbc40c46d55a0940a6e8d4f6ad33e0648b))

## [0.24.0](https://github.com/hpc-maths/samurai/compare/v0.23.0...v0.24.0) (2025-06-25)


### Features

* check if a set is empty ([#332](https://github.com/hpc-maths/samurai/issues/332)) ([f0cf22d](https://github.com/hpc-maths/samurai/commit/f0cf22df3e2e95c08a5a1847a28f82ba5104fcd1))
* parallel restart ([#331](https://github.com/hpc-maths/samurai/issues/331)) ([e7b7a1f](https://github.com/hpc-maths/samurai/commit/e7b7a1f4e02e9f8ee95128e673746769f64e95cd))
* perdiodic mpi now works ([#328](https://github.com/hpc-maths/samurai/issues/328)) ([2e1320f](https://github.com/hpc-maths/samurai/commit/2e1320f331b53d75d952f749416bdf3fdcb4b7aa))
* prevent level jumps near the boundary ([#320](https://github.com/hpc-maths/samurai/issues/320)) ([89236ca](https://github.com/hpc-maths/samurai/commit/89236cadf4716e7ae0c01de742b1bde20ce8f708))


### Bug Fixes

* correct timer ([#334](https://github.com/hpc-maths/samurai/issues/334)) ([934c17d](https://github.com/hpc-maths/samurai/commit/934c17dd1b2ebbd0122f6194dd4a117e4423ec71))
* forward declare LevelCellArray in static_algorithm.hpp ([#330](https://github.com/hpc-maths/samurai/issues/330)) ([4d49633](https://github.com/hpc-maths/samurai/commit/4d4963374b49c698d9fc53888f2387e42bf9de78))
* periodic ([#341](https://github.com/hpc-maths/samurai/issues/341)) ([d2dca87](https://github.com/hpc-maths/samurai/commit/d2dca873b5354e0ee5c38135894c3b96e5e589d4))
* remove warning ([#319](https://github.com/hpc-maths/samurai/issues/319)) ([5ba1d3e](https://github.com/hpc-maths/samurai/commit/5ba1d3e9c38f93e33668bf7bbf9b685e40162ef6))
* subset ([#329](https://github.com/hpc-maths/samurai/issues/329)) ([5c47f38](https://github.com/hpc-maths/samurai/commit/5c47f38cf01012d7a79de68b524110dd6fd4253c))
* subset ([#333](https://github.com/hpc-maths/samurai/issues/333)) ([10e8422](https://github.com/hpc-maths/samurai/commit/10e8422fdbc98f9eb205c9ebeac35a9178b1033c))


### Performance Improvements

* Avoid useless MPI send ([#307](https://github.com/hpc-maths/samurai/issues/307)) ([07794ce](https://github.com/hpc-maths/samurai/commit/07794ce336d648889327943d07b208f8278c688b))
* Naive neighbourhood ([#318](https://github.com/hpc-maths/samurai/issues/318)) ([35ea0e3](https://github.com/hpc-maths/samurai/commit/35ea0e35c50da62f4dacdca8f974fd0f6a49ee86))

## [0.23.0](https://github.com/hpc-maths/samurai/compare/v0.22.0...v0.23.0) (2025-04-18)


### Features

* add copy functions xtensor &lt;-&gt; Petsc Vec ([#309](https://github.com/hpc-maths/samurai/issues/309)) ([c28a2ed](https://github.com/hpc-maths/samurai/commit/c28a2edb0d59b186b9d491d99eb9de7337350969))
* add ScalarField and VectorField ([#297](https://github.com/hpc-maths/samurai/issues/297)) ([b5816e7](https://github.com/hpc-maths/samurai/commit/b5816e7a8ec42390f9295270b09d4a8c91e419d1))
* LevelCellArray with origin point ([#312](https://github.com/hpc-maths/samurai/issues/312)) ([0154c90](https://github.com/hpc-maths/samurai/commit/0154c90e662c250fba7c03c8e829968962557de8))


### Bug Fixes

* enforce xtensor version ([#310](https://github.com/hpc-maths/samurai/issues/310)) ([548819b](https://github.com/hpc-maths/samurai/commit/548819b9c7d95bbdd9ddae2b5d174cf9d10924d1))
* Fix `enable_max_level_flux` for prediction 0 ([#298](https://github.com/hpc-maths/samurai/issues/298)) ([2240ebe](https://github.com/hpc-maths/samurai/commit/2240ebe93811c5ab6b5514bd584b81a66a041fcb))
* Python script to read MPI meshes ([#294](https://github.com/hpc-maths/samurai/issues/294)) ([27bbd69](https://github.com/hpc-maths/samurai/commit/27bbd6911634c1956c1d1b4bd7193f9f2c1d5289))
* remove include ([#281](https://github.com/hpc-maths/samurai/issues/281)) ([288273e](https://github.com/hpc-maths/samurai/commit/288273e16e96f84aba6a194e43ef3cff3cadc31a))
* Repare MPI in 1D ([#293](https://github.com/hpc-maths/samurai/issues/293)) ([f502812](https://github.com/hpc-maths/samurai/commit/f502812cf2aa864917b9533ba45d1a40058405f0))
* Update level_cell_array.hpp ([#286](https://github.com/hpc-maths/samurai/issues/286)) ([584527b](https://github.com/hpc-maths/samurai/commit/584527b977ea3633ffce834d5a21a752a097e06e))


### Performance Improvements

* Avoid multiple serialization ([#287](https://github.com/hpc-maths/samurai/issues/287)) ([79c48ee](https://github.com/hpc-maths/samurai/commit/79c48eef419fe1b50689c984764085c951711136))
* changing graduation to a cell-based algorithm ([#299](https://github.com/hpc-maths/samurai/issues/299)) ([cf8793b](https://github.com/hpc-maths/samurai/commit/cf8793b6fab541e5d4dbb511b06fe76f8173a06c))
* replaced CellList by CellArray in update_field_mr ([#283](https://github.com/hpc-maths/samurai/issues/283)) ([a55f11a](https://github.com/hpc-maths/samurai/commit/a55f11a5267ee885aa12e2f4981dbf3ce6582ae2))

## [0.22.0](https://github.com/hpc-maths/samurai/compare/v0.21.1...v0.22.0) (2025-03-06)


### Features

* finest level flux computation for non-linear schemes ([#268](https://github.com/hpc-maths/samurai/issues/268)) ([780a0ba](https://github.com/hpc-maths/samurai/commit/780a0ba38d53089596f2193003ceaf1a526b582a))
* min and max level can be changed ([#277](https://github.com/hpc-maths/samurai/issues/277)) ([fa9c15d](https://github.com/hpc-maths/samurai/commit/fa9c15d9882ea89eb6f02a5760c195d6e0097550))
* new set algebra ([#275](https://github.com/hpc-maths/samurai/issues/275)) ([c184c7f](https://github.com/hpc-maths/samurai/commit/c184c7f3cd36d7d88cde243f9ea35bad92081545))
* Refine boundary ([#269](https://github.com/hpc-maths/samurai/issues/269)) ([daad5ac](https://github.com/hpc-maths/samurai/commit/daad5ac9ece6ee022404367ec424c8991a414793))
* restart ([#278](https://github.com/hpc-maths/samurai/issues/278)) ([f287f14](https://github.com/hpc-maths/samurai/commit/f287f140c412c13011700d5cd9e088e5ffae6194))


### Bug Fixes

* MPI for vector fields ([#273](https://github.com/hpc-maths/samurai/issues/273)) ([adcb4d8](https://github.com/hpc-maths/samurai/commit/adcb4d8029a231684a6086095a784c49c8d8c0bd))


### Performance Improvements

* minor improvement in algebra of sets ([#279](https://github.com/hpc-maths/samurai/issues/279)) ([9958484](https://github.com/hpc-maths/samurai/commit/9958484407bbed763157cddcf6374d3743492adb))

## [0.21.1](https://github.com/hpc-maths/samurai/compare/v0.21.0...v0.21.1) (2025-02-06)


### Bug Fixes

* add make_field_function ([#266](https://github.com/hpc-maths/samurai/issues/266)) ([057bfbe](https://github.com/hpc-maths/samurai/commit/057bfbecba69d112318dea522d395c87967a9432))

## [0.21.0](https://github.com/hpc-maths/samurai/compare/v0.20.0...v0.21.0) (2025-01-26)


### Features

* find cell from cartesian coordinates ([#259](https://github.com/hpc-maths/samurai/issues/259)) ([2432537](https://github.com/hpc-maths/samurai/commit/24325371dae7669931bde3282df614fdd6ce6236))


### Bug Fixes

* cmake samuraiConfig bug ([#255](https://github.com/hpc-maths/samurai/issues/255)) ([265ee64](https://github.com/hpc-maths/samurai/commit/265ee6431767ae38db349704c78a44209851be61))
* default value for SOA in Array ([#258](https://github.com/hpc-maths/samurai/issues/258)) ([4bab319](https://github.com/hpc-maths/samurai/commit/4bab319b5802cc484f303ee5140d3ef196361ac3))
* Fix type issue in Field.operator() ([#260](https://github.com/hpc-maths/samurai/issues/260)) ([df0115f](https://github.com/hpc-maths/samurai/commit/df0115feb475acbbb86163dceddd4d6ff310cc76))

## [0.20.0](https://github.com/hpc-maths/samurai/compare/v0.19.0...v0.20.0) (2025-01-10)


### Features

* new container types ([#222](https://github.com/hpc-maths/samurai/issues/222)) ([85f12b6](https://github.com/hpc-maths/samurai/commit/85f12b6c156827dbdef57793510bf96091163790))

## [0.19.0](https://github.com/hpc-maths/samurai/compare/v0.18.0...v0.19.0) (2024-12-17)


### Features

* fieldsplit PC for monolithic matrix ([#237](https://github.com/hpc-maths/samurai/issues/237)) ([b4bcdb7](https://github.com/hpc-maths/samurai/commit/b4bcdb7fffc4f652344368317670ccc37fe34684))
* Timers ([#235](https://github.com/hpc-maths/samurai/issues/235)) ([7d39d12](https://github.com/hpc-maths/samurai/commit/7d39d126e4c6cc6abef86683fd68cbb3cccc08aa))


### Bug Fixes

* enlarge ([#238](https://github.com/hpc-maths/samurai/issues/238)) ([8cfbea9](https://github.com/hpc-maths/samurai/commit/8cfbea9fa1b4a4c58dfbb420bcfb9b8ace4274e8))
* fix BC ([#248](https://github.com/hpc-maths/samurai/issues/248)) ([9758202](https://github.com/hpc-maths/samurai/commit/9758202baade42020da0e944ce127f6b1168147d))
* Fix prediction order 0 ([#246](https://github.com/hpc-maths/samurai/issues/246)) ([f37575d](https://github.com/hpc-maths/samurai/commit/f37575d48fb33ae070c88f59bb4b7a5a8b2c5d0a))
* projection cells for periodic case ([#240](https://github.com/hpc-maths/samurai/issues/240)) ([6c03245](https://github.com/hpc-maths/samurai/commit/6c03245c7734df3bd4e51e24c02d8e3c5249c8a0))
* the tagged cells with the graduation ([#247](https://github.com/hpc-maths/samurai/issues/247)) ([3156cd2](https://github.com/hpc-maths/samurai/commit/3156cd23fe9438b4efc187d337c5022a08b58ec7))


### Performance Improvements

* Improve nb cells ([#241](https://github.com/hpc-maths/samurai/issues/241)) ([6cd24b5](https://github.com/hpc-maths/samurai/commit/6cd24b5d988f01885dfc43fadec8505511163a07))
* remove apply interval index ([#242](https://github.com/hpc-maths/samurai/issues/242)) ([c3b3f73](https://github.com/hpc-maths/samurai/commit/c3b3f73ecd349c52bea054114735231f530aec6e))
* remove hardcoded access modes in get_interval ([#243](https://github.com/hpc-maths/samurai/issues/243)) ([8fdf11e](https://github.com/hpc-maths/samurai/commit/8fdf11e0bf279fc6de105d6e44669675d5211274))
* replace views/ranges with loops ([#244](https://github.com/hpc-maths/samurai/issues/244)) ([15849e2](https://github.com/hpc-maths/samurai/commit/15849e2fa52de077cad6ae1f4e7d59e7378bdf05))

## [0.18.0](https://github.com/hpc-maths/samurai/compare/v0.17.0...v0.18.0) (2024-11-20)


### Features

* Scaling ([#229](https://github.com/hpc-maths/samurai/issues/229)) ([faf792a](https://github.com/hpc-maths/samurai/commit/faf792a7b9111263e6d303fb7996b630fc5e3478))


### Bug Fixes

* BC no longer required to assemble the matrix of a local scheme ([#231](https://github.com/hpc-maths/samurai/issues/231)) ([2080c00](https://github.com/hpc-maths/samurai/commit/2080c00f4364ec40beb7d643f2674c311e23482a))
* fill corners on uniform meshes ([#234](https://github.com/hpc-maths/samurai/issues/234)) ([59d81b1](https://github.com/hpc-maths/samurai/commit/59d81b15a94e2b0667b71c85c187ca830885a558))
* fix block sizes for nested matrices ([#233](https://github.com/hpc-maths/samurai/issues/233)) ([142be77](https://github.com/hpc-maths/samurai/commit/142be77364e3567f4b4e419d71678465c6d0e66d))
* fmt usage with version 11 ([#226](https://github.com/hpc-maths/samurai/issues/226)) ([b3902b0](https://github.com/hpc-maths/samurai/commit/b3902b08b809183ccc57c3bb0b57bb61862c5d39))
* The matrix size is set upon calling `assembly.set_unknowns(...)`. ([#227](https://github.com/hpc-maths/samurai/issues/227)) ([5d324de](https://github.com/hpc-maths/samurai/commit/5d324de21cda3ec4b847c9e71b9950d6d5feee0c))


### Performance Improvements

* `update_ghost_mr()` performance improved on uniform meshes ([#230](https://github.com/hpc-maths/samurai/issues/230)) ([2596ba7](https://github.com/hpc-maths/samurai/commit/2596ba7421d62b76af13ce60f4bc08ae4556bac0))

## [0.17.0](https://github.com/hpc-maths/samurai/compare/v0.16.0...v0.17.0) (2024-09-10)


### Features

* Boundary region with coordinates ([#221](https://github.com/hpc-maths/samurai/issues/221)) ([5bcb38e](https://github.com/hpc-maths/samurai/commit/5bcb38e49dac15199aab49fd291fe59dc8e44785))
* export samurai CMake options ([#224](https://github.com/hpc-maths/samurai/issues/224)) ([df0ded2](https://github.com/hpc-maths/samurai/commit/df0ded2e1bb65a02a9e3a88de0394f5617de4d35))
* Flux operators applicable by direction ([#223](https://github.com/hpc-maths/samurai/issues/223)) ([91a7082](https://github.com/hpc-maths/samurai/commit/91a70825ae0b5c4b34802e8fc4faab141075f31d))

## [0.16.0](https://github.com/hpc-maths/samurai/compare/v0.15.0...v0.16.0) (2024-06-18)


### Bug Fixes

* Fix compilation error on Get ([#215](https://github.com/hpc-maths/samurai/issues/215)) ([d855db0](https://github.com/hpc-maths/samurai/commit/d855db048871eedd09c80c44daa040a3e16c66f0))
* Fix some minor issues in the flux-based schemes ([#217](https://github.com/hpc-maths/samurai/issues/217)) ([c15b06d](https://github.com/hpc-maths/samurai/commit/c15b06dade07d639797efbb4d5d099348fd3b945))

## [0.15.0](https://github.com/hpc-maths/samurai/compare/v0.14.0...v0.15.0) (2024-06-12)


### Bug Fixes

* create missing prediction cells outside the subdomain ([#212](https://github.com/hpc-maths/samurai/issues/212)) ([fa693e0](https://github.com/hpc-maths/samurai/commit/fa693e00be8c64581dc4a4afda948a55eef66d00))
* Fix flux-based schemes for MPI ([#209](https://github.com/hpc-maths/samurai/issues/209)) ([768a5f4](https://github.com/hpc-maths/samurai/commit/768a5f4dc15faff9e0d9ac53aa2ebd8031a290a1))
* issue with `get_cell` function in LevelCellArray when `dim &gt; 2`. ([#211](https://github.com/hpc-maths/samurai/issues/211)) ([f5fe326](https://github.com/hpc-maths/samurai/commit/f5fe326ef212d79a685e3f7b6137ea2e81b37e55))
* keep_only_one_coarse_tag function ([#210](https://github.com/hpc-maths/samurai/issues/210)) ([d90957a](https://github.com/hpc-maths/samurai/commit/d90957aac53259c1351d4dd6cb3a61ae1b1eb20c))

## [0.14.0](https://github.com/hpc-maths/samurai/compare/v0.13.0...v0.14.0) (2024-05-23)


### Features

* Jacobian matrix for the flux-based operators ([#202](https://github.com/hpc-maths/samurai/issues/202)) ([95c2877](https://github.com/hpc-maths/samurai/commit/95c28778832989dd94a538b50da2e01940d08769))
* OpenMP for independent Newton solvers ([#201](https://github.com/hpc-maths/samurai/issues/201)) ([712a490](https://github.com/hpc-maths/samurai/commit/712a4900d36bbf8e059760235b59120e8eee16a3))


### Bug Fixes

* add fmt:: prefix to function `format_to` ([#204](https://github.com/hpc-maths/samurai/issues/204)) ([2b665ac](https://github.com/hpc-maths/samurai/commit/2b665ac347cba4603c8ec417df93278875c70e68))
* fix row_shift and col_shift for monolithic assembly of an operator sum ([#205](https://github.com/hpc-maths/samurai/issues/205)) ([f44995f](https://github.com/hpc-maths/samurai/commit/f44995fc91636b21cda1366afa0751ade673c4a9))

## [0.13.0](https://github.com/hpc-maths/samurai/compare/v0.12.0...v0.13.0) (2024-05-06)


### Features

* Block matrix improvements ([#198](https://github.com/hpc-maths/samurai/issues/198)) ([7b7c690](https://github.com/hpc-maths/samurai/commit/7b7c690a0b2989d55082114b8eb521234c24b41c))
* Diffusion operator with different coefficient by field component ([#194](https://github.com/hpc-maths/samurai/issues/194)) ([0a53b5e](https://github.com/hpc-maths/samurai/commit/0a53b5e18782aa069d0b99f559c2d7b59560e6f8))


### Bug Fixes

* data structure xtensor by default if existing CMakeCache ([#192](https://github.com/hpc-maths/samurai/issues/192)) ([bc90aef](https://github.com/hpc-maths/samurai/commit/bc90aef741e12baff5c6da9bdadc290cbdfdb7f3))
* detail operator with samurai_check_nan compilation option ([#190](https://github.com/hpc-maths/samurai/issues/190)) ([08cdb9d](https://github.com/hpc-maths/samurai/commit/08cdb9d14b1cc4663a22ca966845b0bf0f1d88a8))
* Fix compilation issue on non-linear cell-based scheme ([#193](https://github.com/hpc-maths/samurai/issues/193)) ([df0ac4f](https://github.com/hpc-maths/samurai/commit/df0ac4f77fda74b7e86e3e0f3ceecf9322d6800a))
* MPI ([#199](https://github.com/hpc-maths/samurai/issues/199)) ([d599908](https://github.com/hpc-maths/samurai/commit/d599908dbd93662345a75828530fb55390dbb391))
* periodic ([#188](https://github.com/hpc-maths/samurai/issues/188)) ([485377a](https://github.com/hpc-maths/samurai/commit/485377a38ff683edd88a1ae46d364bdd47135491))


### Performance Improvements

* change of FluxValue&lt;cfg&gt; structure ([#186](https://github.com/hpc-maths/samurai/issues/186)) ([b1b8bf2](https://github.com/hpc-maths/samurai/commit/b1b8bf2b0f62b736ebd31a2b0ee2590ba3807ce1))
* Vectorization of linear homogeneous operators ([#182](https://github.com/hpc-maths/samurai/issues/182)) ([153b2f2](https://github.com/hpc-maths/samurai/commit/153b2f25489e37884c23f0b681451899c4345caf))

## [0.12.0](https://github.com/hpc-maths/samurai/compare/v0.11.0...v0.12.0) (2024-04-06)


### Features

* multi resolution adaptation ([#171](https://github.com/hpc-maths/samurai/issues/171)) ([60d5892](https://github.com/hpc-maths/samurai/commit/60d5892e79fe59bb5b42dea29d050bce185cd1c7))
* the direction provided in the boundary conditions ([#183](https://github.com/hpc-maths/samurai/issues/183)) ([8bcb601](https://github.com/hpc-maths/samurai/commit/8bcb601d7cbf23036ca7acfe475847b2173505ec))


### Performance Improvements

* Add MR ghosts only if level_min != level_max ([#185](https://github.com/hpc-maths/samurai/issues/185)) ([57d5b73](https://github.com/hpc-maths/samurai/commit/57d5b733328f1eebf9bbff38b7150af12748346a))
* dynamic regions in extrapolation ([#184](https://github.com/hpc-maths/samurai/issues/184)) ([16eef37](https://github.com/hpc-maths/samurai/commit/16eef37379d47c6b2d1c9ade329383b488b74eeb))

## [0.11.0](https://github.com/hpc-maths/samurai/compare/v0.10.0...v0.11.0) (2024-03-20)


### Features

* Add precision (float--&gt;double) in XDMF files ([#179](https://github.com/hpc-maths/samurai/issues/179)) ([6b28811](https://github.com/hpc-maths/samurai/commit/6b28811b4284227a968da34ed09028e9925731ec))
* check for NaN in the computations ([#176](https://github.com/hpc-maths/samurai/issues/176)) ([3954711](https://github.com/hpc-maths/samurai/commit/3954711848b5328c6730dcce8f2475e36e086841))
* High-order Dirichlet ([#178](https://github.com/hpc-maths/samurai/issues/178)) ([3640062](https://github.com/hpc-maths/samurai/commit/3640062bbe3243802228780d9d3b93ccb011b0f9))
* OpenMP on fluxes ([#180](https://github.com/hpc-maths/samurai/issues/180)) ([57f2de7](https://github.com/hpc-maths/samurai/commit/57f2de774b0110b9e9f5a9f8c59a76c9ac1020ae))


### Bug Fixes

* Fix advection_2d_user_bc ([#177](https://github.com/hpc-maths/samurai/issues/177)) ([8d33411](https://github.com/hpc-maths/samurai/commit/8d33411b95f4f0755a1fc43a2d8394532d3b864e))

## [0.10.0](https://github.com/hpc-maths/samurai/compare/v0.9.0...v0.10.0) (2024-02-29)


### Features

* add dim in field_operator_base ([#168](https://github.com/hpc-maths/samurai/issues/168)) ([d74b6ca](https://github.com/hpc-maths/samurai/commit/d74b6cadd85a2337d74fccdbf730e9a1540432cd))
* additional initial value parameter to make_field  ([#162](https://github.com/hpc-maths/samurai/issues/162)) ([3536eac](https://github.com/hpc-maths/samurai/commit/3536eacd5dcc3ab93866d6c3c29e2e7855481ef5))
* Boundary conditions with configurable stencil ([#167](https://github.com/hpc-maths/samurai/issues/167)) ([e59c2a4](https://github.com/hpc-maths/samurai/commit/e59c2a4aa325cdb9aeb570d7b967e0f3b178271a))
* collapsable flux value in non-linear flux-based schemes ([#165](https://github.com/hpc-maths/samurai/issues/165)) ([bca0474](https://github.com/hpc-maths/samurai/commit/bca04741b9034d96a2983fc480a76e879a0a533c))
* const field in FV operator lambdas ([#164](https://github.com/hpc-maths/samurai/issues/164)) ([7a78d2b](https://github.com/hpc-maths/samurai/commit/7a78d2bc6cd7e711574ea35504775d05bb0dd8f2))
* local newton solvers ([#153](https://github.com/hpc-maths/samurai/issues/153)) ([699468e](https://github.com/hpc-maths/samurai/commit/699468eedb78ccec4212afa57ad3db5d8cecd340))
* Non-conservative fluxes ([#161](https://github.com/hpc-maths/samurai/issues/161)) ([61d1cfd](https://github.com/hpc-maths/samurai/commit/61d1cfd4cde415f3478e597a1734e8320f8ddd17))
* Parallelization using MPI ([#28](https://github.com/hpc-maths/samurai/issues/28)) ([a411c76](https://github.com/hpc-maths/samurai/commit/a411c76fe231c57bcfa393125ff9278468cc37cd))
* use xexpression for get_interval, get_index and get_cell ([#170](https://github.com/hpc-maths/samurai/issues/170)) ([9391d2d](https://github.com/hpc-maths/samurai/commit/9391d2ddf7ff5b7125f699bca3ff28de2122448d))
* WENO5 scheme ([#166](https://github.com/hpc-maths/samurai/issues/166)) ([55fe777](https://github.com/hpc-maths/samurai/commit/55fe777899fe5bd2589cb3d0045ec1d120a9c405))


### Bug Fixes

* cell array instead of cell xtensor for the stencil of non-linear cell-based schemes ([#156](https://github.com/hpc-maths/samurai/issues/156)) ([2487af3](https://github.com/hpc-maths/samurai/commit/2487af3f9225841d0eed73e918e64374b268e4bd))
* Fix flux-based schemes with unsymmetric stencil ([#157](https://github.com/hpc-maths/samurai/issues/157)) ([697cca6](https://github.com/hpc-maths/samurai/commit/697cca6222ac7d7290bf14925f712a2ced0b39f5))
* Fix operators for time integration ([#154](https://github.com/hpc-maths/samurai/issues/154)) ([7993525](https://github.com/hpc-maths/samurai/commit/799352508a99f87f93e5b3102b8d5afe266f058e))
* fixing (almost) all warnings triggered when compiling tests & demos ([#163](https://github.com/hpc-maths/samurai/issues/163)) ([246ebcb](https://github.com/hpc-maths/samurai/commit/246ebcba209d2dac958fee39741943c5123c3dd3))
* right shift on interval ([#169](https://github.com/hpc-maths/samurai/issues/169)) ([67f84e1](https://github.com/hpc-maths/samurai/commit/67f84e1909bd0f45e54d6508d0dc11e7941176f3))
* simplifies, fixes and documents left and right shift of Interval ([#172](https://github.com/hpc-maths/samurai/issues/172)) ([db8ce1c](https://github.com/hpc-maths/samurai/commit/db8ce1c5f6c55adbfe78cd8038cec90a9ddcd9a6))

## [0.9.0](https://github.com/hpc-maths/samurai/compare/v0.8.0...v0.9.0) (2023-11-15)


### Features

* simplify bc and remove dispatch ([#141](https://github.com/hpc-maths/samurai/issues/141)) ([7b06d90](https://github.com/hpc-maths/samurai/commit/7b06d90a8aceb844659efcd3fbe7836e7a4643ab))


### Bug Fixes

* detail computation in MRAdapt ([#148](https://github.com/hpc-maths/samurai/issues/148)) ([906c485](https://github.com/hpc-maths/samurai/commit/906c48567a9f67733e313d821ec7f2e2212a2063))
* non-linear schemes for scalar fields ([#150](https://github.com/hpc-maths/samurai/issues/150)) ([bf80fa2](https://github.com/hpc-maths/samurai/commit/bf80fa253a11f331a7dda349ec0c52f8f1248b35))

## [0.8.0](https://github.com/hpc-maths/samurai/compare/v0.7.0...v0.8.0) (2023-10-19)


### Features

* Newton method ([#135](https://github.com/hpc-maths/samurai/issues/135)) ([36b1244](https://github.com/hpc-maths/samurai/commit/36b12446753084ffe295bc13c5f43d168035a987))
* Portion for a slice ([#136](https://github.com/hpc-maths/samurai/issues/136)) ([19e19ab](https://github.com/hpc-maths/samurai/commit/19e19ab5d81f10f8fa98633c731b3e28dffa9a44))

## [0.7.0](https://github.com/hpc-maths/samurai/compare/v0.6.0...v0.7.0) (2023-10-09)


### Features

* add cell_in in functionBC ([#121](https://github.com/hpc-maths/samurai/issues/121)) ([d6fcdc9](https://github.com/hpc-maths/samurai/commit/d6fcdc9043abcc4015bd0ba12f9fd755c9c0ca79))
* Add direction on bc function ([#131](https://github.com/hpc-maths/samurai/issues/131)) ([2d9ae52](https://github.com/hpc-maths/samurai/commit/2d9ae521f168a5841906dc1d3151d23d45dd8f4f))
* add Field_tuple ([#116](https://github.com/hpc-maths/samurai/issues/116)) ([308abad](https://github.com/hpc-maths/samurai/commit/308abad1c8e0980de56ab45fa99fdf5c2d9005f6))
* Add precision (float--&gt;double) in XDMF files ([#119](https://github.com/hpc-maths/samurai/issues/119)) ([3187d1e](https://github.com/hpc-maths/samurai/commit/3187d1eede4359fa875dec17cb6a79abecafb37a))
* add the possibility to change the field name ([#111](https://github.com/hpc-maths/samurai/issues/111)) ([20a0e79](https://github.com/hpc-maths/samurai/commit/20a0e796332cd0de5ee138309d3ee9aafef2249b))
* heat N-D ([#118](https://github.com/hpc-maths/samurai/issues/118)) ([c45c059](https://github.com/hpc-maths/samurai/commit/c45c059a14174e54b45c8cb00be8256fd8163c16))
* linear heterogeneous fluxes ([#126](https://github.com/hpc-maths/samurai/issues/126)) ([bfefd3d](https://github.com/hpc-maths/samurai/commit/bfefd3dd9fb4648b7709173684623577dcc92a22))
* new flux-based schemes ([#117](https://github.com/hpc-maths/samurai/issues/117)) ([1508453](https://github.com/hpc-maths/samurai/commit/15084535dde5fdb92cf134626c1bdd3457ef6ce3))
* non-linear cell schemes ([#130](https://github.com/hpc-maths/samurai/issues/130)) ([08a95af](https://github.com/hpc-maths/samurai/commit/08a95afe7b42586719c15db383baf27f0dccf512))
* non-linear fluxes ([#123](https://github.com/hpc-maths/samurai/issues/123)) ([25431d0](https://github.com/hpc-maths/samurai/commit/25431d05a80c62954aec05784914a113aef2ca17))
* remove field object from operators ([#124](https://github.com/hpc-maths/samurai/issues/124)) ([8c11e00](https://github.com/hpc-maths/samurai/commit/8c11e0072d111d158978714205780c75f8a882dd))
* remove small entries in `prediction_map` ([#115](https://github.com/hpc-maths/samurai/issues/115)) ([f0302a7](https://github.com/hpc-maths/samurai/commit/f0302a75cc66e93d1966ba03782fdb5b053efa31))


### Bug Fixes

* BC function receives boundary coordinates ([#122](https://github.com/hpc-maths/samurai/issues/122)) ([2b47dd7](https://github.com/hpc-maths/samurai/commit/2b47dd7e6a2ed5a3ec815a096786bf8500514afe))
* extract_mesh ([#110](https://github.com/hpc-maths/samurai/issues/110)) ([3c2c05e](https://github.com/hpc-maths/samurai/commit/3c2c05e4e52f69cefa23a1dc12ffa259def6004b))
* Fix transfer() function for vectorial fields ([#128](https://github.com/hpc-maths/samurai/issues/128)) ([5c58983](https://github.com/hpc-maths/samurai/commit/5c589838c23810f94938dbc63a89745cd3c4cd72))
* heat + ghost elimination ([#112](https://github.com/hpc-maths/samurai/issues/112)) ([5e43087](https://github.com/hpc-maths/samurai/commit/5e430875039a062218d72bc170526b3219effc77))
* index for upwind variable scheme to recover i/j symmetry ([#120](https://github.com/hpc-maths/samurai/issues/120)) ([48d7e58](https://github.com/hpc-maths/samurai/commit/48d7e58aef11642651a28cd1c1495d58f1d52258))
* missing coefficient for Neumann BC ([#114](https://github.com/hpc-maths/samurai/issues/114)) ([b3de2cb](https://github.com/hpc-maths/samurai/commit/b3de2cbbeba82cf60d6d8fccf58f3f7f59356ba2))

## [0.6.0](https://github.com/hpc-maths/samurai/compare/v0.5.0...v0.6.0) (2023-07-09)


### Features

* field constructor from expression ([#107](https://github.com/hpc-maths/samurai/issues/107)) ([0375ed9](https://github.com/hpc-maths/samurai/commit/0375ed96272b947bbe05d42a304c3312c4b99e05))


### Bug Fixes

* bc ([#106](https://github.com/hpc-maths/samurai/issues/106)) ([f955a57](https://github.com/hpc-maths/samurai/commit/f955a577da1c95aef1241771ad7ec2ee92414bc9))
* coarsening when a parent intesects the boundary ([#105](https://github.com/hpc-maths/samurai/issues/105)) ([4aa6411](https://github.com/hpc-maths/samurai/commit/4aa6411408704a553e896a317c620b5bc39ce8a6))

## [0.5.0](https://github.com/hpc-maths/samurai/compare/v0.4.0...v0.5.0) (2023-07-05)


### Features

* explicit schemes ([#100](https://github.com/hpc-maths/samurai/issues/100)) ([c73cfd3](https://github.com/hpc-maths/samurai/commit/c73cfd34f7aef7e57947322b5890d651999b1733))
* Field iterator ([#102](https://github.com/hpc-maths/samurai/issues/102)) ([4551eae](https://github.com/hpc-maths/samurai/commit/4551eaec7294befa0fc030520d4218f1d4b9b9f9))
* own bc ([#103](https://github.com/hpc-maths/samurai/issues/103)) ([7d92323](https://github.com/hpc-maths/samurai/commit/7d92323606bba725364bedadfa146d0cac3e6590))

## [0.4.0](https://github.com/hpc-maths/samurai/compare/v0.3.0...v0.4.0) (2023-07-03)


### Features

* add transfer function ([#95](https://github.com/hpc-maths/samurai/issues/95)) ([d778fad](https://github.com/hpc-maths/samurai/commit/d778fadfcb399591e23cea753dcb34d7a4fb47d6))
* create mesh from CAD object ([#51](https://github.com/hpc-maths/samurai/issues/51)) ([8c53fd1](https://github.com/hpc-maths/samurai/commit/8c53fd134098ddcb95e42f0fe1364018bdbfe8be))
* scheme definition outed ([#96](https://github.com/hpc-maths/samurai/issues/96)) ([0c62af4](https://github.com/hpc-maths/samurai/commit/0c62af4adac84144271c3163b4a059dae423e28f))


### Bug Fixes

* const field ([#98](https://github.com/hpc-maths/samurai/issues/98)) ([8387396](https://github.com/hpc-maths/samurai/commit/8387396273565affe45c8fa491677242561dfc48))
* monolithic Stokes assembly ([#94](https://github.com/hpc-maths/samurai/issues/94)) ([358fc37](https://github.com/hpc-maths/samurai/commit/358fc377f9d37547b3bcbba10e06844e24f8085d))

## [0.3.0](https://github.com/hpc-maths/samurai/compare/v0.2.0...v0.3.0) (2023-05-22)


### Features

* monolithic block matrices ([#89](https://github.com/hpc-maths/samurai/issues/89)) ([6e318c2](https://github.com/hpc-maths/samurai/commit/6e318c234910e09adef8c3e00311456145d8a6d7))


### Bug Fixes

* bc when ghost_width &gt; 1 ([#90](https://github.com/hpc-maths/samurai/issues/90)) ([611f1a1](https://github.com/hpc-maths/samurai/commit/611f1a1e75b5c86f00247d16f9abb80d5869007a))
* clang-tidy error fixes ([#88](https://github.com/hpc-maths/samurai/issues/88)) ([bd131fa](https://github.com/hpc-maths/samurai/commit/bd131fa1d72bd8a5837d3ec448ddbb3cf5343939))
* cpp version in target ([#92](https://github.com/hpc-maths/samurai/issues/92)) ([126cf34](https://github.com/hpc-maths/samurai/commit/126cf349d3bb9b6b9ca4b621277194088a809b7e))

## [0.2.0](https://github.com/hpc-maths/samurai/compare/v0.1.1...v0.2.0) (2023-04-22)


### Features

* convergence order computation ([#83](https://github.com/hpc-maths/samurai/issues/83)) ([50d8419](https://github.com/hpc-maths/samurai/commit/50d841903cfe869a49bfc85db82207c8804b2793))
* unification of the boundary conditions ([#82](https://github.com/hpc-maths/samurai/issues/82)) ([647c314](https://github.com/hpc-maths/samurai/commit/647c3147bf0d174032fecbdf24ec28a6d67c33c2))


### Bug Fixes

* small fixes for vcpkg port ([#85](https://github.com/hpc-maths/samurai/issues/85)) ([0172cb2](https://github.com/hpc-maths/samurai/commit/0172cb2f9e4fcb5af0633153de15ac3b1cc9c26a))

## [0.1.1](https://github.com/hpc-maths/samurai/compare/v0.1.0...v0.1.1) (2023-04-18)


### Bug Fixes

* set a name for check-pr-title action ([#80](https://github.com/hpc-maths/samurai/issues/80)) ([7e72642](https://github.com/hpc-maths/samurai/commit/7e72642e327ad39eeeec437bc48e9b5316cc87fb))
