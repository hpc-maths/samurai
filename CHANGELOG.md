# Changelog

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
