# Changelog

## [0.34.0](https://github.com/hpc-maths/samurai/compare/v0.33.0...v0.34.0) (2026-10-26)


### Features

* add static_dispatch ([#469](https://github.com/hpc-maths/samurai/issues/469)) ([e94519b](https://github.com/hpc-maths/samurai/commit/e94519b3d6de37ab4697e21be6e34714da1c6fed))
* **interval:** write half-open intervals as [a, b) ([#559](https://github.com/hpc-maths/samurai/issues/559)) ([703bd95](https://github.com/hpc-maths/samurai/commit/703bd9561f610528b6375d56a50c227802144506))
* **lbm:** Lattice Boltzmann schemes formalism (schemes/lbm) ([#480](https://github.com/hpc-maths/samurai/issues/480)) ([d4fb02d](https://github.com/hpc-maths/samurai/commit/d4fb02dc9f1a4440c53f07d34f70ad8e2871d1bb))
* **lbm:** remaining LBM cases (multi-block Euler, source term, wall/inflow BC) ([#483](https://github.com/hpc-maths/samurai/issues/483)) ([2036662](https://github.com/hpc-maths/samurai/commit/2036662e639cd9177187ebbf711fec0ca2dd7dbe))
* **numeric:** shared 1D prediction coefficients and the boundary shift ([#490](https://github.com/hpc-maths/samurai/issues/490)) ([bf730e7](https://github.com/hpc-maths/samurai/commit/bf730e71fa24d4cce8b3d426267209c405f91120))
* **restart:** load a checkpoint on any number of processes ([#724](https://github.com/hpc-maths/samurai/issues/724)) ([a516ab8](https://github.com/hpc-maths/samurai/commit/a516ab8f0d8d8940c1f5fb8b8a04de4e55626a54))
* **subset:** dynamic (runtime) set algebra ([#485](https://github.com/hpc-maths/samurai/issues/485)) ([699c096](https://github.com/hpc-maths/samurai/commit/699c0967c4001a5a0b8114b2205fd5f6eb0dcc90))


### Bug Fixes

* address the C++ SonarQube Cloud findings on main ([#758](https://github.com/hpc-maths/samurai/issues/758)) ([7e1ef84](https://github.com/hpc-maths/samurai/commit/7e1ef84c78774c3746bfa14a93bac59e9dd94746))
* **algorithm:** give the right cell index in for_each_cell on a LevelCellArray ([#744](https://github.com/hpc-maths/samurai/issues/744)) ([5e97bb0](https://github.com/hpc-maths/samurai/commit/5e97bb05a1ffc31659702931940be24bae3b3c01))
* **arguments:** describe --dont-redirect-output as it behaves ([#598](https://github.com/hpc-maths/samurai/issues/598)) ([f2ad811](https://github.com/hpc-maths/samurai/commit/f2ad811e4cbd7275fd1f028cc05a068505a4b54a))
* **bc:** delete Bc assignment and accept a predicate in Bc::on ([#708](https://github.com/hpc-maths/samurai/issues/708)) ([62aea0c](https://github.com/hpc-maths/samurai/commit/62aea0c1a34b82ffa8bbf384a650796cc01bad6e))
* **bc:** fill the outer ghosts beyond the third layer ([#505](https://github.com/hpc-maths/samurai/issues/505)) ([01fff1e](https://github.com/hpc-maths/samurai/commit/01fff1ea144080ca640cafa311cef1082e18ae1e))
* **cmake:** export the include directory of samurai::samurai on first configure ([#715](https://github.com/hpc-maths/samurai/issues/715)) ([ba792b6](https://github.com/hpc-maths/samurai/commit/ba792b644aa3a7eeb1a44f9d7ca649fa0dd42ee2))
* compile the PETSc assemblies and explicit flux-based schemes without warnings ([#753](https://github.com/hpc-maths/samurai/issues/753)) ([fca0cd0](https://github.com/hpc-maths/samurai/commit/fca0cd0057302388d96239dd196c519bec31b89c))
* compile the samurai headers without warnings ([#745](https://github.com/hpc-maths/samurai/issues/745)) ([419b31f](https://github.com/hpc-maths/samurai/commit/419b31fc2f551ec1ff82dbe4e7011ddb1f170a0a))
* **conda:** declare metis in the MPI environment ([#497](https://github.com/hpc-maths/samurai/issues/497)) ([17c52da](https://github.com/hpc-maths/samurai/commit/17c52da5d4dde64e1b943b2f49790dd5251f7817))
* **demos:** fix the time step, levels, subdomain check and outputs of burgers_os_2d_mpi ([#509](https://github.com/hpc-maths/samurai/issues/509)) ([fa278a3](https://github.com/hpc-maths/samurai/commit/fa278a3c178dc87aa300f51edef60da4b344fc59))
* **demos:** make --with-corner add the diagonal directions to the graduation ([#702](https://github.com/hpc-maths/samurai/issues/702)) ([4277990](https://github.com/hpc-maths/samurai/commit/42779900dc8571f83cc4eab43110f4c1479af035))
* **demos:** release the PETSc objects of manual_block_matrix_assembly before finalize ([#717](https://github.com/hpc-maths/samurai/issues/717)) ([a03b59e](https://github.com/hpc-maths/samurai/commit/a03b59e44ddc6bf23344ea78a6b4b635c64bc497))
* **demos:** start the read_mesh.py hint of nagumo and heat at file 0 ([#718](https://github.com/hpc-maths/samurai/issues/718)) ([80b6b70](https://github.com/hpc-maths/samurai/commit/80b6b70d7c8e7e1976a36ec8dc0dfc2f545ec871))
* **demos:** swap fields with samurai::swap so the scheme refreshes ghosts ([#752](https://github.com/hpc-maths/samurai/issues/752)) ([ed707c1](https://github.com/hpc-maths/samurai/commit/ed707c14bde6f109b1915512507a659cb27b7ebe))
* **field:** initialize a field with a Gauss-Legendre quadrature ([#613](https://github.com/hpc-maths/samurai/issues/613)) ([6dbf63c](https://github.com/hpc-maths/samurai/commit/6dbf63c07bb505e13a01272256199623a9476196))
* **fv:** copy the constant velocity and update the Rusanov velocity ghosts ([#711](https://github.com/hpc-maths/samurai/issues/711)) ([09dacb2](https://github.com/hpc-maths/samurai/commit/09dacb2eb6c95363dcbc4ff147431ac537021245))
* **fv:** skip the level jump above the max level of the mesh in non-linear flux schemes ([#709](https://github.com/hpc-maths/samurai/issues/709)) ([6a7e69b](https://github.com/hpc-maths/samurai/commit/6a7e69be1ed7d1b8d336bffea3bb1a0c7fa7d499))
* **ghost:** stop the level loop of update_outer_ghosts(field) at level 0 ([#548](https://github.com/hpc-maths/samurai/issues/548)) ([23dc69a](https://github.com/hpc-maths/samurai/commit/23dc69a87983feb5750e932a50dd3561a6b27f90))
* graduation ([#473](https://github.com/hpc-maths/samurai/issues/473)) ([9b4d4fa](https://github.com/hpc-maths/samurai/commit/9b4d4faef16eca6a556c233ec9a660bf60ba645e))
* **interface:** default include_periodic in the direction overload ([#707](https://github.com/hpc-maths/samurai/issues/707)) ([8b793a3](https://github.com/hpc-maths/samurai/commit/8b793a3d646390a20f14d9e0d70e0c04f7ab089d))
* **io:** include the HighFive xtensor plug-in so restart.hpp compiles on its own ([#726](https://github.com/hpc-maths/samurai/issues/726)) ([539ccde](https://github.com/hpc-maths/samurai/commit/539ccde0f2905f1093220e7041102f33da474fa4))
* **io:** write each mesh vertex once when the cell length is not dyadic ([#742](https://github.com/hpc-maths/samurai/issues/742)) ([4942abc](https://github.com/hpc-maths/samurai/commit/4942abc0dfcace64f00409fba35e653974a09432))
* **lbm:** apply the wall reflection on the domain corners ([#487](https://github.com/hpc-maths/samurai/issues/487)) ([a25639c](https://github.com/hpc-maths/samurai/commit/a25639cd2bb1a06e565866b9536ff7bf79a51690))
* **lbm:** make the LBM time step non-const instead of mutating a mutable buffer ([#760](https://github.com/hpc-maths/samurai/issues/760)) ([eab5842](https://github.com/hpc-maths/samurai/commit/eab5842348e8def88e05adf52fd8169d65fa4dc2))
* **mesh:** approximate boxes at their level when a scaling factor is set ([#713](https://github.com/hpc-maths/samurai/issues/713)) ([0e47c98](https://github.com/hpc-maths/samurai/commit/0e47c98da9c1674a5b183777d41f2a579f60e8cb))
* **mesh:** keep the configuration in the mesh built by make_empty_mesh ([#508](https://github.com/hpc-maths/samurai/issues/508)) ([bf6135c](https://github.com/hpc-maths/samurai/commit/bf6135cbf3d874296e73bc5b762b91184ef70524))
* **mesh:** start an AMR mesh at its max level by default and reject a start level outside the level range ([#551](https://github.com/hpc-maths/samurai/issues/551)) ([c487099](https://github.com/hpc-maths/samurai/commit/c4870993d1c9279bb03b4f360ed5f82b845c2653))
* **mr:** avoid a size_t underflow in the harten tag-cells loop at level 0 ([#498](https://github.com/hpc-maths/samurai/issues/498)) ([b94b83f](https://github.com/hpc-maths/samurai/commit/b94b83f5c8ddbf3ecf0c67ff12d315a43aa32f89))
* **mr:** compile the multiresolution headers without warnings ([#723](https://github.com/hpc-maths/samurai/issues/723)) ([9ae6ebd](https://github.com/hpc-maths/samurai/commit/9ae6ebd2edb54e7953d13ecd3b434aed29c64466))
* **petsc:** assemble ghosts and level jumps at subdomain boundaries with MPI ([#545](https://github.com/hpc-maths/samurai/issues/545)) ([a18d277](https://github.com/hpc-maths/samurai/commit/a18d27747a4c9065d37437f2538bfa34355f96cb))
* **petsc:** let nested block matrices and vectors own their blocks ([#743](https://github.com/hpc-maths/samurai/issues/743)) ([63d5b82](https://github.com/hpc-maths/samurai/commit/63d5b82cb1e7e1a97cc669e128a0460e957dd3b0))
* **petsc:** move everything in solver move assignment and forbid copies ([#738](https://github.com/hpc-maths/samurai/issues/738)) ([e209cdc](https://github.com/hpc-maths/samurai/commit/e209cdc369431a4fea2f2f6801d40a7c94c1f1c9))
* **petsc:** number the columns of a non-square operator after its input field ([#747](https://github.com/hpc-maths/samurai/issues/747)) ([6618af1](https://github.com/hpc-maths/samurai/commit/6618af11b035db18badffb0222c1034e0c98d53f))
* **petsc:** solve non-linear cell-based schemes on scalar fields ([#712](https://github.com/hpc-maths/samurai/issues/712)) ([16ed8c0](https://github.com/hpc-maths/samurai/commit/16ed8c04db56acb4884774fb3bdbd35aac5fbe58))
* **petsc:** use global indices in the fieldsplit index sets of a monolithic block matrix ([#710](https://github.com/hpc-maths/samurai/issues/710)) ([1f59db3](https://github.com/hpc-maths/samurai/commit/1f59db3bbc190f5e3997d62df10b6f8258b7fe67))
* **python:** remove regex backtracking flagged by sonar ([#733](https://github.com/hpc-maths/samurai/issues/733)) ([e195858](https://github.com/hpc-maths/samurai/commit/e195858b1fc026670b45482a3b3c31df34358881))
* **reconstruction:** make the prediction caches safe under OpenMP ([#755](https://github.com/hpc-maths/samurai/issues/755)) ([5628fa5](https://github.com/hpc-maths/samurai/commit/5628fa5ee1e055fe4924e7f2a882b488cb7fba34))
* remove the compiler warnings from tests and demos ([#746](https://github.com/hpc-maths/samurai/issues/746)) ([0937e9c](https://github.com/hpc-maths/samurai/commit/0937e9c229c841f34caa756155e50922f07fd641))
* remove the uncompilable MRMesh::exists and negate the boundary layer width as signed ([#736](https://github.com/hpc-maths/samurai/issues/736)) ([d8ae40c](https://github.com/hpc-maths/samurai/commit/d8ae40ccef58a25dfaed61003d155d81332d7425))
* **restart:** dump fields and check the process count for a UniformMesh ([#716](https://github.com/hpc-maths/samurai/issues/716)) ([612cd79](https://github.com/hpc-maths/samurai/commit/612cd79d53fe1729acc513e865e35e8c7c09041f))
* **restart:** keep the box flag of the domain across dump and load ([#507](https://github.com/hpc-maths/samurai/issues/507)) ([1c0d6a2](https://github.com/hpc-maths/samurai/commit/1c0d6a2e5026b9c4cd361ea05422d43ffe3eafa0))
* **schemes:** compile NonLinear flux-based schemes with an odd stencil size ([#705](https://github.com/hpc-maths/samurai/issues/705)) ([69c8c1c](https://github.com/hpc-maths/samurai/commit/69c8c1c388f35423968803a2820f0cd46ad8a74a))
* **schemes:** fill the ghosts of an AMR mesh with update_ghost in the finite volume operators ([#552](https://github.com/hpc-maths/samurai/issues/552)) ([b1c8c90](https://github.com/hpc-maths/samurai/commit/b1c8c90c7b329aa9166fb08281d77babde068971))
* **schemes:** handle unaligned level-jump intervals in the explicit linear flux scheme ([#542](https://github.com/hpc-maths/samurai/issues/542)) ([8e98b0a](https://github.com/hpc-maths/samurai/commit/8e98b0a77425b1a9408140e54f516d055f6c77e4))
* **schemes:** include update_ghost_mr.hpp where update_ghost_mr_if_needed is called ([#547](https://github.com/hpc-maths/samurai/issues/547)) ([1012400](https://github.com/hpc-maths/samurai/commit/10124005c32fea7621bfc317c2063a8a9aeaaf92))
* **schemes:** move the boundary SIMD loop out of a generic lambda to avoid a GCC ICE ([#754](https://github.com/hpc-maths/samurai/issues/754)) ([9ac83e0](https://github.com/hpc-maths/samurai/commit/9ac83e0648cb50d1d80ef2899775ceb3a8c8bf5b))
* **stats:** repair the mesh statistics recorded with WITH_STATS ([#756](https://github.com/hpc-maths/samurai/issues/756)) ([a0effd6](https://github.com/hpc-maths/samurai/commit/a0effd6dce87c1bc3abc5e32953edfef8e457c8f))
* stop printing to std::cout on normal library calls ([#720](https://github.com/hpc-maths/samurai/issues/720)) ([422e47d](https://github.com/hpc-maths/samurai/commit/422e47d8822ee0ccb940d82ded63c70d4f3ede82))
* **tests:** make test_samurai_lib run under mpi ([#749](https://github.com/hpc-maths/samurai/issues/749)) ([d7fef09](https://github.com/hpc-maths/samurai/commit/d7fef092e524af24f2dd63e833c29d2515997237))
* **tests:** pass --mr-eps to the demo in test_finite_volume_burgers_os ([#719](https://github.com/hpc-maths/samurai/issues/719)) ([25f9e68](https://github.com/hpc-maths/samurai/commit/25f9e68b893fe43585f3268d442463d8c88e4f5a))
* **tests:** update the ghosts before applying the b.c. in bc_ghost_values ([#750](https://github.com/hpc-maths/samurai/issues/750)) ([ed2afbe](https://github.com/hpc-maths/samurai/commit/ed2afbe252b051932f391a29eafb4ec987272cf2))
* **timers:** share one timer registry across translation units ([#714](https://github.com/hpc-maths/samurai/issues/714)) ([8481124](https://github.com/hpc-maths/samurai/commit/84811249d903c36b89f9ce8e3581b2293b0b3454))
* **yt:** give series datasets the times found by discover_series ([#741](https://github.com/hpc-maths/samurai/issues/741)) ([8d723ef](https://github.com/hpc-maths/samurai/commit/8d723ef3b091ce68259ff29f0f69bf22203669c1))


### Documentation

* add CITATION.cff and fix the codemeta.json metadata ([#602](https://github.com/hpc-maths/samurai/issues/602)) ([5981c41](https://github.com/hpc-maths/samurai/commit/5981c41424e0545a88a8dd3e4676dabe05913ee3))
* add generated plates and diagrams ([#649](https://github.com/hpc-maths/samurai/issues/649)) ([450fe11](https://github.com/hpc-maths/samurai/commit/450fe11d249227b37b625ab22976687eeb76a2a5))
* add numbered definition and remark directives ([#648](https://github.com/hpc-maths/samurai/issues/648)) ([70ecba8](https://github.com/hpc-maths/samurai/commit/70ecba8702d11c30d4e396c3e134e21a8563c596))
* add the demos and samurai gallery pages ([#612](https://github.com/hpc-maths/samurai/issues/612)) ([8529184](https://github.com/hpc-maths/samurai/commit/852918468cdf422000d55da540e98003b3a4d8be))
* align the graded-mesh check and the figure data location ([#699](https://github.com/hpc-maths/samurai/issues/699)) ([a22e8d3](https://github.com/hpc-maths/samurai/commit/a22e8d39a7f5a7c03e2ceb1f181c616293ef9ddd))
* **api:** add API pages for meshes, fields, adaptation, I/O, timers, schemes and PETSc ([#600](https://github.com/hpc-maths/samurai/issues/600)) ([f87fc23](https://github.com/hpc-maths/samurai/commit/f87fc234c63a0210a5ec6db1205db11602082c41))
* **api:** add the load balancing, LBM and dynamic set algebra API pages ([#622](https://github.com/hpc-maths/samurai/issues/622)) ([18183f9](https://github.com/hpc-maths/samurai/commit/18183f9d016947975db9bee7dbbdda4c740e238b))
* **api:** fix the Doxygen comments of the load balancing, LBM and dynamic set headers ([#637](https://github.com/hpc-maths/samurai/issues/637)) ([ed57555](https://github.com/hpc-maths/samurai/commit/ed57555156d98ad70ae2f715ebcf41206576fd50))
* **api:** make the Doxygen comments match the code, with 0 Doxygen warnings ([#604](https://github.com/hpc-maths/samurai/issues/604)) ([0e5d110](https://github.com/hpc-maths/samurai/commit/0e5d11058ce171f3719bb927c2da024024cf765a))
* **api:** use doxygenstruct for NormalFluxDefinitionBase ([#634](https://github.com/hpc-maths/samurai/issues/634)) ([3b7e681](https://github.com/hpc-maths/samurai/commit/3b7e681e11528d60f9b834fb34ea7fe675badd96))
* **box:** draw the domains of the snippets and the smallest hole ([#688](https://github.com/hpc-maths/samurai/issues/688)) ([187fe8f](https://github.com/hpc-maths/samurai/commit/187fe8f96f9c99b9b2a61f7801514aa81a4eea78))
* **contributing:** document the figure modules and their rules ([#692](https://github.com/hpc-maths/samurai/issues/692)) ([3218694](https://github.com/hpc-maths/samurai/commit/3218694604758ab49255f0c568e38f1766cbdd42))
* **contributing:** rewrite the guide from a fresh clone to passing tests and a merged pull request ([#621](https://github.com/hpc-maths/samurai/issues/621)) ([c9dc9f0](https://github.com/hpc-maths/samurai/commit/c9dc9f0ca24c51d89390fa793bb3ab500e9865bb))
* **css:** keep command-line options on one line ([#697](https://github.com/hpc-maths/samurai/issues/697)) ([2ec5043](https://github.com/hpc-maths/samurai/commit/2ec5043b31e592f8172eeedbc9301a8e5c797eec))
* **css:** keep short hyphenated inline code on one line ([#701](https://github.com/hpc-maths/samurai/issues/701)) ([c59a087](https://github.com/hpc-maths/samurai/commit/c59a08779b1f6f775eb7b63a29c3cd2403c2dcb5))
* **design:** record the serif heading scale and the rank tones ([#661](https://github.com/hpc-maths/samurai/issues/661)) ([e833e1a](https://github.com/hpc-maths/samurai/commit/e833e1aab7f48e3681fca71d556926d450ff4a23))
* **field:** draw how field values are stored ([#696](https://github.com/hpc-maths/samurai/issues/696)) ([28827ce](https://github.com/hpc-maths/samurai/commit/28827ce5cc0d00f97e4738e4d1b3ad35902b45ea))
* **figures:** add arrays, printed intervals and strided intervals ([#658](https://github.com/hpc-maths/samurai/issues/658)) ([2cb0eb5](https://github.com/hpc-maths/samurai/commit/2cb0eb5bdf8e998f49f8d1bfa42168805a20bd65))
* **figures:** add cycles and chains of stations ([#659](https://github.com/hpc-maths/samurai/issues/659)) ([f392150](https://github.com/hpc-maths/samurai/commit/f392150406fa36185a994a5a0f4dc5f344096609))
* **figures:** add plots of 1D fields and details ([#657](https://github.com/hpc-maths/samurai/issues/657)) ([3795c7e](https://github.com/hpc-maths/samurai/commit/3795c7e3c2980ebdaf8a6dfdd85f16a9a4c0849f))
* **figures:** add rank tones and space-filling curve orders ([#660](https://github.com/hpc-maths/samurai/issues/660)) ([a21def1](https://github.com/hpc-maths/samurai/commit/a21def11aded100410cac32ca71135fdda208eb9))
* **figures:** document the constructor of static_svg.Node ([#666](https://github.com/hpc-maths/samurai/issues/666)) ([7255e98](https://github.com/hpc-maths/samurai/commit/7255e980b652221f8e3ac5d50d37d7db8e128411))
* **figures:** draw meshes from the cells a program exports ([#656](https://github.com/hpc-maths/samurai/issues/656)) ([ac1c278](https://github.com/hpc-maths/samurai/commit/ac1c278f52835cee92deb4e6695c9d9dcf7a1cc6))
* **figures:** give each page its own figure module ([#651](https://github.com/hpc-maths/samurai/issues/651)) ([50a3c91](https://github.com/hpc-maths/samurai/commit/50a3c91d1d53f1bcbffbaa2639fb7d6fd2f55366))
* **figures:** let plates and diagrams take a hand-written svg ([#662](https://github.com/hpc-maths/samurai/issues/662)) ([a5f3007](https://github.com/hpc-maths/samurai/commit/a5f3007298904c6d1ec18457cc2f0a50c97ce7ae))
* **figures:** set flow statements glyph by glyph and inset the loop note panel ([#695](https://github.com/hpc-maths/samurai/issues/695)) ([78da424](https://github.com/hpc-maths/samurai/commit/78da424335104427bb703200cf77bbe38ac8eb0e))
* **github:** rewrite the pull request template as writing guidance ([#549](https://github.com/hpc-maths/samurai/issues/549)) ([ed4a1cb](https://github.com/hpc-maths/samurai/commit/ed4a1cb96ab6f19d7a329f7beec11435abdaba9e))
* **howto:** add a guide to build a domain with obstacles ([#617](https://github.com/hpc-maths/samurai/issues/617)) ([f86d5b4](https://github.com/hpc-maths/samurai/commit/f86d5b46813be69be73e28bd7fd9a00aa4b05be7))
* **howto:** add a guide to run a samurai program with MPI ([#596](https://github.com/hpc-maths/samurai/issues/596)) ([aa7c963](https://github.com/hpc-maths/samurai/commit/aa7c963c3cd36d4ffafb9ef325127c5a601188cb))
* **howto:** add a how-to for adapting a mesh with multiresolution ([#554](https://github.com/hpc-maths/samurai/issues/554)) ([b2a0778](https://github.com/hpc-maths/samurai/commit/b2a07780fcbcd80117ef0ac08166dd069a05a63c))
* **howto:** add a how-to for implicit solves with PETSc ([#615](https://github.com/hpc-maths/samurai/issues/615)) ([94b7579](https://github.com/hpc-maths/samurai/commit/94b75790a47f126be361538bd798960ed1a41797))
* **howto:** add a how-to for restarting a simulation from a checkpoint ([#560](https://github.com/hpc-maths/samurai/issues/560)) ([fd4c04e](https://github.com/hpc-maths/samurai/commit/fd4c04e1d6c43f51c37a3e9255ee4a51da8c1997))
* **howto:** add a troubleshooting page for common error messages ([#614](https://github.com/hpc-maths/samurai/issues/614)) ([0396732](https://github.com/hpc-maths/samurai/commit/039673296a917f65d6b77db65e7b3976ca5efa03))
* **howto:** add the boundary conditions and finite volume scheme how-tos ([#644](https://github.com/hpc-maths/samurai/issues/644)) ([fcd61fb](https://github.com/hpc-maths/samurai/commit/fcd61fb8e5f94bcef4da330afc076ed34c6a3487))
* **howto:** compile every how-to code sample and show real output ([#618](https://github.com/hpc-maths/samurai/issues/618)) ([7d5f851](https://github.com/hpc-maths/samurai/commit/7d5f851ad5ca210c49634c660694551175e66f37))
* **howto:** cover field initialization and samurai CMake options in the how-tos ([#525](https://github.com/hpc-maths/samurai/issues/525)) ([de48ace](https://github.com/hpc-maths/samurai/commit/de48ace1c78d3214a0e2c0be9fa597ef183c65c9))
* **howto:** draw the ghost ring of the uniform mesh ([#690](https://github.com/hpc-maths/samurai/issues/690)) ([4a878da](https://github.com/hpc-maths/samurai/commit/4a878daad6b940eb99e5bdb97f69d4fd880f3e7d))
* **howto:** draw the ghost values of the boundary conditions ([#682](https://github.com/hpc-maths/samurai/issues/682)) ([f15d421](https://github.com/hpc-maths/samurai/commit/f15d4214749526544dd758e8c502965bfad77547))
* **howto:** draw the obstacle mesh of the geometry guide from its program ([#684](https://github.com/hpc-maths/samurai/issues/684)) ([8ceb414](https://github.com/hpc-maths/samurai/commit/8ceb4147e6b03f8d969e3807e69d76efbbd33829))
* **howto:** draw the stencil indices of a wider flux ([#683](https://github.com/hpc-maths/samurai/issues/683)) ([93471a8](https://github.com/hpc-maths/samurai/commit/93471a8719d619a05bb1fe2361bfce245e1e9578))
* **howto:** fix --help in the options how-to and list the predefined options ([#519](https://github.com/hpc-maths/samurai/issues/519)) ([d868570](https://github.com/hpc-maths/samurai/commit/d8685706979c343d80aaa98fa0ac90e05c20ce91))
* **howto:** fix the box how-to for union, holes and mra::make_mesh ([#524](https://github.com/hpc-maths/samurai/issues/524)) ([eafecb9](https://github.com/hpc-maths/samurai/commit/eafecb98a9e48e8229c920cc691b88146e460f17))
* **howto:** fix the installation how-to for conda-forge, MPI builds and build options ([#520](https://github.com/hpc-maths/samurai/issues/520)) ([2afd9e1](https://github.com/hpc-maths/samurai/commit/2afd9e1a4efadce8964e66857273284b69fd3a1b))
* **howto:** fix the loop how-to so the text matches its snippets ([#521](https://github.com/hpc-maths/samurai/issues/521)) ([ee8b78b](https://github.com/hpc-maths/samurai/commit/ee8b78bab3a45bb2a7681253c7b4ffa54612cd84))
* **howto:** fix the mesh how-to for the mesh configuration API ([#517](https://github.com/hpc-maths/samurai/issues/517)) ([be16d61](https://github.com/hpc-maths/samurai/commit/be16d6173e66664a8497a05b2cd591373bd0c510))
* **howto:** fix the plot how-to and list every viewer ([#523](https://github.com/hpc-maths/samurai/issues/523)) ([de8805c](https://github.com/hpc-maths/samurai/commit/de8805c39dd530c2b076a245aba5cba4729457ec))
* **howto:** fix the save how-to and make the dump snippet reload its file ([#518](https://github.com/hpc-maths/samurai/issues/518)) ([fbe1811](https://github.com/hpc-maths/samurai/commit/fbe1811835bbdcd2ab02088f9d0eb1df25225521))
* **howto:** illustrate the multiresolution adaptation ([#673](https://github.com/hpc-maths/samurai/issues/673)) ([ff02d72](https://github.com/hpc-maths/samurai/commit/ff02d720e14b8dc3e7cd66462643bc6f9c64b594))
* **howto:** point the mesh and domain how-tos to the internal pages ([#646](https://github.com/hpc-maths/samurai/issues/646)) ([9fd0b5c](https://github.com/hpc-maths/samurai/commit/9fd0b5c62f4e5a3f676d58c5ba25c1bac69f065d))
* **howto:** rewrite the timers how-to for ScopedTimer, nesting, cell counts and MPI ([#522](https://github.com/hpc-maths/samurai/issues/522)) ([c63a298](https://github.com/hpc-maths/samurai/commit/c63a29887d038bc87438e3b15bb2bbaed376791f))
* **howto:** show the arguments of for_each_interval in the loop how-to ([#689](https://github.com/hpc-maths/samurai/issues/689)) ([be46b25](https://github.com/hpc-maths/samurai/commit/be46b25093dbb5555bae53b1e016254f3321e174))
* **howto:** show the MPI subdomains before and after load balancing ([#685](https://github.com/hpc-maths/samurai/issues/685)) ([5e84393](https://github.com/hpc-maths/samurai/commit/5e843932517dd83b09feb9967e8783cee34814ab))
* **howto:** use spaced separator rows in the option tables ([#528](https://github.com/hpc-maths/samurai/issues/528)) ([b686dd7](https://github.com/hpc-maths/samurai/commit/b686dd7c1f1c5cc764dfa020d23bd6c417e54052))
* **index:** give the landing page its plate ([#671](https://github.com/hpc-maths/samurai/issues/671)) ([53a1376](https://github.com/hpc-maths/samurai/commit/53a1376bec6918a88f4e7083ed2173e21f3cef90))
* introduce and group the section landing pages ([#567](https://github.com/hpc-maths/samurai/issues/567)) ([a1bdf4e](https://github.com/hpc-maths/samurai/commit/a1bdf4e888cdc573078e65f25e771a4610dacbe2))
* **lbm:** draw the lattice Boltzmann scheme figures from code ([#678](https://github.com/hpc-maths/samurai/issues/678)) ([988d321](https://github.com/hpc-maths/samurai/commit/988d321290462d6bb07e3240b76c5ad9828db5dd))
* **lbm:** draw the velocity sets and the setups of the test cases ([#679](https://github.com/hpc-maths/samurai/issues/679)) ([32caea3](https://github.com/hpc-maths/samurai/commit/32caea3227507e9a0c51e01ba1cd5aa382c959d5))
* **lbm:** port the LBM test cases page to MyST and fix its equations ([#561](https://github.com/hpc-maths/samurai/issues/561)) ([995e9ee](https://github.com/hpc-maths/samurai/commit/995e9eeccdd60c81d5c780a8d1820c75217cdcf1))
* **lbm:** walk through a D2Q4 scheme written with the schemes/lbm formalism ([#619](https://github.com/hpc-maths/samurai/issues/619)) ([a7d4010](https://github.com/hpc-maths/samurai/commit/a7d40105b6b367541dc7d2a1c2de6eb2bf17bd26))
* move the code of conduct to Contributor Covenant 2.1 ([#603](https://github.com/hpc-maths/samurai/issues/603)) ([3dc8d9e](https://github.com/hpc-maths/samurai/commit/3dc8d9e0306107a4d2171beb264cca14a4d0890c))
* name the order of a prediction as 2s + 1 everywhere ([#704](https://github.com/hpc-maths/samurai/issues/704)) ([22a09a3](https://github.com/hpc-maths/samurai/commit/22a09a34d14341737a84c78746b7d78aa9f413d0))
* **petsc:** draw the PETSc numbering of the 4 by 4 mesh on two ranks ([#691](https://github.com/hpc-maths/samurai/issues/691)) ([39f3d58](https://github.com/hpc-maths/samurai/commit/39f3d581fb4ef962681e637348b2578aa8961d3a))
* **readme:** port the code samples to the current API and fix stated facts ([#516](https://github.com/hpc-maths/samurai/issues/516)) ([e7d01d9](https://github.com/hpc-maths/samurai/commit/e7d01d994238d2bba5d727c84d2597ce46d1c95e))
* **readme:** turn the README into a front door ([#623](https://github.com/hpc-maths/samurai/issues/623)) ([82b31b5](https://github.com/hpc-maths/samurai/commit/82b31b592d71dbd1cd38a28f16d3d59e28fd5fec))
* **readme:** use make_convection_upwind in the advection example and demo ([#539](https://github.com/hpc-maths/samurai/issues/539)) ([b5a03e1](https://github.com/hpc-maths/samurai/commit/b5a03e152079701fb38754051677b6d2d880f1ff))
* record the product and the visual system of the documentation ([#652](https://github.com/hpc-maths/samurai/issues/652)) ([778388f](https://github.com/hpc-maths/samurai/commit/778388f40efbd7fa213eacacab3368f2ce42f94f))
* **reference:** complete the local schemes reference and port it to MyST ([#563](https://github.com/hpc-maths/samurai/issues/563)) ([f659cb0](https://github.com/hpc-maths/samurai/commit/f659cb0afbd576841ba17d062119741cbba04c3b))
* **reference:** draw the cells the boundary conditions read and write ([#680](https://github.com/hpc-maths/samurai/issues/680)) ([0ba0308](https://github.com/hpc-maths/samurai/commit/0ba0308ecb85b7a41075b2a47c884b18b11d340a))
* **reference:** draw the finite volume faces and stencils from code ([#670](https://github.com/hpc-maths/samurai/issues/670)) ([fed72e5](https://github.com/hpc-maths/samurai/commit/fed72e555fef4bec1db16e5a46c199a547d92677))
* **reference:** draw the outward rounding of a set projection ([#687](https://github.com/hpc-maths/samurai/issues/687)) ([8fd77c6](https://github.com/hpc-maths/samurai/commit/8fd77c61db0fc5550c58211eeba1c61bbf5e57e5))
* **reference:** draw the stencil order of the cell-based schemes ([#686](https://github.com/hpc-maths/samurai/issues/686)) ([b5d5352](https://github.com/hpc-maths/samurai/commit/b5d53520e94cda16e517736b2e8dae3d6dd2092d))
* **reference:** illustrate the load balancing strategies ([#681](https://github.com/hpc-maths/samurai/issues/681)) ([5f55e87](https://github.com/hpc-maths/samurai/commit/5f55e87769b0e162b41f279c06e297d3b23fd3c3))
* **reference:** keep the origin labels of the finite volume stencils legible ([#698](https://github.com/hpc-maths/samurai/issues/698)) ([c2a902a](https://github.com/hpc-maths/samurai/commit/c2a902aa43f12950e2cb7979b667ac8ae6969ac4))
* **reference:** port the boundary conditions reference to MyST and the current API ([#556](https://github.com/hpc-maths/samurai/issues/556)) ([96ed6d5](https://github.com/hpc-maths/samurai/commit/96ed6d545b0dc5481856f2df72f1fb46db54f73d))
* **reference:** port the finite volume schemes reference to MyST and fix its code ([#557](https://github.com/hpc-maths/samurai/issues/557)) ([23ffd3d](https://github.com/hpc-maths/samurai/commit/23ffd3dd5b90ed806430b4c683ff2c299627cd66))
* **reference:** port the load balancing reference to MyST and anchor it to the code ([#565](https://github.com/hpc-maths/samurai/issues/565)) ([c3f3d6d](https://github.com/hpc-maths/samurai/commit/c3f3d6df3e57cb4f8bbdbb77ff2d45c152141222))
* **reference:** print the dynamic set algebra snippet intervals as [a, b) ([#564](https://github.com/hpc-maths/samurai/issues/564)) ([95d78c3](https://github.com/hpc-maths/samurai/commit/95d78c3a8bd61a27da7004f73295e86dddf95da7))
* **reference:** recommend -snes_mf, not -snes_fd, for schemes without a Jacobian ([#633](https://github.com/hpc-maths/samurai/issues/633)) ([608e0cd](https://github.com/hpc-maths/samurai/commit/608e0cd9f719142a1500156839e3d9c653e7a312))
* **reference:** remove figure files no page uses ([#653](https://github.com/hpc-maths/samurai/issues/653)) ([a0e7fee](https://github.com/hpc-maths/samurai/commit/a0e7feefaffc5e624bf50d6b44459e65d397b8c7))
* **reference:** rewrite the set algebra reference for the current API and port it to MyST ([#558](https://github.com/hpc-maths/samurai/issues/558)) ([8466406](https://github.com/hpc-maths/samurai/commit/8466406b9fa70d04d76e1373ad7798a318b4f541))
* **reference:** say where --finer-level-flux applies in implicit solves ([#706](https://github.com/hpc-maths/samurai/issues/706)) ([945202e](https://github.com/hpc-maths/samurai/commit/945202e83bba1884c1ea024f6c331367bb599c7b))
* rewrite the landing page and keep the header on one row ([#566](https://github.com/hpc-maths/samurai/issues/566)) ([8034430](https://github.com/hpc-maths/samurai/commit/80344308401a6b7d07424f0587029299135599a9))
* **schemes:** fix the enlarged stencil example of FluxDefinition ([#643](https://github.com/hpc-maths/samurai/issues/643)) ([5f1c951](https://github.com/hpc-maths/samurai/commit/5f1c951e39ced7533f9092f0f9a3c98ae1becdb1))
* **theme:** even out spaced capitals and fit tables to the column ([#654](https://github.com/hpc-maths/samurai/issues/654)) ([7367b84](https://github.com/hpc-maths/samurai/commit/7367b84af27cae94d118a455293f91a9ae4dea89))
* **theme:** make section headings stand out from the text ([#655](https://github.com/hpc-maths/samurai/issues/655)) ([f7ac641](https://github.com/hpc-maths/samurai/commit/f7ac641e062dd2da2550c884978a98a5702fa8b6))
* **theme:** switch the documentation to Furo with a book-like stylesheet ([#645](https://github.com/hpc-maths/samurai/issues/645)) ([73778be](https://github.com/hpc-maths/samurai/commit/73778be46bc23feff7bb0a813469526f5226e936))
* **tutorial:** adapt the mesh at each time step in the Burgers tutorial and compile its program ([#540](https://github.com/hpc-maths/samurai/issues/540)) ([7172b7a](https://github.com/hpc-maths/samurai/commit/7172b7aec0159dc6db8c2d55e99395f8170de152))
* **tutorial:** add a getting started tutorial ([#536](https://github.com/hpc-maths/samurai/issues/536)) ([d53fa52](https://github.com/hpc-maths/samurai/commit/d53fa5230637bda7c5563b318789f943c92dbaa9))
* **tutorial:** add a page presenting the Hands-on samurai course ([#533](https://github.com/hpc-maths/samurai/issues/533)) ([cd005c4](https://github.com/hpc-maths/samurai/commit/cd005c45efe95c66ffcbf434022db2688f4bb452))
* **tutorial:** build the interval tutorial snippets and regenerate their outputs ([#620](https://github.com/hpc-maths/samurai/issues/620)) ([9ecbebe](https://github.com/hpc-maths/samurai/commit/9ecbebe927a00a4a09255b81049eaef40af7b84a))
* **tutorial:** draw the graduation case 1 meshes from the demo ([#667](https://github.com/hpc-maths/samurai/issues/667)) ([e9e8462](https://github.com/hpc-maths/samurai/commit/e9e8462609455743bd38330ef6e3d95c9ada6fec))
* **tutorial:** draw the graduation case 3 meshes and the even/odd split ([#672](https://github.com/hpc-maths/samurai/issues/672)) ([1afe7c3](https://github.com/hpc-maths/samurai/commit/1afe7c35bc9e47b88c3a84372853056ac0d157aa))
* **tutorial:** draw the interval figures from code ([#664](https://github.com/hpc-maths/samurai/issues/664)) ([59faf7d](https://github.com/hpc-maths/samurai/commit/59faf7d6a9fe0d0560ebea6df387a51232263229))
* **tutorial:** draw the shifted interval of the upwind scheme ([#677](https://github.com/hpc-maths/samurai/issues/677)) ([a3660c8](https://github.com/hpc-maths/samurai/commit/a3660c8aec43a5ddf154d1e0680c1a710a4cbe46))
* **tutorial:** draw the stencils of the level set schemes ([#675](https://github.com/hpc-maths/samurai/issues/675)) ([234a6b4](https://github.com/hpc-maths/samurai/commit/234a6b4e153c5469f20a3ac2d7ca8a97198db0b2))
* **tutorial:** draw the subset figures of the operator on subset tutorial ([#663](https://github.com/hpc-maths/samurai/issues/663)) ([ab456d5](https://github.com/hpc-maths/samurai/commit/ab456d52e4b4d688bdb63b4f80ed79a2384471f3))
* **tutorial:** illustrate the 1D Burgers AMR tutorial ([#676](https://github.com/hpc-maths/samurai/issues/676)) ([89177df](https://github.com/hpc-maths/samurai/commit/89177df02c3db24506fa95d2789ef9fa7eb1254d))
* **tutorial:** include the demo listings by marker instead of line number ([#647](https://github.com/hpc-maths/samurai/issues/647)) ([6cf579c](https://github.com/hpc-maths/samurai/commit/6cf579c3594b4d44871b0ea0e4bdde6f10021fc0))
* **tutorial:** keep the disc outline inside the finest cells ([#700](https://github.com/hpc-maths/samurai/issues/700)) ([668b094](https://github.com/hpc-maths/samurai/commit/668b094225deb34962442fb27e4b19fbc44a2768))
* **tutorial:** port the Burgers tutorial to MyST and fix its math ([#538](https://github.com/hpc-maths/samurai/issues/538)) ([e090767](https://github.com/hpc-maths/samurai/commit/e090767a04da83f1db98a2a33c5748611494fe47))
* **tutorial:** port the field tutorial to MyST and the current API ([#529](https://github.com/hpc-maths/samurai/issues/529)) ([9261830](https://github.com/hpc-maths/samurai/commit/9261830ef2bc30370d1daaf31d973cc4fa49f12d))
* **tutorial:** port the graduation tutorials to MyST and the current demos ([#532](https://github.com/hpc-maths/samurai/issues/532)) ([f3a8b75](https://github.com/hpc-maths/samurai/commit/f3a8b75c7d6b4ee03a515fbc767ede6249a5493f))
* **tutorial:** port the interval tutorial to MyST and fix its 2D example ([#537](https://github.com/hpc-maths/samurai/issues/537)) ([d42cab5](https://github.com/hpc-maths/samurai/commit/d42cab53bfbd0d5f2450a107115f9fd4b1d6a9a8))
* **tutorial:** port the level-set tutorial to MyST and the current demo ([#534](https://github.com/hpc-maths/samurai/issues/534)) ([d632331](https://github.com/hpc-maths/samurai/commit/d63233179216ee187f019d6ddbdebe479f9ed6dd))
* **tutorial:** port the loop tutorial to MyST and the current API ([#530](https://github.com/hpc-maths/samurai/issues/530)) ([01ff659](https://github.com/hpc-maths/samurai/commit/01ff659aed748525fe2ac63a432c2cdd2f1876c3))
* **tutorial:** port the reaction-diffusion tutorial to MyST and the current API ([#535](https://github.com/hpc-maths/samurai/issues/535)) ([d139f88](https://github.com/hpc-maths/samurai/commit/d139f8856ea99743f3cdb71ae9f65fc8271d024f))
* **tutorial:** port the set operator tutorial to MyST and the current set algebra ([#531](https://github.com/hpc-maths/samurai/issues/531)) ([91d20c6](https://github.com/hpc-maths/samurai/commit/91d20c6beaa01f32088a2cbeee94c6cb2c7bc794))
* **tutorial:** redraw the graduation case 2 meshes from the demo ([#668](https://github.com/hpc-maths/samurai/issues/668)) ([40a38a8](https://github.com/hpc-maths/samurai/commit/40a38a8b38987b362f7628b780f352c7bef98c24))
* **tutorial:** show the detail in the getting started tutorial ([#665](https://github.com/hpc-maths/samurai/issues/665)) ([e59d0df](https://github.com/hpc-maths/samurai/commit/e59d0dfd2209ec29bc5082b0990385d0f226d52f))
* **tutorial:** show the front of the reaction-diffusion tutorial ([#693](https://github.com/hpc-maths/samurai/issues/693)) ([6647aa6](https://github.com/hpc-maths/samurai/commit/6647aa6a065d2366131c8cd45ac895ad0eb41e7b))
* **tutorial:** show what a graded mesh allows ([#674](https://github.com/hpc-maths/samurai/issues/674)) ([774acda](https://github.com/hpc-maths/samurai/commit/774acda525fa02f89ec7518eac2b09074cf05a74))
* write the philosophy page and port it to MyST ([#562](https://github.com/hpc-maths/samurai/issues/562)) ([712aa46](https://github.com/hpc-maths/samurai/commit/712aa4620613129975ce8b53a96b6e525860c3b3))


### Code Refactoring

* **fv:** share the boundary equations and the two-point flux of the diffusion schemes ([#762](https://github.com/hpc-maths/samurai/issues/762)) ([9bd19fd](https://github.com/hpc-maths/samurai/commit/9bd19fdee99b9cf5fff81e0e3bc26f6b670f5dde))
* **io:** share the hdf5 writer code between mesh types and builds ([#761](https://github.com/hpc-maths/samurai/issues/761)) ([359d258](https://github.com/hpc-maths/samurai/commit/359d258413b412a622b8c74d63f461db2d592b98))
* **mr:** remove dead code around detail computation and refine_ghost ([#488](https://github.com/hpc-maths/samurai/issues/488)) ([5bbb995](https://github.com/hpc-maths/samurai/commit/5bbb9953b2ca70b201a0d518fcf50bc47c741820))
* remove legacy flux_correction from FV demos ([#479](https://github.com/hpc-maths/samurai/issues/479)) ([09f17bb](https://github.com/hpc-maths/samurai/commit/09f17bb3123761d4403974828280c5b5b901880e))
* route the four prediction consumers through the shared coefficients ([#491](https://github.com/hpc-maths/samurai/issues/491)) ([4cbe7c3](https://github.com/hpc-maths/samurai/commit/4cbe7c3b1ca727089cc5c0956e4623e0385be2dd))
* **schemes:** give the schemes and assemblies real move operations ([#739](https://github.com/hpc-maths/samurai/issues/739)) ([4fb4d1c](https://github.com/hpc-maths/samurai/commit/4fb4d1cec6dc12497bdbbe0aeea35ef4a2b10e17))
* share cells_changed_by_tag and ghosts_below_cells, fix interval header warnings ([#727](https://github.com/hpc-maths/samurai/issues/727)) ([3256038](https://github.com/hpc-maths/samurai/commit/3256038f5997cffde373759bf1f9f8700e310b8b))
* **tools:** share file helpers between the paraview and yt readers ([#740](https://github.com/hpc-maths/samurai/issues/740)) ([c68e9d0](https://github.com/hpc-maths/samurai/commit/c68e9d0d10b5c6a5945286c781f2a5c3f1eb6375))


### Performance

* **lbm:** faster multi-level stream (slice reconstruction, hoisted scalar-loop stencils) ([#484](https://github.com/hpc-maths/samurai/issues/484)) ([732a40c](https://github.com/hpc-maths/samurai/commit/732a40c71f69e03e1b8a0b1f8d437855014b5af2))
* **mesh:** detect the adaptation fixed point before constructing the new mesh ([#476](https://github.com/hpc-maths/samurai/issues/476)) ([76832e2](https://github.com/hpc-maths/samurai/commit/76832e291be75a13971ed9683072219ba6992ac3))
* **mr:** faster multiresolution detail computation ([#481](https://github.com/hpc-maths/samurai/issues/481)) ([3051810](https://github.com/hpc-maths/samurai/commit/3051810d815977dc1a6bf0ae7e19b9a4eda3f522))
* speed up the multiresolution mesh-update phase ([#482](https://github.com/hpc-maths/samurai/issues/482)) ([2b8d4db](https://github.com/hpc-maths/samurai/commit/2b8d4db9a855bedbebe7f435eea0680acf098917))
* subset traverser ([#474](https://github.com/hpc-maths/samurai/issues/474)) ([599aa5d](https://github.com/hpc-maths/samurai/commit/599aa5d1f646bdd02373d633e880cf96acf5b6a9))


### Tests

* **bc:** assert boundary reproduction order in the cell-average convention ([#489](https://github.com/hpc-maths/samurai/issues/489)) ([cb67b73](https://github.com/hpc-maths/samurai/commit/cb67b7382a7d860f32daccd0d02d1640a9a624fd))
* **benchmark:** check every backend against reference checksums and plot the sweep ([#730](https://github.com/hpc-maths/samurai/issues/730)) ([4050ed6](https://github.com/hpc-maths/samurai/commit/4050ed62d68355548d983189614adc1dd1a1147d))
* **benchmark:** measure the set algebra on synthetic sets and on the call sites of the library ([#721](https://github.com/hpc-maths/samurai/issues/721)) ([aef3c38](https://github.com/hpc-maths/samurai/commit/aef3c38047a29cd08f44219055e98887ecdbaae7))
* **benchmark:** run the set algebra benchmarks over the level, on more geometries and on partitions ([#729](https://github.com/hpc-maths/samurai/issues/729)) ([9fd14f7](https://github.com/hpc-maths/samurai/commit/9fd14f7d7b932001595ff6dae0fe5b4df85675aa))
* **benchmark:** time the call sites that build level cell arrays and the ways to consume a set ([#728](https://github.com/hpc-maths/samurai/issues/728)) ([334ff56](https://github.com/hpc-maths/samurai/commit/334ff56b036e42e726780f16eb598881a79169f3))
* **demos:** sort the file lists before comparing them in the h5diff check ([#553](https://github.com/hpc-maths/samurai/issues/553)) ([c8be8a3](https://github.com/hpc-maths/samurai/commit/c8be8a34c042039efe97864f99d532fe4b8a5f80))
* ghost update parallel tests ([#475](https://github.com/hpc-maths/samurai/issues/475)) ([6177b86](https://github.com/hpc-maths/samurai/commit/6177b86b12d347f0c427162060e710d88b58f38d))
* **graduation:** give the width tests' tag ownership of its mesh ([#499](https://github.com/hpc-maths/samurai/issues/499)) ([7fa7237](https://github.com/hpc-maths/samurai/commit/7fa7237e0b4f24df96487846bfde240404340ac0))
* **restart:** edit the checkpoint on one process only in restart_file_without_is_box ([#725](https://github.com/hpc-maths/samurai/issues/725)) ([493e392](https://github.com/hpc-maths/samurai/commit/493e392b9b01793d2341dd14cef1472460d69dd7))


### Chores

* add a markdownlint configuration matching the docs conventions ([#527](https://github.com/hpc-maths/samurai/issues/527)) ([fc04700](https://github.com/hpc-maths/samurai/commit/fc0470084cc2c5dfd454cbdd995de108d5ddeadd))
* **demos:** remove the legacy lattice boltzmann demos ([#748](https://github.com/hpc-maths/samurai/issues/748)) ([7f86c06](https://github.com/hpc-maths/samurai/commit/7f86c065ba27dd790f2963fb3c8e399a143f4aa5))
* **demos:** remove the weno and bz demos that no target builds ([#737](https://github.com/hpc-maths/samurai/issues/737)) ([6d26ca4](https://github.com/hpc-maths/samurai/commit/6d26ca4f83bacba44d178d2fde33715ddee7daf5))
* **github:** ask for what a bug needs to be reproduced in the issue forms ([#601](https://github.com/hpc-maths/samurai/issues/601)) ([9a39f95](https://github.com/hpc-maths/samurai/commit/9a39f9598290d35cc16abfdfb3049b1292301da1))
* remove Codacy leftovers ([#734](https://github.com/hpc-maths/samurai/issues/734)) ([d324bfe](https://github.com/hpc-maths/samurai/commit/d324bfe66d0b52b65eb46830722bee86b280ecc0))


### CI/CD

* **deps:** bump the github-actions group across 1 directory with 7 updates ([#751](https://github.com/hpc-maths/samurai/issues/751)) ([7e404bc](https://github.com/hpc-maths/samurai/commit/7e404bc86b6831e6bd2af7a4398f1f9288f113b7))
* **docs:** fail the docs build on warnings and check links weekly ([#636](https://github.com/hpc-maths/samurai/issues/636)) ([af9eb12](https://github.com/hpc-maths/samurai/commit/af9eb126b4ac165339201177257a6ea10ed5c3e2))
* give Codacy a Prospector profile without the D213 docstring rule ([#650](https://github.com/hpc-maths/samurai/issues/650)) ([c857042](https://github.com/hpc-maths/samurai/commit/c8570428da9ae519d710a5073c182907e1cc94e6))
* pin third-party actions to commit SHAs and let dependabot bump them ([#731](https://github.com/hpc-maths/samurai/issues/731)) ([d510b65](https://github.com/hpc-maths/samurai/commit/d510b65217330fb597a71754e118eb066157a169))
* **pre-commit:** run markdownlint-cli2 on the Markdown files ([#694](https://github.com/hpc-maths/samurai/issues/694)) ([8620ef4](https://github.com/hpc-maths/samurai/commit/8620ef43aed3a41ac71a6e86fb99adccabad909b))
* **release:** read release-please-config.json in the release workflow ([#606](https://github.com/hpc-maths/samurai/issues/606)) ([4cc4d5b](https://github.com/hpc-maths/samurai/commit/4cc4d5b658ef8ee66134080bfa8660086037e90a))
* skip build and test jobs when a pull request only changes documentation ([#526](https://github.com/hpc-maths/samurai/issues/526)) ([56ed802](https://github.com/hpc-maths/samurai/commit/56ed80259683ff1cb8ec9f720195a45162ed00a3))
* **snippets:** build the snippets with PETSc ([#635](https://github.com/hpc-maths/samurai/issues/635)) ([5889202](https://github.com/hpc-maths/samurai/commit/58892023b57e94e3848c05a29986c82a56ebd97a))


### Build System

* **docs:** accept only MyST sources and let make html run without Doxygen ([#595](https://github.com/hpc-maths/samurai/issues/595)) ([1894c71](https://github.com/hpc-maths/samurai/commit/1894c71dde95299d2192811ed2f523df4911b2c2))
* **docs:** configure Sphinx for MyST pages and local Doxygen runs ([#515](https://github.com/hpc-maths/samurai/issues/515)) ([fc52506](https://github.com/hpc-maths/samurai/commit/fc5250666d2ab42a257a932c40ad227c234eb308))
* **docs:** generate the snippet outputs and check them in CI ([#638](https://github.com/hpc-maths/samurai/issues/638)) ([2dd8e0e](https://github.com/hpc-maths/samurai/commit/2dd8e0ee0d1de46193cad2c9a7c259dc6d87b15a))

## [0.33.0](https://github.com/hpc-maths/samurai/compare/v0.32.0...v0.33.0) (2026-07-20)


### Features

* Add timers for AMR: update field ([#470](https://github.com/hpc-maths/samurai/issues/470)) ([fc03234](https://github.com/hpc-maths/samurai/commit/fc0323467891cf9abeec7d9ce59bc1d774fb33ac))


### Performance Improvements

* improve interval insertion for list of intervals ([#471](https://github.com/hpc-maths/samurai/issues/471)) ([5c86fe6](https://github.com/hpc-maths/samurai/commit/5c86fe6fbbba4e3a1a7ba67d03693f8d8bacf6b2))

## [0.32.0](https://github.com/hpc-maths/samurai/compare/v0.31.2...v0.32.0) (2026-07-13)


### Features

* get samurai version and its dependencies ([#458](https://github.com/hpc-maths/samurai/issues/458)) ([bda9784](https://github.com/hpc-maths/samurai/commit/bda9784930d67377ff48c96fcfef9d13d2698eee))
* hdf5/xdmf metadata provided by the user ([#459](https://github.com/hpc-maths/samurai/issues/459)) ([50b8837](https://github.com/hpc-maths/samurai/commit/50b8837575f0f0a0fd460cb44174102fe84ccc62))
* paraview reader ([#460](https://github.com/hpc-maths/samurai/issues/460)) ([83d6cc1](https://github.com/hpc-maths/samurai/commit/83d6cc1f2c9c759b38ee83d60bcb1c709d622baa))
* yt reader ([#461](https://github.com/hpc-maths/samurai/issues/461)) ([0e02fe3](https://github.com/hpc-maths/samurai/commit/0e02fe384fca3b5500c44ee9e4e4ff3783db1b9c))

## [0.31.2](https://github.com/hpc-maths/samurai/compare/v0.31.1...v0.31.2) (2026-07-09)


### Bug Fixes

* update_fields ambiguity ([#456](https://github.com/hpc-maths/samurai/issues/456)) ([1c2bdc3](https://github.com/hpc-maths/samurai/commit/1c2bdc3ebea7d1ec4d238da90e954249fe577168))

## [0.31.1](https://github.com/hpc-maths/samurai/compare/v0.31.0...v0.31.1) (2026-07-09)


### Bug Fixes

* acknowledge windows build fix for release ([7067279](https://github.com/hpc-maths/samurai/commit/70672797684ec0a6633853890fb15cb6e4e62d39))

## [0.31.0](https://github.com/hpc-maths/samurai/compare/v0.30.1...v0.31.0) (2026-07-09)


### Features

* Instrument AMR main functions ([#446](https://github.com/hpc-maths/samurai/issues/446)) ([1cbf142](https://github.com/hpc-maths/samurai/commit/1cbf142ec7e839d10094db18a509eafeb48da101))
* new load balancing ([#443](https://github.com/hpc-maths/samurai/issues/443)) ([b1e9142](https://github.com/hpc-maths/samurai/commit/b1e914268eda9b2122427404216e3513a138777a))
* update compare.py script ([#447](https://github.com/hpc-maths/samurai/issues/447)) ([68cf509](https://github.com/hpc-maths/samurai/commit/68cf509e2ca7115e46745495f06cd2a4f9de6b4b))


### Bug Fixes

* upper bounds of the loop in the graduation when the mesh is empty ([#444](https://github.com/hpc-maths/samurai/issues/444)) ([b46f293](https://github.com/hpc-maths/samurai/commit/b46f293a903fdc8c7b8f83d84d0faab867b711f9))


### Performance Improvements

* graduation ([#449](https://github.com/hpc-maths/samurai/issues/449)) ([e301b8a](https://github.com/hpc-maths/samurai/commit/e301b8a0d5b15da3e372282a419f6e404ead8175))
* Improve update ghost mr for mpi ([#445](https://github.com/hpc-maths/samurai/issues/445)) ([a999b56](https://github.com/hpc-maths/samurai/commit/a999b563f1ffe0bd9b2d47f8c892af093ca419d6))
* update fields refactor ([#451](https://github.com/hpc-maths/samurai/issues/451)) ([10cf55d](https://github.com/hpc-maths/samurai/commit/10cf55d510d22c784fb5b6ac79a41ce0a34829ed))

## [0.30.1](https://github.com/hpc-maths/samurai/compare/v0.30.0...v0.30.1) (2026-06-11)


### Bug Fixes

* boundary conditions for 3d case ([#441](https://github.com/hpc-maths/samurai/issues/441)) ([6cf3064](https://github.com/hpc-maths/samurai/commit/6cf306421ee353a67612d5f975f38aca29ba30ef))

## [0.30.0](https://github.com/hpc-maths/samurai/compare/v0.29.2...v0.30.0) (2026-05-04)


### Features

* ND projection and prediction operator ([#435](https://github.com/hpc-maths/samurai/issues/435)) ([55829ab](https://github.com/hpc-maths/samurai/commit/55829ab14ab491b75deb9260e8b57fe5d5e5986e))


### Bug Fixes

* mesh_config in load ([#438](https://github.com/hpc-maths/samurai/issues/438)) ([7f43222](https://github.com/hpc-maths/samurai/commit/7f432227c5e39f0a47b5d241d2a8d4a77a6b4a28))
* set tag type to `uint8_t` ([#437](https://github.com/hpc-maths/samurai/issues/437)) ([44e2ce7](https://github.com/hpc-maths/samurai/commit/44e2ce777b9fe326d8f6e807220d4ba44fc180ea))


### Performance Improvements

* improve tag computation ([#436](https://github.com/hpc-maths/samurai/issues/436)) ([b0aadc5](https://github.com/hpc-maths/samurai/commit/b0aadc53fcc791196b5b5bccbc93c5a5e2b570c7))
* optimizations of local non-linear solvers ([#439](https://github.com/hpc-maths/samurai/issues/439)) ([f0eedea](https://github.com/hpc-maths/samurai/commit/f0eedeaa89b5e8adb84c6e079c12c4869f5a2000))

## [0.29.2](https://github.com/hpc-maths/samurai/compare/v0.29.1...v0.29.2) (2026-03-11)


### Bug Fixes

* compilation with clang-19  ([#433](https://github.com/hpc-maths/samurai/issues/433)) ([f9d28fb](https://github.com/hpc-maths/samurai/commit/f9d28fb1073dfb0a49f79ff9dcaca5ad2d3a0ea5))

## [0.29.1](https://github.com/hpc-maths/samurai/compare/v0.29.0...v0.29.1) (2026-03-06)


### Bug Fixes

* load balance call ([#431](https://github.com/hpc-maths/samurai/issues/431)) ([f2c44e4](https://github.com/hpc-maths/samurai/commit/f2c44e4fc34329cad348a9d49d64c4cd9b308807))
* mesh config ([#430](https://github.com/hpc-maths/samurai/issues/430)) ([cc21127](https://github.com/hpc-maths/samurai/commit/cc2112743be5478daffe0ec0e2a492eceac77e32))

## [0.29.0](https://github.com/hpc-maths/samurai/compare/v0.28.0...v0.29.0) (2026-03-04)


### Features

* 2D load balancing ([#424](https://github.com/hpc-maths/samurai/issues/424)) ([87cc3f5](https://github.com/hpc-maths/samurai/commit/87cc3f5c2b3d924b30985aac05fbb533636bf66b))
* add SAMURAI_INLINE ([#413](https://github.com/hpc-maths/samurai/issues/413)) ([eee5539](https://github.com/hpc-maths/samurai/commit/eee5539e46622364c765772147a648fb49dacbcd))
* Adding a `find_cell` that takes a `cell_t::indices_t` as an argument ([#423](https://github.com/hpc-maths/samurai/issues/423)) ([38f985d](https://github.com/hpc-maths/samurai/commit/38f985d5528495d8d8d3be77404db75744bc99f9))
* find neighbours ([#422](https://github.com/hpc-maths/samurai/issues/422)) ([2f4c9ca](https://github.com/hpc-maths/samurai/commit/2f4c9ca9eaf13e9d37f510372a2d699decd70f2d))
* Hdf5 comm self ([#421](https://github.com/hpc-maths/samurai/issues/421)) ([237b602](https://github.com/hpc-maths/samurai/commit/237b6021120189dcf1f23b4721bf26864cfb3739))
* Improve timers ([#425](https://github.com/hpc-maths/samurai/issues/425)) ([eac3085](https://github.com/hpc-maths/samurai/commit/eac3085eacfdc3791539eb5a6a72d53ccbf4a73a))
* Non-linear block solver ([#410](https://github.com/hpc-maths/samurai/issues/410)) ([76713c1](https://github.com/hpc-maths/samurai/commit/76713c10ebcb3cd7a91f0455eda74afd64fedd8d))
* remove qs template and fix prediction sign ([#428](https://github.com/hpc-maths/samurai/issues/428)) ([cadbfb4](https://github.com/hpc-maths/samurai/commit/cadbfb4d92f742812f3f91e4798ae2db2139e56e))
* set config file ([#412](https://github.com/hpc-maths/samurai/issues/412)) ([9119e67](https://github.com/hpc-maths/samurai/commit/9119e67f6d99c360927afe4a20332d93dbb02e9e))


### Bug Fixes

* mpi and periodic ([#420](https://github.com/hpc-maths/samurai/issues/420)) ([bc06408](https://github.com/hpc-maths/samurai/commit/bc0640835faf29ed9915353d94b331c84ec30cf9))
* simplify expression during detail computation ([#426](https://github.com/hpc-maths/samurai/issues/426)) ([9f01e6b](https://github.com/hpc-maths/samurai/commit/9f01e6beeec5adfc02aab827a78a83a5163f80bf))

## [0.28.0](https://github.com/hpc-maths/samurai/compare/v0.27.1...v0.28.0) (2026-01-22)


### Features

* add dynamic configuration for mesh ([#374](https://github.com/hpc-maths/samurai/issues/374)) ([68f9c25](https://github.com/hpc-maths/samurai/commit/68f9c25e07a7d37b86b00b991dd9e249a3ae7c31))
* PETSc with MPI ([#399](https://github.com/hpc-maths/samurai/issues/399)) ([ec08407](https://github.com/hpc-maths/samurai/commit/ec084071e2277e81c02b9d1406bc62936f08e5e3))


### Bug Fixes

* add include ([#323](https://github.com/hpc-maths/samurai/issues/323)) ([c749583](https://github.com/hpc-maths/samurai/commit/c7495833ecf8ae64fe3cba1ab2d749649518e44b))
* Fix '+' operator between linear and non-linear local shemes with scalar field ([#404](https://github.com/hpc-maths/samurai/issues/404)) ([f1792ad](https://github.com/hpc-maths/samurai/commit/f1792ad1acff2bcab2b3d994476202dd560002da))
* fix collapsed stencil ([#408](https://github.com/hpc-maths/samurai/issues/408)) ([989a212](https://github.com/hpc-maths/samurai/commit/989a2121f8eea5eab62b103956976c7315100631))
* fix macro SAMURAI_PARSE ([#406](https://github.com/hpc-maths/samurai/issues/406)) ([b39857a](https://github.com/hpc-maths/samurai/commit/b39857a2780426f96562cfd8d0a4e31c929fc681))
* Fix signatures ([#398](https://github.com/hpc-maths/samurai/issues/398)) ([2c91ab3](https://github.com/hpc-maths/samurai/commit/2c91ab34a213c71b6be995f6b0f51333c5f1c133))
* flag `ghosts_updated` in AMR ([#407](https://github.com/hpc-maths/samurai/issues/407)) ([c048662](https://github.com/hpc-maths/samurai/commit/c048662784dc2acb4b8a39afc60aa710e1db9a06))

## [0.27.1](https://github.com/hpc-maths/samurai/compare/v0.27.0...v0.27.1) (2025-11-24)


### Bug Fixes

* Use average velocity in upwind (and fix compilation issues) ([#396](https://github.com/hpc-maths/samurai/issues/396)) ([8771dc6](https://github.com/hpc-maths/samurai/commit/8771dc6d775114f54167f7c76f2a19ed453bb4fa))

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
