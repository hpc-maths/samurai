# How-to: set options in samurai

samurai reads command-line options with the [CLI11](https://github.com/CLIUtils/CLI11) library. `samurai::initialize` registers a set of predefined options (mesh levels, multiresolution threshold, timers, ...). This guide shows how to parse them, how to add your own options, and how to read option values from a TOML file.

Before you start, you need a program that builds against samurai. If you don't have one yet, follow the [installation guide](installation.md) and the [CMake guide](cmake.md).

## Parse the command line

1. Call `samurai::initialize(description, argc, argv)` at the start of `main`. It registers the predefined options and reads their values from the command line. The description is the first line of the `--help` output.
2. If you have custom options, add them now (see {ref}`howto-options-custom`).
3. Call `SAMURAI_PARSE(argc, argv)`. It parses the command line again with every option, custom ones included, and handles `--help`.
4. Call `samurai::finalize()` before `main` returns.

The program below follows these steps and builds a multiresolution mesh with levels 2 to 5:

```{literalinclude} snippet/options/predefined_options.cpp
  :language: c++
```

`samurai::initialize` turns `--help` off while it reads the predefined options, so `--help` only works once `SAMURAI_PARSE` runs. If you leave out `SAMURAI_PARSE`, `--help` prints nothing and the program runs as if the flag were not there.

`SAMURAI_PARSE` is a macro that contains a `return` statement: on `--help` or on a parse error, it prints a message and returns from the function that calls it. Call it in `main`.

## Check the result

Run the program with `--help` (or `-h`):

```bash
./predefined_options --help
```

The output shows the description, the usage line, then the options by group. For a build without MPI, it starts like this:

```{literalinclude} snippet/options/predefined_options_help_output.txt
  :language: text
```

The program exits after printing the help. Then run it with a mesh option:

```bash
./predefined_options --max-level 7
```

The command-line value replaces the maximum level set in the code:

```{literalinclude} snippet/options/predefined_options_output.txt
  :language: text
```

## Predefined options

`samurai::initialize` registers the options below. The `--help` output lists them by group.

### Options

| Option | Value | Effect |
| --- | --- | --- |
| `--config` | file path | Reads option values from a TOML file (see {ref}`howto-options-config`). |
| `-h`, `--help` | flag | Prints the help message and exits. Needs `SAMURAI_PARSE`. |

### SAMURAI

| Option | Value | Effect |
| --- | --- | --- |
| `--min-level` | integer | Minimum level of the mesh. Overrides `mesh_config::min_level`. |
| `--max-level` | integer | Maximum level of the mesh. Overrides `mesh_config::max_level`. |
| `--start-level` | integer | Start level of an AMR mesh. Overrides `mesh_config::start_level`, which defaults to the maximum level. A multiresolution mesh always starts at its maximum level and ignores this option. |
| `--graduation-width` | integer | Graduation width of the mesh. Overrides `mesh_config::graduation_width`. |
| `--max-stencil-radius` | integer | Largest stencil radius of the numerical scheme. Overrides `mesh_config::max_stencil_radius`. |
| `--load-balancing-at` | integer, default 0 | With MPI, rebalances the mesh every N multiresolution adaptations. 0 turns load balancing off. |
| `--sleep-at-startup` | seconds, default 0 | Waits this long in `samurai::initialize`, so you can attach a debugger to a process started with `mpirun` or `mpiexec`. |
| `--finer-level-flux` | integer, default 0 | Level at which fluxes are computed: 0 at the current level, -1 at the maximum level, N > 0 at the current level + N. |
| `--refine-boundary` | flag | Keeps the cells along the boundary at the maximum level. |
| `--print-petsc-numbering` | flag | With PETSc, prints the local and global numbering used by PETSc. |
| `--save-debug-fields` | flag | Adds debug fields (coordinates, indices, levels, ...) to the HDF5 output files. |

### IO (MPI builds only)

| Option | Value | Effect |
| --- | --- | --- |
| `--dont-redirect-output` | flag | Keeps the standard output of every MPI rank. By default, only rank 0 prints. |

### Tools

| Option | Value | Effect |
| --- | --- | --- |
| `--timers` | flag | Prints the timers when `samurai::finalize` runs (see the [timers guide](timers.md)). |
| `--info` | flag | Prints the samurai version, its dependencies and the build configuration, then exits. |

### Multiresolution

| Option | Value | Effect |
| --- | --- | --- |
| `--mr-eps` | real | Threshold on the details used to adapt the mesh. Overrides `mra_config::epsilon` (default `1e-4`). |
| `--mr-reg` | real | Regularity used to adapt the mesh. Overrides `mra_config::regularity` (default `1`). |
| `--mr-rel-detail` | flag | Uses relative details instead of absolute details. |

### Mesh options override the mesh configuration

The mesh options (`--min-level`, `--max-level`, `--start-level`, `--graduation-width` and `--max-stencil-radius`) override the values of the `mesh_config` when `samurai::mra::make_mesh` or `samurai::amr::make_mesh` builds the mesh. A value you set in the code is a default that the user can change at run time. If `--max-level` ends up lower than `--min-level`, or the start level ends up outside that range, building the mesh throws `std::invalid_argument`. Uniform meshes don't read these options.

To keep the values of a `mesh_config` whatever the command line says, call `disable_args_parse()` on it:

```{literalinclude} snippet/options/disable_args_parse.cpp
  :language: c++
  :start-at: auto config
  :end-at: disable_args_parse();
  :dedent:
```

With this configuration, `./disable_args_parse --max-level 7` prints:

```{literalinclude} snippet/options/disable_args_parse_output.txt
  :language: text
```

The multiresolution options work the same way: `--mr-eps`, `--mr-reg` and `--mr-rel-detail` override the `mra_config` you pass to the adaptation, each time the mesh is adapted.

(howto-options-custom)=

## Define custom options

`samurai::initialize` returns the `CLI::App` object that holds the options. Add your options to it between `samurai::initialize` and `SAMURAI_PARSE`, so that `SAMURAI_PARSE` reads them and `--help` lists them:

```{literalinclude} snippet/options/custom_options.cpp
  :language: c++
```

`group("Custom")` puts the two options under a `Custom` heading in the `--help` output. `capture_default_str()` shows the default value next to the option. Run the program with your options:

```bash
./custom_options --my-option 3 --my-flag
```

```{literalinclude} snippet/options/custom_options_output.txt
  :language: text
```

For the other ways to declare options (vectors, validators, required options, ...), see the [CLI11 book](https://cliutils.github.io/CLI11/book/).

```{warning}
samurai accepts unknown options without an error, so a misspelled option such as `--max-levl 7` is ignored. If a value seems to have no effect, check its spelling against the `--help` output.
```

(howto-options-config)=

## Read options from a TOML file

Every option, predefined or custom, can also come from a TOML file. A key is the long option name without the leading `--`, and a flag takes `true` or `false`:

```{literalinclude} snippet/options/config.toml
  :language: toml
```

Pass the file with `--config`:

```bash
./custom_options --config config.toml
```

```{literalinclude} snippet/options/custom_options_config_output.txt
  :language: text
```

An option given on the command line takes precedence over the same key in the file. Keys that match no option are ignored.

## Related

- [How-to: create a samurai mesh](mesh.md) for the `mesh_config` values these options override.
- [How-to: timers in samurai](timers.md) for the output of `--timers`.
- The [CLI11 book](https://cliutils.github.io/CLI11/book/) for the full option API.
