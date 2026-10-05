# How-to: measure execution time with timers

Timers measure the wall-clock time spent in parts of a samurai program, so you can find out where a simulation spends its time.
samurai has built-in timers in its own functions (mesh adaptation, ghost update, data saving, ...), and you can add your own around any part of your code.
All timers record only when you run the program with the `--timers` option.

## Before you start

- Your program calls `samurai::initialize` at the start of `main` and `samurai::finalize` at the end.
  `initialize` registers the `--timers` option and starts the `total runtime` timer; `finalize` stops it and prints the report.
  See the [command line options how-to](options.md) if your program does not call them yet.
- Your code includes `samurai/timers.hpp` to create its own timers.

## Print the timers report

Run your program with the `--timers` option:

```bash
./my_program --timers
```

When the program reaches `samurai::finalize`, it prints a table of every timer that ran.
Without `--timers`, timers are disabled: starting and stopping them does nothing and nothing is printed, so you can leave your own timers in production code.

## Time a block of code with `ScopedTimer`

To time a block, create a `samurai::ScopedTimer` at its start.
The timer starts when it is constructed and stops when it goes out of scope.
The following program times the initialization of a field:

```{literalinclude} snippet/timers/scoped_timer.cpp
:language: c++
```

Prefer `ScopedTimer` over explicit calls: the timer stops on every exit from the scope, including an early `return`.

## Start and stop a timer explicitly

When the code to time does not fit in one scope, call `start` and `stop` on the global registry `samurai::times::timers` with the same name.
The following lines time the same initialization as above:

```{literalinclude} snippet/timers/custom_timer.cpp
:language: c++
:start-at: samurai::times::timers.start
:end-at: samurai::times::timers.stop
:dedent:
```

The rest of the program, in `docs/source/howto/snippet/timers/custom_timer.cpp`, is the same as the `ScopedTimer` example.

`stop` must match a timer that was started with the same name and is still running.
Otherwise, when you run with `--timers`, the program writes `[Timers::stop] No active timer named '<name>' on the stack!` to the standard error and terminates.

## Nest timers

A timer started while another timer is running becomes its child in the report.
The order of the calls defines the nesting, with nothing to configure.
Every timer of your program is a child of `total runtime`, because `samurai::initialize` starts that timer first.

The following program solves $\partial_t u = u (1 - u)$ with the explicit Euler method.
It times the initialization, the whole time loop, and two stages inside the loop:

```{literalinclude} snippet/timers/nested_timers.cpp
:language: c++
```

A timer with the same name started under two different parents gives two separate rows, one under each parent.
A timer started several times under the same parent gives one row: its time adds up and its call count grows.

## Measure cell throughput

To see how many cells per second a stage processes, give the number of cells it handled each time it stops:

- with explicit calls, use `samurai::times::timers.stop(name, nb_cells)`;
- with a `ScopedTimer`, call `set_cells(nb_cells)` before the end of the scope.

Count the cells with `mesh.nb_cells(mesh_id_t::cells)`, as the example does.
Without an argument, `mesh.nb_cells()` returns the size of the field storage, which includes the ghost cells.

The counts add up over all calls.
If at least one timer has a non-zero count, the report adds a `Mcells/s` column: the total number of cells divided by the total time, in millions of cells per second.
Timers without a count leave that column empty.

## Read the report

Run the program of the previous section with `--timers` on one process:

```bash
./nested_timers --timers
```

It prints this report (the times change from run to run):

```text
 Timers
Timer                              Elapsed (s)  % total   % parent   Calls   Mcells/s
-------------------------------------------------------------------------------------
total runtime                            0.170   100.0%                  1
+-- time loop                            0.167    98.2%      98.2%       1
|   +-- update                           0.091    53.5%      54.5%     100     1152.2
|   `-- rhs computation                  0.076    44.6%      45.5%     100     1380.0
+-- init field                           0.002     1.4%       1.4%       1
+-- update_sub_mesh                      0.000     0.2%       0.2%       1
|   `-- update_meshid_neighbour          0.000     0.0%       0.0%       1
+-- construct_union                      0.000     0.1%       0.1%       1
+-- construct_corners                    0.000     0.0%       0.0%       1
+-- renumbering                          0.000     0.0%       0.0%       1
+-- construct_subdomain                  0.000     0.0%       0.0%       1
+-- exchange neighbour meshes            0.000     0.0%       0.0%       1
+-- update_mesh_neighbour                0.000     0.0%       0.0%       1
`-- update_meshid_neighbour              0.000     0.0%       0.0%       1
-------------------------------------------------------------------------------------
(untimed)                                0.000     0.1%
-------------------------------------------------------------------------------------
total runtime                            0.170   100.0%
```

The rows `update_sub_mesh`, `construct_union` and the others below `init field` are built-in samurai timers: `samurai::mra::make_mesh` runs them when it builds the mesh.

- `Elapsed (s)`: total time spent in the timer, over all its calls.
- `% total`: share of `total runtime`.
- `% parent`: share of the parent timer.
- `Calls`: number of times the timer stopped.
- `(untimed)`: the part of `total runtime` that no direct child of `total runtime` covers.

Children are sorted by decreasing time under each parent.
In color, the root rows are bold and the other rows are red from 20% of the total runtime, yellow from 5% to 20% and green below 5%.
If the direct children of `total runtime` add up to more than its time, for example because two of them overlap, the `(untimed)` row reads `(overlap / pause-resume timers)` instead.

## Run with MPI

With MPI, every rank measures its own times.
When `samurai::finalize` runs, rank 0 gathers the times of all ranks and prints a single report, with the minimum, the maximum and the average over the ranks for each timer.
The `[r]` columns give the rank that reached the minimum and the maximum, which shows load imbalance.
`% total`, `% parent`, the sort order and `(untimed)` use the average.

Run the program with `mpirun` and `--timers`:

```bash
mpirun -np 4 ./nested_timers --timers
```

The same program on 4 ranks prints:

```text
 Timers
Timer                                Min (s)   [r]    Max (s)   [r]    Ave (s)  % total   % parent   Calls
----------------------------------------------------------------------------------------------------------
total runtime                          0.047   [1]      0.054   [0]      0.051   100.0%                  1
+-- time loop                          0.046   [1]      0.052   [0]      0.050    96.9%      96.9%       1
|   +-- update                         0.025   [1]      0.029   [2]      0.027    52.6%      54.3%     100
|   `-- rhs computation                0.020   [1]      0.025   [0]      0.023    44.2%      45.6%     100
+-- init field                         0.001   [2]      0.001   [3]      0.001     1.6%       1.6%       1
+-- update_mesh_neighbour              0.000   [0]      0.000   [2]      0.000     0.6%       0.6%       1
+-- exchange neighbour meshes          0.000   [3]      0.000   [1]      0.000     0.2%       0.2%       1
+-- update_sub_mesh                    0.000   [1]      0.000   [0]      0.000     0.2%       0.2%       1
|   `-- update_meshid_neighbour        0.000   [0]      0.000   [2]      0.000     0.1%      28.1%       1
+-- construct_corners                  0.000   [0]      0.000   [3]      0.000     0.1%       0.1%       1
+-- update_meshid_neighbour            0.000   [1]      0.000   [3]      0.000     0.1%       0.1%       1
+-- construct_subdomain                0.000   [0]      0.000   [2]      0.000     0.0%       0.0%       1
+-- construct_union                    0.000   [2]      0.000   [3]      0.000     0.0%       0.0%       1
`-- renumbering                        0.000   [3]      0.000   [2]      0.000     0.0%       0.0%       1
----------------------------------------------------------------------------------------------------------
(untimed)                                                     0.000     0.1%
----------------------------------------------------------------------------------------------------------
total runtime (ave)                                           0.051   100.0%
```

The MPI report has no `Mcells/s` column, even when you pass cell counts.

```{important}
Start the same timers, under the same parents, on every rank.
Rank 0 matches the times of the other ranks to its own list of timers, so a timer that runs on only some ranks makes the report wrong and can crash the program.
```

## Related

- [Command line options how-to](options.md): other options that `samurai::initialize` registers.
- `include/samurai/timers.hpp`: the `samurai::Timers` and `samurai::ScopedTimer` classes.
