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
The timer starts when it is constructed and stops when it goes out of scope:

```cpp
#include <samurai/timers.hpp>

{
    samurai::ScopedTimer loop_timer("time loop");
    // ... the code to time
} // loop_timer stops here
```

Prefer `ScopedTimer` over explicit calls: the timer stops on every exit from the scope, including an early `return`.

## Start and stop a timer explicitly

When the code to time does not fit in one scope, call `start` and `stop` on the global registry `samurai::times::timers` with the same name.
The following example times the initialization of a field:

```{literalinclude} snippet/timers/custom_timer.cpp
:language: c++
```

`stop` must match a timer that was started with the same name and is still running.
Otherwise, when you run with `--timers`, the program writes `[Timers::stop] No active timer named '<name>' on the stack!` to the standard error and terminates.

## Nest timers

A timer started while another timer is running becomes its child in the report.
The order of the calls defines the nesting, with nothing to configure.
Every timer of your program is a child of `total runtime`, because `samurai::initialize` starts that timer first.

The following time loop times the whole loop and two stages inside it:

```cpp
{
    samurai::ScopedTimer loop_timer("time loop");
    for (std::size_t nt = 0; nt < nb_steps; ++nt)
    {
        {
            samurai::ScopedTimer flux_timer("flux computation");
            // ... compute the fluxes
            flux_timer.set_cells(mesh.nb_cells());
        }

        samurai::times::timers.start("update");
        // ... update the solution
        samurai::times::timers.stop("update", mesh.nb_cells());
    }
}
```

A timer with the same name started under two different parents gives two separate rows, one under each parent.
A timer started several times under the same parent gives one row: its time adds up and its call count grows.

## Measure cell throughput

To see how many cells per second a stage processes, give the number of cells it handled each time it stops:

- with explicit calls, use `samurai::times::timers.stop(name, nb_cells)`;
- with a `ScopedTimer`, call `set_cells(nb_cells)` before the end of the scope.

The counts add up over all calls.
If at least one timer has a non-zero count, the report adds a `Mcells/s` column: the total number of cells divided by the total time, in millions of cells per second.
Timers without a count leave that column empty.

## Read the report

The report of the time loop above, run on one process, has this shape:

```{note}
This output was not measured.
It was built from the printing code in `include/samurai/timers.hpp` to show the format, with made-up times.
A real run also lists the built-in samurai timers that ran (for example `mesh adaptation` or `ghost update`), and the terminal shows the rows in color.
```

```text
 Timers
Timer                       Elapsed (s)  % total   % parent   Calls   Mcells/s
------------------------------------------------------------------------------
total runtime                     2.000   100.0%                  1
+-- time loop                     1.704    85.2%      85.2%       1
|   +-- flux computation          1.138    56.9%      66.8%     100        1.4
|   `-- update                    0.431    21.6%      25.3%     100        3.8
`-- init field                    0.052     2.6%       2.6%       1
------------------------------------------------------------------------------
(untimed)                         0.244    12.2%
------------------------------------------------------------------------------
total runtime                     2.000   100.0%
```

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
mpirun -np 4 ./my_program --timers
```

The same time loop on 4 ranks gives a report of this shape (not measured either, built the same way as the sequential one):

```text
 Timers
Timer                         Min (s)   [r]    Max (s)   [r]    Ave (s)  % total   % parent   Calls
---------------------------------------------------------------------------------------------------
total runtime                   0.802   [2]      0.806   [0]      0.804   100.0%                  1
+-- time loop                   0.517   [1]      0.524   [3]      0.521    64.8%      64.8%       1
|   +-- flux computation        0.271   [2]      0.312   [0]      0.290    36.1%      55.7%     100
|   `-- update                  0.104   [1]      0.121   [3]      0.112    13.9%      21.5%     100
`-- init field                  0.011   [3]      0.016   [1]      0.013     1.6%       1.6%       1
---------------------------------------------------------------------------------------------------
(untimed)                                              0.270    33.6%
---------------------------------------------------------------------------------------------------
total runtime (ave)                                    0.804   100.0%
```

The MPI report has no `Mcells/s` column, even when you pass cell counts.

```{important}
Start the same timers, under the same parents, on every rank.
Rank 0 matches the times of the other ranks to its own list of timers, so a timer that runs on only some ranks makes the report wrong and can crash the program.
```

## Related

- [Command line options how-to](options.md): other options that `samurai::initialize` registers.
- `include/samurai/timers.hpp`: the `samurai::Timers` and `samurai::ScopedTimer` classes.
