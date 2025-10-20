# How-to: Timers in samurai

In this how-to guide, we will show you how to use timers in samurai. Timers are useful for measuring the execution time of different parts of your code, which can help you identify performance bottlenecks and optimize your simulations.

## Get default timers output

By default, samurai provides built-in timers that measure the execution time of various components, such as mesh refinement, field updates, and I/O operations. To print the default timers output, you can use the `--timers` command line option when running your simulation. This will print a summary of the timers to the console at the end of the simulation.

## Using timers

You can also create your own timers to measure specific parts of your code. samurai provides a simple timer class that you can use to start and stop timers. Here is an example of how to use timers in your code:

```{literalinclude} snippet/timers/custom_timer.cpp
 :language: c++
```

In this example, we create a 2D multi-resolution mesh and a scalar field on the mesh. We then use the `times::timers.start("init field")` and `times::timers.stop("init field")` functions to start and stop a timer named "init field" around the loop that initializes the field values. This will measure the time taken to initialize the field.

```{note}
Make sure to include the header file `samurai/timers.hpp` to use the timer functionality.
