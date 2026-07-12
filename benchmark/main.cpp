// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Entry point for the bench_samurai executable (see benchmark/CMakeLists.txt,
// SAMURAI_BENCHMARKS): the translation units listed there are compiled into
// this single binary, each contributing the benchmarks it registers via
// BENCHMARK(...). BENCHMARK_MAIN() expands to Google Benchmark's own main(),
// which parses the --benchmark_* command-line flags and runs every
// benchmark registered process-wide.

#include <benchmark/benchmark.h>

BENCHMARK_MAIN();
