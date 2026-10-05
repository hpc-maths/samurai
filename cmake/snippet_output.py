#!/usr/bin/env python3
"""Run a documentation snippet and write its output to a file."""

# The pages of the documentation include these files with `literalinclude`.
# CMake calls this script for each `samurai_snippet_output` declaration (see
# cmake/snippetOutputs.cmake); do not edit the generated files by hand.
#
# The program runs in a temporary directory, so that the files it writes
# (HDF5, xdmf, ...) do not land in the source tree. It is started as
# `./<name>`, the way the pages show it, so that a program printing its own
# name prints the same text on every machine. Only the standard output is
# captured, unless --stderr is given.

import argparse
import os
import re
import shutil
# The script runs the snippets that CMake built, with the arguments of
# their CMakeLists.txt: no untrusted input reaches subprocess.
import subprocess  # nosec B404
import sys
import tempfile


def parse_args():
    """Read the command line, check the options that go together."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="file to write")
    parser.add_argument(
        "--exit-code", type=int, default=0, help="expected exit code of the program"
    )
    parser.add_argument(
        "--stderr",
        action="store_true",
        help="capture the standard error with the standard output",
    )
    parser.add_argument(
        "--input", action="append", default=[], help="file copied into the working directory"
    )
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="REGEX",
        help="replace every match of REGEX (Python syntax, multiline mode) in the output"
        " by the next --by",
    )
    parser.add_argument(
        "--by",
        action="append",
        default=[],
        metavar="REPLACEMENT",
        help="replacement of a --replace",
    )
    parser.add_argument(
        "--head", type=int, help="keep only the first HEAD lines, followed by '...'"
    )
    parser.add_argument(
        "--tail", type=int, help="keep only the last TAIL lines, preceded by '...'"
    )
    parser.add_argument("--mpiexec", help="MPI launcher, to run the program on --np processes")
    parser.add_argument("--np", type=int, help="number of MPI processes")
    parser.add_argument(
        "--numproc-flag", default="-n", help="option of the launcher for the number of processes"
    )
    parser.add_argument(
        "--mpiexec-preflag", action="append", default=[], help="option of the launcher"
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="time limit of the run, in seconds"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- program [arguments]")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing the program to run after --")
    if len(args.replace) != len(args.by):
        parser.error("each --replace needs a --by")
    if (args.mpiexec is None) != (args.np is None):
        parser.error("--mpiexec and --np go together")
    return args


def elide(text, head, tail):
    """Keep the first `head` and the last `tail` lines, with '...' in place of the others."""
    lines = text.splitlines()
    head = head or 0
    tail = tail or 0
    if head + tail >= len(lines):
        return text
    kept = lines[:head] + ["..."] + (lines[len(lines) - tail :] if tail else [])
    return "\n".join(kept) + "\n"


def run(args):
    """Run the program in a temporary directory, return its standard output."""
    program = os.path.abspath(args.command[0])
    name = os.path.basename(program)
    with tempfile.TemporaryDirectory(prefix="samurai-snippet-") as workdir:
        for path in args.input:
            shutil.copy(path, workdir)
        os.symlink(program, os.path.join(workdir, name))
        launcher = []
        if args.mpiexec:
            launcher = [args.mpiexec, args.numproc_flag, str(args.np)] + args.mpiexec_preflag
        command = launcher + [f"./{name}"] + args.command[1:]
        # One thread for the BLAS and OpenMP reductions: their result must not
        # depend on the number of cores of the machine
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            result = subprocess.run(  # nosec B603
                command,
                cwd=workdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if args.stderr else None,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            sys.exit(f"{' '.join(command)}: no result after {error.timeout} s")
        output = result.stdout.decode("utf-8", errors="replace")
        # Paths that contain the temporary directory change at each run
        output = output.replace(os.path.realpath(workdir), ".").replace(workdir, ".")

    if result.returncode != args.exit_code:
        sys.stderr.write(output)
        sys.exit(
            f"{' '.join(command)}: exit code {result.returncode}, expected {args.exit_code}"
            f" (set EXIT_CODE in samurai_snippet_output to accept it)"
        )
    return output


def main():
    """Run the program, normalize its output and write the file if it changed."""
    args = parse_args()
    output = run(args)
    for pattern, replacement in zip(args.replace, args.by):
        output = re.sub(pattern, replacement, output, flags=re.MULTILINE)
    if args.head is not None or args.tail is not None:
        output = elide(output, args.head, args.tail)

    try:
        with open(args.output, encoding="utf-8", newline="") as file:
            previous = file.read()
    except FileNotFoundError:
        previous = None
    # Rewrite the file only when it changes, to keep its timestamp otherwise
    if output != previous:
        with open(args.output, "w", encoding="utf-8", newline="") as file:
            file.write(output)
        print(f"updated {args.output}")


if __name__ == "__main__":
    main()
