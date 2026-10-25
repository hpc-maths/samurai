#!/usr/bin/env python3
# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
"""Prepare the compilation database read by the SonarQube Cloud C++ analyzer.

The coverage job (.github/workflows/coverage.yml) builds pull request code, and
the Sonar workflow (.github/workflows/sonar.yml) scans it with SONAR_TOKEN. The
compilation database travels between them as an artifact, so the scan treats
it as untrusted: the analyzer runs the compiler of each entry to query its
configuration, and a database written by pull request code could otherwise
name any program or pass options such as -B, -specs or @file that make GCC run
one.

merge
    Concatenate compilation databases, keep the entries of the samurai sources
    (the files under the workspace, outside the build directories) and write
    the include directories found under the build directories, which the scan
    job does not have (the googletest and Google Benchmark headers).

sanitize
    Rewrite each entry with a compiler chosen by the scan job and only the
    options that select headers and macros: -I, -isystem, -iquote, -D, -U and
    -std. Entries whose file is not a source file of the workspace are dropped.

The script only uses the standard library. Run it with `python3 -I` from a
directory outside the checkout, so that no module of the checkout is imported.
"""

import argparse
import json
import os
import re
import shlex
import sys

INCLUDE_OPTIONS = ("-isystem", "-iquote", "-I")
MACRO_OPTIONS = ("-D", "-U")
STD_RE = re.compile(r"^-std=(c|gnu)\+\+[0-9a-z]{2}$")
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx")


def is_under(path, directory):
    return os.path.commonpath([path, directory]) == directory


def load(path):
    with open(path, encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise ValueError(f"{path}: a compilation database is a JSON array")
    return entries


def entry_arguments(entry):
    if isinstance(entry.get("arguments"), list):
        arguments = entry["arguments"]
    elif isinstance(entry.get("command"), str):
        arguments = shlex.split(entry["command"])
    else:
        raise ValueError(f"entry without arguments nor command: {entry!r}")
    if not all(isinstance(argument, str) for argument in arguments):
        raise ValueError(f"non-string argument in {entry!r}")
    return arguments


def entry_file(entry):
    directory, file = entry.get("directory"), entry.get("file")
    if not isinstance(directory, str) or not isinstance(file, str):
        raise ValueError(f"entry without directory or file: {entry!r}")
    return directory, os.path.normpath(os.path.join(directory, file))


def safe_options(arguments, directory):
    """Yield the header and macro options of a command line, joined to their value."""
    iterator = iter(arguments[1:])
    for argument in iterator:
        option = next((o for o in INCLUDE_OPTIONS + MACRO_OPTIONS if argument.startswith(o)), None)
        if option is None:
            if STD_RE.match(argument):
                yield argument
            continue
        value = argument[len(option) :]
        if not value:
            value = next(iterator, "")
        if not value:
            continue
        if option == "-I":
            yield option + os.path.normpath(os.path.join(directory, value))
        elif option in INCLUDE_OPTIONS:
            yield from (option, os.path.normpath(os.path.join(directory, value)))
        else:
            yield option + value


def include_paths(arguments, directory):
    """Yield the absolute include directories of a command line."""
    options = list(safe_options(arguments, directory))
    for i, option in enumerate(options):
        if option in INCLUDE_OPTIONS[:2]:
            yield options[i + 1]
        elif option.startswith("-I"):
            yield option[2:]


def merge(args):
    workspace = os.path.realpath(args.workspace)
    build_dirs = [os.path.realpath(os.path.join(workspace, d)) for d in args.build_dir]
    merged, seen, include_dirs = [], set(), set()
    for database in args.databases:
        for entry in load(database):
            directory, file = entry_file(entry)
            if not is_under(file, workspace) or any(is_under(file, d) for d in build_dirs):
                continue
            if file in seen:
                continue
            seen.add(file)
            merged.append(entry)
            for path in include_paths(entry_arguments(entry), directory):
                if any(is_under(path, d) for d in build_dirs) and os.path.isdir(path):
                    include_dirs.add(os.path.relpath(path, workspace))
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(merged, stream, indent=1)
    with open(args.include_dirs, "w", encoding="utf-8") as stream:
        stream.writelines(f"{d}\n" for d in sorted(include_dirs))
    print(f"{len(merged)} entries, {len(include_dirs)} include directories under the build directories")


def sanitize(args):
    workspace = os.path.realpath(args.workspace)
    compiler = os.path.realpath(args.compiler)
    if not os.path.isfile(compiler) or not os.access(compiler, os.X_OK):
        raise ValueError(f"{args.compiler} is not an executable file")
    sanitized, seen = [], set()
    for entry in load(args.input):
        directory, file = entry_file(entry)
        if (
            not is_under(file, workspace)
            or not file.endswith(SOURCE_SUFFIXES)
            or file in seen
            or not os.path.isfile(file)
        ):
            print(f"dropped: {file}", file=sys.stderr)
            continue
        seen.add(file)
        options = list(safe_options(entry_arguments(entry), directory))
        sanitized.append({"directory": workspace, "file": file, "arguments": [compiler, *options, "-c", file]})
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(sanitized, stream, indent=1)
    print(f"{len(sanitized)} entries kept")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    merge_parser = commands.add_parser("merge", help="merge the databases of the coverage job")
    merge_parser.add_argument("--workspace", required=True)
    merge_parser.add_argument("--build-dir", action="append", default=[], help="build directory, relative to the workspace (repeatable)")
    merge_parser.add_argument("--output", required=True)
    merge_parser.add_argument("--include-dirs", required=True, help="file listing the include directories to ship")
    merge_parser.add_argument("databases", nargs="+")
    merge_parser.set_defaults(func=merge)

    sanitize_parser = commands.add_parser("sanitize", help="rewrite an untrusted database for the scan")
    sanitize_parser.add_argument("--workspace", required=True)
    sanitize_parser.add_argument("--compiler", required=True)
    sanitize_parser.add_argument("--output", required=True)
    sanitize_parser.add_argument("input")
    sanitize_parser.set_defaults(func=sanitize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
