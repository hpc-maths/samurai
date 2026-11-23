# How-to: keep the traversal of a set expression in an entry table

This guide shows how to run a computation again and again on the cells a set expression selects, without traversing the set expression each time.
You describe the cells once, in a key, and `samurai::for_each_entry` keeps what the traversal gives in a table of the mesh.
The ghost update of {{ project }} fills its ghosts this way.

## Before you start

- You have a mesh and a field. See the {doc}`mesh how-to <mesh>` and the {doc}`field how-to <field>`.
- You know how to traverse a set expression with a function of an interval `i` and an index `index`. See the {doc}`subset reference <../reference/subset>`.
- Your computation runs many times between two changes of the mesh. A table is recorded the first time it is asked for, and dropped when the mesh rebuilds its cells: it pays off when it is read more than once.
- Your set expression depends on the mesh and on a few integers only (a level, a direction, a number of layers). What it depends on becomes the members of the key.

The examples read and write the values of a field through its storage, `field.data()`, where the cell of storage index `k` of a scalar field is `field.data()[k]`.

## Describe the entries with a key

A key is a struct that gives three things:

- the type of the entries, `entry_type`. `samurai::Rows<N>` is N rows of consecutive cells of the same length, each given by the storage index of its first cell. `samurai::Cells<N>` is N cells. You can also write your own struct, as long as it is trivially copyable.
- as members, what the set expression depends on besides the mesh.
- a member function `produce(mesh, emit)` that traverses the set expression and calls `emit(entry)` for each entry.

The following program defines the key `cell_and_next`: at one level, the rows of cells whose next cell along $x$ is also a cell.
For each interval of the set, `samurai::make_rows` turns the first cell of the row and its next cell into storage indices:

```{literalinclude} snippet/entry_tables/entry_table.cpp
  :language: c++
  :start-at: // The cells of one level
  :end-before: int main
```

Two keys are the same when their bytes are the same.
The members of a key must therefore be of one integer type, such as `std::size_t`, so that the struct has no padding.
To keep a direction, store `samurai::direction_code(direction)` and get the vector back with `samurai::direction_from_code<dim>(code)`.

## Run a kernel on the entries

Call `samurai::for_each_entry(mesh, key, kernel)`.
The kernel is a function of one entry.
Here it computes, for each cell, the difference with its next cell, three times in a row:

```{literalinclude} snippet/entry_tables/entry_table.cpp
  :language: c++
  :start-at: // the kernel
  :end-at: differences per step
  :dedent:
```

The first call for a key records its table, by calling `produce`.
The next calls read the table: the set expression is not traversed again.
The entries come in the order `produce` emits them.

## Use entries with a variable number of cells

When the number of cells changes from one entry to the next, put a `samurai::PoolRange` in your entry struct and fill it in `produce`:

```c++
struct my_entry
{
    std::size_t cell;
    samurai::PoolRange neighbours;
};

// in produce
my_entry entry{cell_index, emit.new_range()};
emit.add(entry.neighbours, neighbour_index); // as many times as needed
emit(entry);
```

The kernel then takes the pool as second argument and reads the cells of the range with `samurai::cells_of`:

```c++
samurai::for_each_entry(mesh, my_key{level},
                        [&](const auto& entry, std::span<const std::size_t> pool)
                        {
                            for (const auto neighbour : samurai::cells_of(pool, entry.neighbours))
                            {
                                // ...
                            }
                        });
```

Fill one range at a time: `emit.add` appends to the range opened last.

## Run without tables

Set the environment variable `SAMURAI_ENTRY_TABLES` to `0` to make every `for_each_entry` traverse its set expression at each call:

```bash
SAMURAI_ENTRY_TABLES=0 ./my_program
```

The kernel and the results are the same; only the time changes.
This is the way to compare both paths, or to save the memory of the tables.
In a program, `samurai::entry_tables_enabled() = false;` does the same.

A mesh type that has no `entry_tables()` member also runs without tables.

## Know when a table is dropped

A mesh empties its tables each time it rebuilds its cells or the meshes of its neighbours: after a mesh adaptation or a load balancing, the next call records the table again.
A copy of a mesh starts without tables, and a swap of two meshes swaps their tables.

A table does not follow a change made to the mesh through its non-const accessors, such as `mesh.cfg()`.
Change the configuration of a mesh before the first call to `for_each_entry`.

## Next steps

- {doc}`Loop over cells <loop>`: the loops to use when the computation runs once.
- {doc}`Subset reference <../reference/subset>`: the set expressions a key can traverse.
- {doc}`Measure execution time <timers>`: find the computations worth a table.
