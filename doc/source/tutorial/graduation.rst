This tutorial will highlight three different ways to make the graduation of a mesh. A mesh is graduated when the neighbors of a cell at level :math:`l` are at most at the next or previous level.

The three cases that will be considered in the following are:

- The mesh is constituted of cells at different levels but without overlap and we want a graduated mesh at the end.
- The mesh is constituted of cells at different levels with overlap and we want a graduated mesh at the end.
- The mesh is already graduated and a mesh adaptation algorithm is performed on it. A tag array indicates which cells must be refined, kept, or coarsen to build the new mesh. We want to modify the tag array to be sure that the new mesh created from the tag will be graduated.