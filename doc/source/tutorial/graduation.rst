Algorithm examples for the graduation of a mesh
===============================================

.. toctree::
   :hidden:

   graduation_case_1
   graduation_case_2
   graduation_case_3

This tutorial will highlight three different ways to make the graduation of a mesh. A mesh is graduated when the neighbors of a cell at level :math:`l` are at most at the next or previous level. This process is important when we use adaptive mesh refinement techniques. The graduation ensures that the reconstruction of the ghost cell values of level `l` can be made with the previous or the next level.

The three cases that will be considered in the following are:

   - :doc:`Case 1 <./graduation_case_1>`

        The mesh is constituted of cells at different levels but without overlap and we want a graduated mesh at the end.

   - :doc:`Case 2 <./graduation_case_2>`

        The mesh is constituted of cells at different levels with overlap and we want a graduated mesh at the end.

   - :doc:`Case 3 <./graduation_case_3>`

        The mesh is already graduated and a mesh adaptation algorithm is performed on it. A tag array indicates which cells must be refined or kept to build the new mesh. We want to modify the tag array to be sure that the new mesh created from the tag will be graduated.
