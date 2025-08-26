---
task_categories:
- question-answering
---
# Hardest Sudoku Puzzle Dataset V2

This dataset contains a mixture of easy and very hard Sudoku puzzles collected from the Sudoku community.

## Dataset Composition

### Sources

- [tdoku benchmarks](https://github.com/t-dillon/tdoku/blob/master/benchmarks/README.md#benchmarked-data-sets)
- [enjoysudoku](http://forum.enjoysudoku.com/the-hardest-sudokus-new-thread-t6539-600.html#p277835)

### Easy Puzzles (1.1M)

- puzzles0_kaggle
- puzzles1_unbiased
- puzzles2_17_clue

### Hard Puzzles (3.1M)

- puzzles3_magictour_top1465
- puzzles4_forum_hardest_1905
- puzzles6_forum_hardest_1106
- ph_2010/01_file1.txt

## Dataset Characteristics

- All puzzles have been exact-deduped and randomly permuted by row, column, box, and digit.
- Each puzzle is guaranteed to have a unique solution.
- Puzzles in the train set are [mathematically inequivalent](http://sudopedia.enjoysudoku.com/Mathematically_equivalent.html) to those in the test set.

## Dataset Structure

- Train set: `train.csv` (3.8M examples)
- Test set: `test.csv` (423k examples)

Puzzles and solutions are flattened in row-major order. Rating is evaluated by number of backtracks needed by [tdoku solver]((https://github.com/t-dillon/tdoku) required to solve the puzzle (higher is harder).

## Usage Guidelines

1. Train models using only the train set.
2. Evaluate models on the test set using exact accuracy (all numbers must be correct).