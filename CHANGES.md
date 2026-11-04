# Changes relative to the original scripts

## Class names

- `ClonalTree` and `ARG` now keep their own class names (`type(x).__name__`
  and `repr`). Before, `_Tree.__init__` overwrote `self.__class__.__name__`
  with `"Tree"` on every instance, which renamed the subclass itself: every
  `ARG` reported its class as `Tree`, and `ClonalTree` was renamed to `Tree`
  and back each time a tree was built.
- The base class `_Tree` is named `Tree`. This is set once when the class is
  defined, not on each instance. Its `__qualname__` is still `_Tree`, so
  pickling works.

## Public API

- `segment_summary_stats(tree, seg_mat)` is public. It can be imported from
  `bacnpe` and from `bacnpe.summary_stats`, and it is listed in both
  `__all__` lists.
