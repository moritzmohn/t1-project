import numpy as np
from einops import rearrange, reduce, repeat

rng = np.random.default_rng()

original_tensor = rng.integers(low=0, high=10, size=(9,8,3))
print(original_tensor)

#transposition of axes
transposed = rearrange(original_tensor, "a b c -> b c a")
print(transposed)
print(transposed.shape)


#composition of axes
composed = rearrange(original_tensor, "a b c -> a (b c)")
print(composed)

#length of the last dimension is 8 * 3 = 24, the new shape is [9, (8 * 3)]
print(composed.shape)

#compose more dimensions (shape gets [(9 * 8 * 3)])
composed_2 = rearrange(original_tensor, "a b c -> (a b c)")
print(composed_2.shape)

#decomposition of axes, set b1 to 2 so b2 gets 4
decomposed = rearrange(original_tensor, "a (b1 b2) c -> a b1 b2 c", b1=2)
print(decomposed)
print(decomposed.shape)


original_tensor_2 = rng.uniform(low=0, high=10, size=(6, 5, 5, 3))
print(original_tensor_2)

#reduce axes, take the average over the first one
reduced = reduce(original_tensor_2, "a b c d -> b c d", "mean")

#prove that the mean was calculated correctly
tensor_slice = original_tensor_2[:, 0, 0, 0]
print(np.mean(tensor_slice))
print(reduced[0, 0, 0])

tensor_slice = original_tensor_2[:, 0, 0, 1]
print(np.mean(tensor_slice))
print(reduced[0, 0, 1])

