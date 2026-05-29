import numpy as np
from einops import rearrange, reduce, repeat
import math

rng = np.random.default_rng()

original_tensor = np.arange(216).reshape(9, 8, 3)
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

#reduce axes, take the average over the first axis, compare to the numpy notation
reduced_einops = reduce(original_tensor_2, "a b c d -> b c d", "mean")
reduced_np = np.mean(original_tensor_2, axis=0)
print(reduced_einops)
print(reduced_np)

np.testing.assert_allclose(reduced_einops, reduced_np, atol=1e-15)
np.testing.assert_equal(reduced_einops, reduced_np)

#prove that the mean was calculated correctly
tensor_slice = original_tensor_2[:, 0, 0, 0]
print(np.mean(tensor_slice))
print(reduced_einops[0, 0, 0])

for i in range(5):
    for j in range(5):
        for k in range(3):
            slice = original_tensor_2[:, i, j, k]
            a = np.mean(slice)
            b = reduced_einops[i, j, k]
            assert math.isclose(a, b, abs_tol=1e-6)


#summing over axis

original_tensor_4 = np.arange(120).reshape(5, 2, 3, 4)

summed = reduce(original_tensor_4, "a b c d -> a c d", "sum")
print(summed)



#add or remove axes

#add axes of size one (compare to numpy expand_dims)
expanded_einops = rearrange(original_tensor_2, "a b c d -> a b 1 c 1 d")
expanded_np = np.expand_dims(original_tensor_2, axis=(2, 4))
print(expanded_einops)
print(expanded_np.shape)
np.testing.assert_equal(expanded_einops, expanded_np)


#repeat

original_tensor_3 = np.arange(36).reshape(3, 3, 4)

print(original_tensor_3)

print("\n -------------------------------- \n")

print(repeat(original_tensor_3, "a b c -> a (repeat b) c", repeat=3))

print("\n -------------------------------- \n")

#the position of repeat matters: here each row is repeated 3 times / the same row appears 3 times directly after each other
print(repeat(original_tensor_3, "a b c -> a (b repeat) c", repeat=3))

print("\n -------------------------------- \n")

print(repeat(original_tensor_3, "a b c -> a b (c repeat)", repeat=3))