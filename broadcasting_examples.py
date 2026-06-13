import numpy as np
import einx
import einops

#np.broadcast_to in einx and einops
x = np.array([1, 2, 3])
y = np.broadcast_to(x, (2, 3, 3))

#result = (2,3,3)
print(y.shape)


print(einx.id("c-> a b c", x, a = 2, b = 3))

print(einx.id("c-> 2 3 c", x))

print(einops.repeat(x, "c -> 2 3 c"))