# The key idea of einx notaion is that 
# Axis names describe tensor structure directly, instead of relying on positional indexing.

import einx
import numpy as np

x = np.ones((10, 20, 30))

y = x.mean(axis=1) # Traditional tensor code
print(y.shape)
# can be written as
y = einx.mean("b t d -> b d", x) 
print(y.shape)
# With the einx notaion we specifically show that we are reducing the 't' axis, 
# and that the output will have 'b' and 'd' axes only.

# A simple add opeartion between two tensors

x = np.random.randn(32, 128)
y = np.random.randn(32, 128)

z = einx.add("b d, b d -> b d", x, y)
print(z.shape)

# Broadcasting works as well
x= np.arange(24).reshape(8, 3)
y = np.arange(3)

# Normal broadcasting in numpy
z = np.add(x, y)
print(z.shape)

# Can be written as 
z = einx.add("a b, b -> a b", x, y)
print(z)
# Here we explicitly show that the bias is being broadcasted along the 'b' axis, and that the output will have both 'b' and 'd' axes.

#implicitly without specifying the output shape
z = einx.add("a b, b", x, y)
print(z)

# Matrix multiplication in einx notation
x = np.random.randn(32, 128)
w = np.random.randn(128, 256)

y = einx.dot("b i, i o -> b o", x, w)
print(y.shape)
# Here we specify that we are multiplying along the 'i' axis, and that the output will have 'b' and 'o' axes.


# Multi Head Attention in einx notation
x = np.random.randn(32, 128, 512)

y = einx.rearrange("b t (h d) -> b h t d", x, h=8)
print(y.shape)
# Splitting and Grouping axes is much easier and understandable 
