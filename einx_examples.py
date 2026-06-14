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
# With the einx notation we specifically show that we are reducing the 't' axis, 
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

y = einx.id("b t (h d) -> b h t d", x, h=8)
print(y.shape)
# Splitting and Grouping axes is much easier and understandable 

#indexing

x = np.arange(12).reshape(3, 4)
indices = np.array([0, 1])

print(einx.get_at("[z] a, b-> b a", x, indices))

x = np.arange(12).reshape(3, 4)
indices = np.array([0, 1])

print(einx.get_at("a [z], b-> a b", x, indices))

x = np.arange(12).reshape(3, 4)
y = np.array([[0, 1,2, 2],[0, 1, 0, 1]])

print(einx.get_at("[x] a, b a-> b a", x, y))



# A simpile neural netwrok layer showing various arbitrary operations in einx


# ============================================================
# Simulated input: batch of token/image-patch features
# Shape convention:
#   b = batch
#   n = sequence length / number of patches
#   c = channels
# ============================================================
batch_size = 32
num_positions = 256
channels = 64

x = np.random.randn(batch_size, num_positions, channels)  # (32, 256, 64)


# 1. Add a learnable bias to each channel
bias = np.random.randn(channels)  # (64,)
x_bias_added = einx.add("b n c, c -> b n c", x, bias)
print("After bias addition:", x_bias_added.shape)  # (32, 256, 64)


# 2. Grouped linear projection
# Split the 64 channels into 4 groups of 16 channels each
# and apply a separate linear map per group
num_groups = 4
in_channels_per_group = channels // num_groups   # 16
out_channels_per_group = 32

# Weight tensor: one matrix per group
# shape = (g, c_in, c_out) = (4, 16, 32)
w = np.random.randn(num_groups, in_channels_per_group, out_channels_per_group)

x_projected = einx.dot(
    "b n (g [c_in]), g [c_in] c_out -> b n (g c_out)",
    x_bias_added,
    w,
    g=num_groups,
    c_out=out_channels_per_group,
)
print("After grouped linear:", x_projected.shape)  # (32, 256, 128)

# 3. Mean-pool across positions

x_mean = einx.mean("b [n] c -> b c", x_projected)
print("After mean pooling:", x_mean.shape)  # (32, 128)

# 4. Channel-wise scaling

scale = np.random.randn(num_groups * out_channels_per_group)  # (128,)
x_scaled = einx.multiply("b c, c -> b c", x_mean, scale)
print("After scaling:", x_scaled.shape)  # (32, 128)

# 5. Softmax over channels to get normalized scores

x_scores = einx.softmax("b [c] -> b [c]", x_scaled)
print("After softmax:", x_scores.shape)  # (32, 128)

# 6. Select a few batch items

sample_indices = np.array([0, 5, 10, 20])
x_selected = einx.get_at("[b] c, i -> i c", x_scores, sample_indices)
print("After selection:", x_selected.shape)  # (4, 128)

print("\nFinal tensor shape:", x_selected.shape) # (4,128)
print("Row sums (should be close to 1.0 after softmax):")
print(np.sum(x_selected, axis=1))