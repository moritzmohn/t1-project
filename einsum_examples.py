import numpy as np

"""
original_tensor = np.arange(120).reshape(5, 2, 3, 4)
print(original_tensor)


#reorder the axes
#shape gets (5,3,4,2)

#implicit form / axes get ordered alphabetically
print(np.einsum('adbc', original_tensor).shape)

#explicit form
print(np.einsum('abcd -> acdb', original_tensor).shape)

print(np.einsum(original_tensor, [0, 3, 1, 2]).shape)



#summing over one axis / requires explicit form

a = np.arange(12).reshape(3, 4)

print(a)

#sum over second axis (applied along second)
print(np.einsum('ij->i', a))

np.einsum(a, [0,1], [0])

print(np.sum(a, axis=1))

#sum over the last axis in multidimensional tensor
print(np.einsum('...j->...', original_tensor).shape)

print(np.einsum(original_tensor, [Ellipsis,1], [Ellipsis]).shape)
"""
x = np.arange(120).reshape(5, 2, 3, 4)

print(np.einsum('abcd->acd', x))
'''
print(np.einsum(x, [0,1,2,3], [0,2,3]))

print(np.sum(x, axis=1))
'''