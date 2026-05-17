import numpy as np


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

x = np.arange(120).reshape(5, 2, 3, 4)

print(np.einsum('abcd->acd', x))

print(np.einsum(x, [0,1,2,3], [0,2,3]))

print(np.sum(x, axis=1))



#-----------------------------------------------X----------------------------------------------#

# In Einstein notation, repeated indices imply summation automatically

# Simple summation over j
A = np.array([[1, 2],
              [3, 4]])

x = np.array([5, 6])

y = np.einsum('ij,j->i', A, x)

print(y)


# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

s = np.einsum('i,i->', a, b)

print(s)


# Inner Product

A = np.random.rand(2, 3)
B = np.random.rand(3, 4)

C = np.einsum('ij,jk->ik', A, B)

print(C.shape)

# Outer Product
a = np.array([1, 2])
b = np.array([3, 4, 5])

M = np.einsum('i,j->ij', a, b)

print(M)
