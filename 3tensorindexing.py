################################################
#               Tensor Indexing                #
################################################
import torch

batch_size = 10
features = 25

x = torch.rand((batch_size,features))

print(x[0].shape)  # x[0,:]

print(x[:,0].shape)      # First features of all examples

print(x[2,0:10])  # 3rd exmaple in batch which contain element from [0,1,....9]

# fancy indexing
x = torch.arange(10)
indices = [2,5,8]
print(x[indices])

x = torch.rand((3,5))
print(x)
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols])   # first value is 2nd row,5th column AND second value is 1st row,1st column

# More Advanced Indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8 )])   # print all element, which is  x < 2 and x > 8  , It is UNION condition
print(x[(x < 2) & (x > 8)])    # which is impossible condition or there is no common element in the condition(INTERSECTION) to full fill so it will return empty list
print(x[x.remainder(2) == 0])  # Print all even element

# Useful operations
print(torch.where(x > 5, x, x*2))  # if x > 5 the element are remain same else x=x*2

print(torch.tensor([0,0,1,2,2,3,4,4]).unique())

x = torch.rand((5,5))
print(x.ndimension())   # check dimension of x

print(x.numel())   # count number of elements in x