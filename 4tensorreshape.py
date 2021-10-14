import torch

x = torch.arange(9)
print(x)

x_3x3 = x.view(3,3)     # view work on contiguous memory block where reshape no need of memory block,it make a copy of element of matrix
print(x_3x3)
# print(x_3x3.shape)

x_3x3 = x.reshape(3,3)   # alternate way to reshape | view and reshape are very much similar
# print(x_3x3)
y = x_3x3.t()
# print(y.view(9))    # contiguous SUBSPACES ERROR
print(y.contiguous().view(9))       # It will resolve the contiguous subspaces error before line


x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2),dim=0).shape)
print(torch.cat((x1,x2),dim=1).shape)

z = x1.view(-1)     # Flattern the shape of matrix i.e 2x5 to 1x10 matrix in this example.
print(z.shape)

batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)

# If we want the dimension in (batch,5,2) format in the example
z = x.permute(0,2,1)    # it is same as transpose of matrix; transpose is the special case of permute
print(z.shape)

x = torch.arange(10)    # [10]
print(x.unsqueeze(0).shape)   # it will convert shape of vector to 1x10 matrix
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
print(x.shape)
z = x.squeeze(1)
print(z.shape)