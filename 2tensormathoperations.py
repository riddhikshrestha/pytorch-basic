import torch

###############################
#   Tensor Math Operations    #
###############################

a = torch.tensor([1,2,3])
b = torch.tensor([1,5,6])

# Addition Operation
c = torch.empty(3)
torch.add(a,b,out=c)

c = torch.add(a,b)

c = a + b
print(c)

# Subtraction
c = a - b
print(c)

# Division
c = torch.true_divide(a,b)  #  Element wise division operation i.e. a/b
print(c)

# Inplace operation (operation on exact the variable not to the copied of variable) 
t = torch.zeros(3)
t.add_(a)          # Note: All the function which has 'fun_' means it is inplace operation in pytorch
t +=a           # another way to perform inplace operation, if we do, t = t + a; it is not same as t +=a; it will first make copy and perform operation
print(t)

# Exponentation
#c = a.pow_(2)        # Element wise
c = a ** 2
print(c)

# Simple Comparision
c = a > 0    # Element wise comparision
print(c)


# Matrix Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
#x3 = torch.mm(x1,x2)    # Output matrix will be of order 2x3
x3 = x1.mm(x2)
print(x3)

# Matrix exponentiation
matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)
print(matrix_exp)

# Element wise multiplication
c = a * b
print(c)

# Dot product i.e element wise multiplication and its sum
c = torch.dot(a,c)
print(c)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand((batch,n,m))
t2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(t1,t2)   # Size of resultant batch matrix will be of shape (batch,n,p) i.e of (32,10,30)
print(out_bmm)


# BoardCasting in PyTorch i.e to match matrix size for operation which is not exactly of same shape in following example
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
print(z)

z = x1 ** x2
print(z)

# Some useful tensor operations
sum_a = torch.sum(a,dim=0)
print(sum_a)            # e.g. 1+2+3 = 6

values, indices = torch.max(a,dim=0)            # a.max(dim=0)
print(values,indices)   
values, indices = torch.min(a,dim=0)
print(values,indices)

abs_a = torch.abs(a)
print(abs_a)

c = torch.argmax(a,dim=0)   # special function of torch.max which will return max's value index
print(c)

c = torch.argmin(a,dim=0)
print(c)

mean_a = torch.mean(a.float(),dim=0)
print(mean_a)

# Element wise vector or matrix comparision
c = torch.eq(a,b)   # Check equal element of a and b
print(c)


sorted_b,indices_b = torch.sort(b,dim=0,descending=True)
print(sorted_b,indices_b)

c = torch.clamp(a, min=0)  # It will clamp all values less than 0 to 0 and other values it will not touch
print(c)

x = torch.tensor([1,0,1,1,1],dtype=torch.bool)
# z = torch.any(x)
z = torch.all(x)  # False  because list x has one 0 value
print(z)
