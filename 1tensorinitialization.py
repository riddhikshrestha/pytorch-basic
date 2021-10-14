import torch


#################################
#      Initialization Tensor    #
#################################

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,device=device,requires_grad=True)

print(my_tensor)            # print tensor

print(my_tensor.dtype)      # print tensor data type

print(my_tensor.device)     # print device where tensor exist

print(my_tensor.shape)      # print tensor's shape

print(my_tensor.requires_grad)  # print to check is requires_grad is enable or not

# Other common initialization methods
x = torch.empty(size=(3,3))
print(x)
x = torch.zeros(3,3)
print(x)

x = torch.rand(3,3)
print(x)

x = torch.ones(3,3)
print(x)

x = torch.eye(5,5)  # Identity Matrix
print(x)

x = torch.arange(start=0,end=5,step=1)  
print(x)

x = torch.linspace(start=0.1,end=1,steps=10)  # start with 0.1 and end with 1 and its have 10 values between them
print(x)

x = torch.empty(size=(1,5)).normal_(mean=0,std=1) # initialization of 1x5 matraix with normal distribution
print(x)

x = torch.empty(size=(1,5)).uniform_(0,1)   # uniform distribution
print(x)

x = torch.diag(torch.ones(3))
print(x)


# How to inialize and convert tensors to other types(int,double,float)
tensor = torch.arange(4)   #By default step will be 1 and start will be 0
print(tensor)
print(tensor.bool())   # boolean T/F

print(tensor.short())   # int16
print(tensor.long())    # int64 (important and mostly used)
print(tensor.half())    # float16
print(tensor.float())   # float32 (important)
print(tensor.double())  # float64

# Array to Tensor and Vice-Versa
import numpy as np

np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

print(np_array)
print(tensor)
print(np_array_back)