import torch


x = torch.arange(3,dtype=float,requires_grad=True)

print(x)

y = x + 2   # Addition function

print(y)    # AddBackward

z = y * y * 2   # Multiplication function

print(z)    # MulBackward

# z = z.mean()    # Mean function

# print(z)        # MeanBackward      
# z.backward()        # dz/dx

v = torch.tensor([0.1,1.0,0.001],dtype=torch.float32)  # we have to pass vector to match size of z; because it is not in scalar; Note to rememeber most of the time the last calculated value will be some scalar value so no need to do this operation.
# If the last calculate value is not sacalar we need to pass vector to make it scalar value

z.backward(v)  # dz/dx
print(x.grad)   # Vector Jacobian Product   

# --------------------------- #
# Prevent from AutoGrad
## x.requires_grad_(False)
## x.detach()
## with torch.no_grad():
# --------------------------- #

a = torch.randn(3,requires_grad=True)
print(a)

# a.requires_grad_(False)  # Function with _ means inplace modification of function
# b = a.detach()
# print(b)
with torch.no_grad():
    y = a + 2
    print(y)


# Examples
weights = torch.ones(4,requires_grad=True)

for epoch in range(3):

    model_output = (weights*3).sum()
    # print(model_output)
    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()  # before we want to do next operation or next iteration in our optimization steps, we must empty our gradients.

