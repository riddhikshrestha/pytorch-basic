# 1. Design model (input size, output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training loop
#   - forward pass : compute prediction
#   - backward pass : gradients
#   - update weights

import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

X_test = torch.tensor([5],dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples,n_features)

# Model Construction

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size,output_size)

class LinearRegression(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        #Define layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
# n_iters = 10
n_iters = 50

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    # Prediction = Forward pass
    y_pred = model(X)

    # Loss
    l = loss(Y,y_pred)

    # Gradients = backward pass
    l.backward()  # dl/dw

    # Update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()

    # if epoch % 1 == 0:
    if epoch % 2 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')