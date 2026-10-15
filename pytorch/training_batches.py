import torch
import torch.nn as nn
class SimpleModel(nn.Module):
    def __init__(self, input, middle ,output):
        super().__init__()
        self.in_middle = nn.Linear(input, middle)
        self.mid_out = nn.Linear(middle, output)
    def forward(self, input):
        res1 = self.in_middle(input)
        layer1 = torch.relu(res1)
        res2 = self.mid_out(layer1)
        output = torch.relu(res2)
        return output
    
x = torch.rand(20, 1).to('cuda')
y = torch.rand(20, 1).to('cuda')

model = SimpleModel(input=1, middle=10, output=1)
model.to('cuda')
creditional = nn.MSELoss()

opt = torch.optim.SGD(model.parameters(), lr=0.01)

dataset = torch.utils.data.TensorDataset(x, y)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)

epochs = 1000
for epoch in range(epochs):
    for x_batch, y_batch in dataloader:    
        pred = model(x_batch)
        loss = creditional(pred, y_batch)
        loss.backward()
        opt.step()
        opt.zero_grad()

