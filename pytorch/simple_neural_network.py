import torch
import torch.nn as nn
#class that extend the origin class
#any model neural network must as __init__ && forward functions
class SimpleModel(nn.Module):
    def __init__(self, input, middle, output):
        super().__init__()
        self.weight1 = nn.Linear(input, middle)
        self.weight2 = nn.Linear(middle, output)
    def forward(self, input):
        res1 = self.weight1(input)
        result = torch.relu(res1)
        output = self.weight2(result)
        final = torch.relu(output)
        return final
device = ''
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


model = SimpleModel(input=20, middle=10, output=1)


data = torch.rand(20)
model.to(device)
x = data.to(device)
target = torch.rand(1).to(device)
start = 0
end = 100
loss = 32
credition = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for start in range(end):
    y = model(x)
    loss = credition(target, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(loss)


torch.save(model.state_dict(), 'file.pth')