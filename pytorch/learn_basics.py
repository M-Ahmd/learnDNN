import torch



cuda_ival = torch.cuda.is_available()

print(cuda_ival)

if(cuda_ival):
    tensor1 = torch.rand(3, 3).to('cuda')
    tensor2 = torch.rand(3, 3).to('cuda')
else:
    tensor1 = torch.rand(3, 3).to('cpu')
    tensor2 = torch.rand(3, 3).to('cpu')

print(tensor1)
print(tensor2)

result = tensor1 @ tensor2
print(result)