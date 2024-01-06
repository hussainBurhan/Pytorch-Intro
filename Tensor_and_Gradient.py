import numpy as np
import torch

t1 = torch.tensor([4.])
t2 = torch.tensor([1, 2, 3, 4])
t3 = torch.tensor([[1., 2], [3, 4], [5, 6]])
t4 = torch.tensor([[[1., 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]])

print('Tensor shapes')
print(t1.shape)
print(t2.shape)
print(t3.shape)
print(t4.shape)

print('Tensor data type')
print(t1.dtype)
print(t2.dtype)
print(t3.dtype)
print(t4.dtype)

print('Tensors')
print(f't1 : {t1}')
print(f't2 : {t2}')
print(f't3 : {t3}')
print(f't4 : {t4}')


x = torch.tensor([[2., 2],[8, 1]])
w = torch.tensor([[4., 5],[1, 2]], requires_grad=True)
b = torch.tensor([[4., 7],[3, 2]], requires_grad=True)

y = w*x + b

y.backward()

print(f'dy/dx: {x.grad}')
print(f'dy/dw: {w.grad}')
print(f'dy/db: {b.grad}')

x = np.array([[1, 2], [3, 4.]])

y = torch.from_numpy(x)
