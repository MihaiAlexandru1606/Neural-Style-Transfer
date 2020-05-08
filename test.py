import torch
from torch import cuda
from torchvision import models
import torch
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
from torch import tensor
from torch.nn.functional import mse_loss

t1 = tensor([1., 22., 33., 4., 23., 33.]).cuda()
t2 = tensor([1., 22., 12., 323., 13., 32133.]).cuda()
print(t1)

loss = torch.zeros(1).cuda()
loss += mse_loss(t1, t2).cuda()
loss *= 0.2
print(mse_loss(t1, t2))
print(loss)


new_lose = (t1 - t2).pow(2).sum() / (4 * 5 * 5)
new_lose2 = (t1 - t2).pow(2)

print(new_lose2)
print(new_lose)

# IMG_SIZE = 512
# print(cuda.is_available())
# print(type(cuda.current_device()))
#
# CUDA_DEVICE = torch.device("cuda:{}".format(cuda.current_device()))
# print(CUDA_DEVICE)
#
# vgg19 = models.vgg19(pretrained=True)
# vgg19.cuda(CUDA_DEVICE)
# for p in vgg19.parameters():
#     p.requires_grad = False
#
# print(type(vgg19.features))
#
# layers_index = {'0': 'conv1_1',
#                 '5': 'conv2_1',
#                 '10': 'conv3_1',
#                 '19': 'conv4_1',
#                 '21': 'conv4_2',
#                 '28': 'conv5_1'
#                 }
#
# features = {}
#
# for name, layer in vgg19.features._modules.items():
#     print(name, layer)
#     if name in layers_index:
#         features.update({layers_index[name]: layer})
#
# print(features)
#
# l = [1, 2, 3, 4, 5]
# d = {}
# from copy import deepcopy
# for i in range(10):
#     l.append(1)
#     d[str(i)] = deepcopy(l)
#
# print(d)
#
# for name, layer in vgg19.features():
#     print(name)

# for name, layer in vgg19._modules.items():
#     print(name)
#     if name == "features":
#         for n, l in layer._modules.items():
#             print(n)

# img = Image.open("cat.png").convert('RGB')
# img_transforms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# img = img_transforms(img)
# print(type(img))
# print(img.shape)
# print(torch.unsqueeze(img, 0).shape)
# print(torch.squeeze(torch.unsqueeze(img, 0)).shape)
# b = torch.tensor([1., 2.]).cuda()
# img = img.cuda()
# print(img.is_cuda)
# img = img.cpu()
# print(img.is_cuda)
# print(b.is_cuda)
# # print(vgg19.is_cuda)
