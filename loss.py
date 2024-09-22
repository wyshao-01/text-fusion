from torch import nn
import torch


class g_content_loss(nn.Module):
    def __init__(self):
        super(g_content_loss, self).__init__()
        self.L2_loss = nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss(reduction="mean")
        self.gradient=gradient()

    def forward(self, img_ir, img_vi, img_fusion):
        lambda_2=1
        lambda_3=10
        image_vi_grad = self.gradient(img_vi)
        image_ir_grad = self.gradient(img_ir)
        image_fusion_grad = self.gradient(img_fusion)
        image_max_grad = torch.round((image_vi_grad + image_ir_grad) // (
                torch.abs(image_vi_grad + image_ir_grad) + 0.0000000001)) * torch.max(
            torch.abs(image_vi_grad), torch.abs(image_ir_grad))
        grad_loss = 15*self.L1_loss(image_fusion_grad, image_max_grad)

        image_vi = img_vi
        image_ir = img_ir
        image_max_int = torch.round((image_vi + image_ir) // (
                torch.abs(image_vi + image_ir) + 0.0000000001)) * torch.max(
            torch.abs(image_vi), torch.abs(image_ir))

        intensity_loss = 1*self.L1_loss(img_fusion, image_max_int)

        texture_loss = grad_loss



        content_loss = intensity_loss + texture_loss
        return content_loss,  intensity_loss , texture_loss




class gradient(nn.Module):
    def __init__(self,channels=1):
        super(gradient, self).__init__()
        laplacian_kernel = torch.tensor([[1/8,1/8,1/8],[1/8,-1,1/8],[1/8,1/8,1/8]]).float()

        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        return self.laplacian_filter(x) ** 2



# from torch import nn
# import torch
# import torch.nn.functional as F
# import cv2
# import numpy as np
#
# class g_content_loss(nn.Module):
#     def __init__(self):
#         super(g_content_loss, self).__init__()
#         self.sobelconv = Sobelxy()
#         self.mse_criterion = torch.nn.MSELoss()
#         self.L1_loss = torch.nn.L1Loss(reduction="mean")
#
#     def forward(self, image_ir,image_vis, generate_img):
#         image_y = image_vis
#         B, C, W, H = image_vis.shape
#
#         image_ir = image_ir.expand(B, C, W, H)
#         x_in_max = torch.max(image_y, image_ir)
#         loss_in =self.L1_loss(generate_img, x_in_max)
#
#         # Gradient
#         y_grad = self.sobelconv(image_y)
#         ir_grad = self.sobelconv(image_ir)
#         B, C, K, W, H = y_grad.shape
#         ir_grad = ir_grad.expand(B, C, K, W, H)
#         generate_img_grad = self.sobelconv(generate_img)
#         x_grad_joint = torch.maximum(y_grad, ir_grad)
#         loss_grad = self.L1_loss(generate_img_grad, x_grad_joint)
#
#         content_loss = loss_in + loss_grad
#         return content_loss,loss_in, loss_grad
#
# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                    [0, 0, 0],
#                    [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#
#     def forward(self, x):
#         b, c, w, h = x.shape
#         batch_list = []
#         for i in range(b):
#             tensor_list = []
#             for j in range(c):
#                 sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
#                 sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
#                 add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
#                 tensor_list.append(add_0)
#
#             batch_list.append(torch.stack(tensor_list, dim=1))
#
#         return torch.cat(batch_list, dim=0)