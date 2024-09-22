
import numpy as np
import os
from args import args
import scipy.misc
import cv2
import torch
from PIL import Image
from os import listdir
from os.path import join
from imageio import imread, imsave
import imageio
from torchvision import transforms

def load_dataset(ir_imgs_path,vi_imgs_path,BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(ir_imgs_path)
    ir_imgs_path = ir_imgs_path[:num_imgs]
    vi_imgs_path = vi_imgs_path[:num_imgs]

    # random
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        ir_imgs_path =  ir_imgs_path[:-mod]
        vi_imgs_path = vi_imgs_path[:-mod]

    batches = int(len(ir_imgs_path) // BATCH_SIZE)
    return ir_imgs_path,vi_imgs_path, batches

def make_floor(path1,path2):
    path = os.path.join(path1,path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path


def get_train_images_auto_ir(paths, height=args.hight, width=args.width, mode='L'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image_ir(path, height, width, mode='L')
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)


    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = images / 255
    #print(images.shape)
    return images

def get_train_images_auto_vi(paths, height=args.hight, width=args.width, mode='L'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image_vi(path, height, width, mode='L')
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            #print(image.shape)
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)


    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = images / 255

    return images


def prepare_data(directory):
    directory = os.path.join(os.getcwd(), directory)
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def save_feat(index,C,ir_atten_feat,vi_atten_feat,result_path):
    ir_atten_feat = (ir_atten_feat  * 255)#.byte()
    vi_atten_feat = (vi_atten_feat  * 255)#.byte()

    ir_feat_path = make_floor(result_path, "ir_feat")
    index_irfeat_path = make_floor(ir_feat_path, str(index))

    vi_feat_path = make_floor(result_path, "vi_feat")
    index_vifeat_path = make_floor(vi_feat_path, str(index))

    for c in range(C):
        ir_temp = ir_atten_feat[:, c, :, :].squeeze()
        vi_temp = vi_atten_feat[:, c, :, :].squeeze()

        feat_ir = (ir_temp.cpu().clamp(0, 255).data.numpy()).astype(np.uint8)
        feat_vi = (vi_temp.cpu().clamp(0, 255).data.numpy()).astype(np.uint8)


        ir_feat_filenames = 'ir_feat_C' + str(c) + '.png'
        ir_atten_path = index_irfeat_path + '/' + ir_feat_filenames
        imsave(ir_atten_path, feat_ir)

        vi_feat_filenames = 'vi_feat_C' + str(c) + '.png'
        vi_atten_path = index_vifeat_path + '/' + vi_feat_filenames
        imsave(vi_atten_path, feat_vi)

def get_image_ir(path, height=args.hight, width=args.width, mode='L'):
    if mode == 'L':
        image = imageio.imread(path,mode='L')
        #image = (image - 127.5) / 127.5
        image = image
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    if height is not None and width is not None:
        image = cv2.resize(image, [height, width], interpolation=cv2.INTER_AREA)


    return image


def get_image_rgb(path, height, width, mode='RGB'):
    # 使用PIL打开图像
    image = Image.open(path)

    # 根据模式转换图像
    if mode == 'RGB':
        image = image.convert('RGB')
    elif mode == 'L':
        image = image.convert('L')
        # 如果是灰度图像，添加一个通道维度
        image = np.array(image, dtype=np.uint8)
        image = np.stack([image] * 3, axis=-1)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 将PIL图像转换为NumPy数组
    image = np.array(image)

    # 调整图像大小
    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return image

def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    # ori = ori *255
    cv2.imwrite(path, ori)

# def get_image_vi(path, height=args.hight, width=args.width, mode='RGB'):
#     if mode == 'L':
#         image = imageio.imread(path,mode='L')
#         #image = (image - 127.5) / 127.5
#         image = image
#     elif mode == 'RGB':
#         image = Image.open(path).convert('RGB')
#     if height is not None and width is not None:
#         image = cv2.resize(image, [height, width], interpolation=cv2.INTER_AREA)
#
#
#     return image

def get_image(path, height=128, width=128, mode='L'):
    global image
    if mode == 'L':
        image = cv2.imread(path, 0)
    elif mode == 'RGB':
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image

def get_image_vi(path, height=args.hight, width=args.width, mode='L'):
    if mode == 'L':
        image = imageio.imread(path, mode='L')
        #image = (image - 127.5) / 127.5
        image = image
        #print(image.shape)
    elif mode == 'RGB':
        image = np.array(Image.open(path).convert('RGB'))  # Convert PIL image to numpy array
    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return image



def get_test_images(paths, height=None, width=None, mode='L'):
    global image
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        w, h = image.shape[0], image.shape[1]
        w_s = 128 - w % 128
        h_s = 128 - h % 128
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,value=256)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = images/ 255
    return images

def get_test_images_vis(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image_vi(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images =images/255

    return images



# def save_images(path, data):
#     if data.shape[2] == 1:
#         data = data.reshape([data.shape[0], data.shape[1]])
#     imsave(path, data)

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images