import random
from args import args
import math
from Models import Generator
import torch
import torch.optim as optim
from torch.autograd import Variable
from loss import g_content_loss
import time
from tqdm import tqdm, trange
import numpy as np
import os
import scipy.io as scio
from utils import make_floor
import utils
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from torchvision.transforms import ToPILImage
import random

import torch
import os
from tqdm import trange
from torch.autograd import Variable
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def reset_grad(g_optimizer):

    g_optimizer.zero_grad()

def train(train_data_ir, train_data_vi):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    models_save_path = make_floor(os.getcwd(), args.save_model_dir)
    print(models_save_path)
    loss_save_path = make_floor(models_save_path,args.save_loss_dir)
    print(loss_save_path)

    G = Generator().cuda()

    g_content_criterion = g_content_loss().cuda()


    optimizerG = optim.Adam(G.parameters(), args.g_lr)

    # scheduler = CosineAnnealingLR(optimizerG, T_max=20, eta_min=0.0001)

    print("\nG_model : \n", G)


    tbar = trange(args.epochs)


    content_loss_lst = [] 
    all_intensity_loss_lst = []
    all_texture_loss_lst = []
    g_loss_lst = []


    all_content_loss = 0.
    all_intensity_loss = 0.
    all_texture_loss = 0.


    for epoch in tbar:
        print('Epoch %d.....' % epoch)

        G.train()
        # scheduler.step()

        batch_size=args.batch_size
        image_set_ir,image_set_vi,batches = utils.load_dataset(train_data_ir,train_data_vi, batch_size,num_imgs=None)

        count = 0


        for batch in range(batches):
            count +=1
            reset_grad(optimizerG)
            img_model = 'L'

            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]


            img_vi = utils.get_train_images_auto_vi(image_paths_vi, height=args.hight, width=args.width, mode=img_model)
            img_ir = utils.get_train_images_auto_ir(image_paths_ir, height=args.hight, width=args.width, mode=img_model)




            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)

            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()

            img_fusion = G(img_ir, img_vi)

            content_loss,  intensity_loss , texture_loss = g_content_criterion(img_ir, img_vi,img_fusion)  # models_4

            g_loss =content_loss

            all_intensity_loss += intensity_loss.item()
            all_texture_loss +=texture_loss.item()

            all_content_loss += content_loss.item()

            reset_grad(optimizerG)

            g_loss.backward()
            optimizerG.step()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tepoch {}:[{}/{}]\n " \
                       "\t content_loss:{:.6}\t g_loss:{:.6}"  \
                       "\t intensity_loss:{:.6}\t  texture_loss:{:.6}".format(
                    time.ctime(), epoch+1, count, batches,
                    all_content_loss / args.log_interval,(all_content_loss) / args.log_interval,
                    all_intensity_loss / args.log_interval,  all_texture_loss / args.log_interval
                )
                tbar.set_description(mesg)


                content_loss_lst.append(all_content_loss / args.log_interval)
                all_intensity_loss_lst.append(all_intensity_loss / args.log_interval)
                all_texture_loss_lst.append(all_texture_loss / args.log_interval)
                g_loss_lst.append((all_content_loss ) / args.log_interval)

                all_content_loss = 0.
                all_intensity_loss = 0
                all_texture_loss = 0

        if (epoch+1) % args.log_iter == 0:
            # SAVE MODELS
            G.eval()
            G.cuda()
            G_save_model_filename = "G_Epoch_" + str(epoch) + ".model"
            G_model_path = os.path.join(models_save_path,G_save_model_filename)
            torch.save(G.state_dict(), G_model_path)

            # SAVE LOSS DATA

            # content_loss
            content_loss_part = np.array(content_loss_lst)
            loss_filename_path = "content_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'content_loss_part': content_loss_part})

            all_intensity_loss_part = np.array(all_intensity_loss_lst)
            loss_filename_path = "all_intensity_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'all_intensity_loss_part': all_intensity_loss_part})

            all_texture_loss_part = np.array(all_texture_loss_lst)
            loss_filename_path = "all_texture_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'all_texture_loss_part': all_texture_loss_part})

            # g_loss
            g_loss_part = np.array(g_loss_lst)
            loss_filename_path = "g_loss_epoch_" + str(epoch) + ".mat"
            save_loss_path = os.path.join(loss_save_path, loss_filename_path)
            scio.savemat(save_loss_path, {'g_loss_part': g_loss_part})

    # SAVE LOSS DATA

    # content_loss
    content_loss_total = np.array(content_loss_lst)
    loss_filename_path = "content_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'content_loss_total': content_loss_total})

    # all_intensity_loss
    all_intensity_loss_total = np.array(all_intensity_loss_lst)
    loss_filename_path = "all_intensity_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'all_intensity_loss_total': all_intensity_loss_total})

    # all_texture_loss
    all_texture_loss_total = np.array(all_texture_loss_lst)
    loss_filename_path = "all_texture_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'all_texture_loss_total': all_texture_loss_total})

    # g_loss
    g_loss_total = np.array(g_loss_lst)
    loss_filename_path = "g_loss_total_epoch_" + str(epoch) + ".mat"
    save_loss_path = os.path.join(loss_save_path, loss_filename_path)
    scio.savemat(save_loss_path, {'g_loss_total': g_loss_total})

    # SAVE MODELS
    G.eval()
    G.cuda()

    G_save_model_filename = "Final_G_Epoch_" + str(epoch) + ".model"
    G_model_path = os.path.join(models_save_path, G_save_model_filename)
    torch.save(G.state_dict(), G_model_path)

    print("\nDone, trained Final_G_model saved at", G_model_path)





