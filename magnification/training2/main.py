import numpy as np, os, sys, pandas as pd, ast, imp, time, gc
from datetime import datetime
from tqdm import tqdm, trange
import argparse

from PIL import Image
import torch, torch.nn as nn, pickle as pkl
import torchvision
from torchvision import transforms

sys.path.append('./magnification/')
import network as net

sys.path.append('./magnification/training2/')
import vae_auxiliaries as aux
from vae_dataloader import dataset
from vae_losses import Loss

import config_pytorch as cfg
#import config_pytorch_human as cfg

"""=================================================================================="""
"""=================================================================================="""
def trainer(network, Vgg, epoch, data_loader, Metrics, optimizer, loss_func):

    _ = network.train()

    start_time = time.time()

    epoch_coll_acc, epoch_coll_loss  = [],[]

    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)

    for batch_idx, file_dict in enumerate(data_iter):

        iter_start_time = time.time()

        img_original    = torch.autograd.Variable(file_dict["image_orig"]).type(torch.FloatTensor).cuda()
        image_inter     = torch.autograd.Variable(file_dict["image_inter"]).type(torch.FloatTensor).cuda()
        image_inter_t   = torch.autograd.Variable(file_dict["image_inter_truth"]).type(torch.FloatTensor).cuda()
        img_rand1       = torch.autograd.Variable(file_dict["image_rand1"]).type(torch.FloatTensor).cuda()
        img_rand2       = torch.autograd.Variable(file_dict["image_rand2"]).type(torch.FloatTensor).cuda()
        fc6             = torch.autograd.Variable(file_dict["fc6"]).type(torch.FloatTensor).cuda()
        fc6_inter       = torch.autograd.Variable(file_dict["fc6_inter"]).type(torch.FloatTensor).cuda()

        #--- Run Training ---
        z_app2, _           = network.get_latent_var(img_rand1, fc6)
        img_recon, z_app, z_pos, mu_app, mu_pos, logvar_app, logvar_pos = network(img_rand2, fc6)
        latent_orig         = network.get_latent_var(img_original, fc6)
        latent_fut          = network.get_latent_var(image_inter, fc6_inter)
        img_inter_recon     = network.decode(torch.cat((latent_orig[0], (latent_orig[1]+latent_fut[1])/2), dim=1))

        ### BASE LOSS
        loss, acc, loss_dic = loss_func(img_recon, img_original, img_inter_recon, image_inter_t, z_app, z_app2, z_pos, mu_app, mu_pos, logvar_app, logvar_pos, Vgg, opt.Network['z_dim'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Get Scores
        epoch_coll_acc.append(acc)
        epoch_coll_loss.append(loss.item())

        if batch_idx%opt.Training['verbose_idx']==0 and batch_idx or batch_idx == 5:
            inp_string = 'Epoch {} || Loss: {} | Acc: {}'.format(epoch, np.round(np.mean(epoch_coll_loss)/opt.Training['bs'],4), np.round(np.mean(epoch_coll_acc)/opt.Training['bs'],4))
            data_iter.set_description(inp_string)

    ### Empty GPU cache
    torch.cuda.empty_cache()

    Metrics['Train Loss'].append(np.round(np.mean(epoch_coll_loss),4))
    Metrics['Train Acc'].append(np.round(np.sum(epoch_coll_acc)/len(epoch_coll_acc),4))
    return loss_dic, img_original, img_recon, image_inter_t, img_inter_recon

"""=================================================================================="""
"""=================================================================================="""
def main(opt):
    """================== /export/home/mdorkenw/data/human_protein/=========================="""
    ### Load Network
    imp.reload(net)
    VAE         = net.VAE_FC6(opt.Network).cuda()
    Vgg         = net.Vgg16().cuda()
    print('Network initialized')
    
    ### Set Optimizer
    loss_func   = Loss(opt)
    optimizer   = torch.optim.Adam(filter(lambda p: p.requires_grad, VAE.parameters()), lr=opt.Training['lr'], weight_decay=opt.Training['weight_decay'])
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.Training['step_size'], gamma=opt.Training['gamma2'])

    """============================================"""
    ### Set Dataloader
    train_dataset     = dataset(opt)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers = opt.Training['kernels'],
                                                    batch_size = opt.Training['bs'], shuffle=True)
    print('DataLoader initialized')
    """============================================"""
    ### Set Logging Files ###
    dt = datetime.now()
    dt = '{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    opt.Training['name'] = 'VAE_'+dt
    if opt.Training['savename']!="":
        opt.Training['name'] += '_'+opt.Training['savename']

    save_path = opt.Paths['save_path']+"/"+opt.Training['name']

    #Make the saving directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        count = 1
        while os.path.exists(save_path):
            count        += 1
            svn          = opt.name+"_"+str(count)
            save_path    = opt.save_path+"/"+svn
        opt.name = svn
        os.makedirs(save_path)
    opt.Paths['save_path'] = save_path

    #Generate save string
    save_str = aux.gimme_save_string(opt)

    ### Save rudimentary info parameters to text-file and pkl.
    with open(opt.Paths['save_path']+'/Parameter_Info.txt','w') as f:
        f.write(save_str)
    pkl.dump(opt,open(opt.Paths['save_path']+"/hypa.pkl","wb"))
    
    
    """============================================"""
    logging_keys    = ["Train Acc", "Train Loss"]
    Metrics         = {key:[] for key in logging_keys}
    Timer           = {key:[] for key in ["Train"]}

    ### Setting up CSV writers
    full_log  = aux.CSVlogger(save_path+"/log_per_epoch.csv", ["Epoch", "Time", "Training Loss", "Training Acc", "Learning Rate", "Loss_Recon", "Loss_KLD_APP", "Loss_KLD_Pos", "Loss_APP", "Loss_Inter"])
    print('Logger initialized')

    """============================================="""
    epoch_iterator = tqdm(range(opt.Training['n_epochs']),position=1)
    epoch_iterator.set_description('Train epoch')
    best_val_acc   = 2000

    for epoch in epoch_iterator:
        epoch_time = time.time()

        scheduler.step()

        ###### Training ########
        epoch_iterator.set_description("Training with lr={}".format(np.round(scheduler.get_lr(),8)))
        loss_dic, img_orig, img_recon, img_inter_orig, img_inter = trainer(VAE, Vgg, epoch, train_data_loader, Metrics, optimizer, loss_func)

        ###### SAVE CHECKPOINTS ########
        save_dict = {'epoch': epoch+1, 'state_dict':VAE.state_dict(),
                     'optim_state_dict':optimizer.state_dict()}

        # Best Validation Score
        if Metrics['Train Acc'][-1] < best_val_acc:
            torch.save(save_dict, opt.Paths['save_path']+'/checkpoint_lowest_loss.pth.tar')
            best_val_acc = Metrics['Train Acc'][-1]

        # After Epoch
        torch.save(save_dict, opt.Paths['save_path']+'/checkpoint.pth.tar')

        ### Save images
        torchvision.utils.save_image(img_recon.cpu().data,  opt.Paths['save_path']+ '/{:03d}_recon.png'.format(epoch+1))
        torchvision.utils.save_image(img_orig.cpu().data,  opt.Paths['save_path']+ '/{:03d}_original.png'.format(epoch+1))
        torchvision.utils.save_image(img_inter.cpu().data,  opt.Paths['save_path']+ '/{:03d}_inter_recon.png'.format(epoch+1))
        torchvision.utils.save_image(img_inter_orig.cpu().data,  opt.Paths['save_path']+ '/{:03d}_inter_original.png'.format(epoch+1))

        ###### Logging Epoch Data ######
        full_log.write([epoch, time.time()-epoch_time, 
                        np.mean(Metrics["Train Loss"][epoch]),
                        np.mean(Metrics["Train Acc"][epoch]), 
                        scheduler.get_lr()[0], loss_dic])

        ###### Generating Summary Plots #######
        sum_title = 'Max Train Acc: {0:2.3f}'.format(np.min([np.mean(Metrics["Train Acc"][ep]) for ep in range(epoch+1)]))
        aux.progress_plotter(np.arange(epoch+1), \
                             [np.mean(Metrics["Train Loss"][ep]) for ep in range(epoch+1)],[np.mean(Metrics["Train Acc"][ep]) for ep in range(epoch+1)],
                             save_path+'/training_results.png', sum_title)
        _ = gc.collect()

    aux.appearance_trafo(VAE, opt)

"""============================================"""
### Start Training ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int,default=0,
                        help="ID of the GPU device to use for training.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    ### GET TRAINING SETUPs ###
    #Read network and training setup from text file.
    opt = argparse.Namespace()
    opt.Paths = {'detections':cfg.detection_file,
                 'img':cfg.crops_path,
                 'frame_path':cfg.frames_path,
                 'fc6':cfg.features_path+'/fc6_fc7_feat/',
                 'save_path':cfg.vae_checkpoint_path}
    opt.Network = {'z_dim':cfg.vae_encode_dim,'feature_size':4096}
    opt.Training = {'n_epochs':cfg.vae_n_epochs,
                    'lr':cfg.vae_lr,
                    'bs':cfg.vae_bs,
                    'verbose_idx':cfg.vae_verbose_idx,
                    'weight_decay':cfg.vae_weight_decay,
                    'step_size':cfg.vae_step_size,
                    'gamma2':cfg.vae_gamma2,
                    'kernels':cfg.vae_kernels,
                    'alpha':cfg.vae_alpha,
                    'beta':cfg.vae_beta,
                    'gamma':cfg.vae_gamma,
                    'delta':cfg.vae_delta,
                    'size':cfg.vae_size,
                    'savename':cfg.vae_savename}
    print('Starting MAIN')
    print('BatchSize: ',opt.Training['bs'])
    main(opt)
