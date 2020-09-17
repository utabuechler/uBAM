import numpy as np, random, csv, os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torchvision import transforms
import torchvision

"""======================================================="""
def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str


"""==================================================="""
class CSVlogger():
    def __init__(self, logname, header_names):
        self.header_names = header_names
        self.logname      = logname
        with open(logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(header_names)
    def write(self, inputs):
        with open(self.logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(inputs)


"""===================================================="""
def progress_plotter(x, train_loss, train_metric, savename='result.svg', title='No title'):
    plt.style.use('ggplot')
    f,ax = plt.subplots(1)
    ax.plot(x, train_loss,'b--',label='Training Loss')

    axt = ax.twinx()
    axt.plot(x, train_metric, 'b', label='Training Dice')

    ax.set_title(title)
    ax.legend(loc=0)
    axt.legend(loc=2)

    f.suptitle('Loss and Evaluation Metric Progression')
    f.set_size_inches(15,10)
    f.savefig(savename)
    plt.close()

"""==================================================="""
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

"""==================================================="""
def appearance_trafo(model, opt):

    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor()
            ])
        }

    dic             = np.load(opt.Paths['detections'], encoding="latin1").item()
    path_save       = opt.Paths['save_path'] + '/evaluation/'

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    model.eval()

    for i in range(15):
        rand = np.random.randint(0,opt.Training['size'],2)

        img_name    = opt.Paths['img'] + dic['videos'][rand[0]] + '/' + str(int(dic['frames'][rand[0]])) + '.jpg'
        image1      = data_transforms['train'](Image.open(img_name)).view(1,3,128,128).cuda()

        fc_name     = opt.Paths['fc6'] + dic['videos'][rand[0]] + '.npz'
        fc1         = torch.from_numpy(np.load(fc_name)['fc6'][int(dic['frames'][rand[0]]),:]).float().view(1,4096).cuda()

        img_name    = opt.Paths['img'] + dic['videos'][rand[1]] + '/' + str(int(dic['frames'][rand[1]])) + '.jpg'
        image2      = data_transforms['train'](Image.open(img_name)).view(1,3,128,128).cuda()

        fc_name     = opt.Paths['fc6'] + dic['videos'][rand[1]] + '.npz'
        fc2         = torch.from_numpy(np.load(fc_name)['fc6'][int(dic['frames'][rand[1]]),:]).float().view(1,4096).cuda()

        latent1     = model.get_latent_var(image1, fc1)
        latent2     = model.get_latent_var(image2, fc2)
        zero        = torch.zeros((1,opt.Training['z_dim'])).cuda()

        img_change  = model.decode(torch.cat((latent1[0], latent2[1]), dim=1))
        img_app     = model.decode(torch.cat((latent1[0], latent1[1]), dim=1))
        img_pos     = model.decode(torch.cat((latent2[0], latent2[1]), dim=1))

        torchvision.utils.save_image(img_change.cpu().data  ,  path_save + '/{:01d}_change.png'.format(i+1))
        torchvision.utils.save_image(img_app.cpu().data     ,  path_save + '/{:01d}_app.png'.format(i+1))
        torchvision.utils.save_image(img_pos.cpu().data     ,  path_save + '/{:01d}_pos.png'.format(i+1))
