import numpy as np, time, random, csv
import torch, ast, pandas as pd, copy, itertools as it, os, torch.nn as nn
from torchvision import transforms
import torchvision
import scipy.io as sio
from tqdm import tqdm
from PIL import Image
from skimage import io, transform
import itertools as it, copy
import network as net
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils import load_table, load_features


def exist_frame(data_path,v,f):
    return os.path.exists('%s%s/%06d.jpg'%(data_path,v,f))

#data_path = opt.Paths['img']
#table = load_table(opt.Paths['detections'],asDict=False)#np.load(opt.Paths['dic'], encoding="latin1").item()
#info = {'videos':table['videos'].values,
             #'frames':table['frames'].values}
#### Removing missing crops
##detecions_iterator = tqdm(zip(info['videos'],info['frames']),desc='Validate detections')
##selection = [exist_frame(data_path,v,f) for v, f in detecions_iterator]
##selection = np.array(selection)

##info['videos'] = info['videos'][selection]
##info['frames'] = info['frames'][selection]
#### Determine length of our training set
#uni_videos = np.unique(info['videos'])
##if opt.Training['size'] is None:
    ##length = len(info['frames'])
##else:
#per_video = int(opt.Training['size']/len(uni_videos))
##length = min(opt.Training['size'],len(info['frames']))
#selection = [np.where(info['videos']==v)[0][:per_video] 
                #for v in uni_videos]#np.random.permutation(len(info['frames']))
#selection = np.concatenate(selection)
#info['videos'] = info['videos'][selection]
#info['frames'] = info['frames'][selection]
#length = len(info['frames'])


#data_iter = tqdm(np.unique(info['videos']),position=2)
#data_iter.set_description('Load Posture Representation')

#fc6 = []
#for i, v in enumerate(data_iter):
    #frames = info['frames'][info['videos']==v]
    #fc_name       = opt.Paths['fc6'] + v + '.npz'
    #features_file = np.load(fc_name)
    #selection = [f for f, frame in enumerate(features_file['frames']) if frame in frames]
    #fc6.append(features_file['fc6'][selection])

#fc6 = np.concatenate(fc6,0)
#assert fc6.shape[0]==info['videos'].shape[0],'Wrong number of features loaded'
# 
# idx = np.random.randint(len(info['videos']))
# video, frame = info['videos'][idx], info['frames'][idx]
# frames_sel = info['videos']==video
# frames = info['frames'][frames_sel]
# iframe = np.where(frames==frame)[0][0]
# ### Load original image
# img_name            = '%s%s/%06d.jpg'%(data_path,video,frame)
# image_orig          = io.imread(img_name)
# 
# rand_inter          = np.random.randint(5,8,1)[0]
# if (np.random.rand()<0.5 and iframe>5) or iframe>len(frames)-5:
#     rand_inter *= -1
# 
# # if int(idx%200) > 100:
# #     rand_inter      *= -1
# img_name            = '%s%s/%06d.jpg'%(data_path,video,frames[iframe+rand_inter])
# image_inter         = io.imread(img_name)
# 
# rand                = int(rand_inter/2)
# img_name            = '%s%s/%06d.jpg'%(data_path,video,frames[iframe+rand])
# image_inter_truth   = io.imread(img_name)
# 
# ### Load random images
# rand            = np.random.randint(0,len(frames),1)[0]
# img_name        = '%s%s/%06d.jpg'%(data_path,video,frames[rand])
# image_rand1     = io.imread(img_name)
# 
# rand            = np.random.randint(0,len(frames),1)[0]
# img_name        = '%s%s/%06d.jpg'%(data_path,video,frames[rand])
# image_rand2     = io.imread(img_name)

"""============================================"""
#dataset for dataloader
class dataset(torch.utils.data.Dataset):

    def __init__(self, opt):

        self.data_path = opt.Paths['img']
        table = load_table(opt.Paths['detections'],asDict=False)#np.load(opt.Paths['dic'], encoding="latin1").item()
        self.info = {'videos':table['videos'].values,
                     'frames':table['frames'].values}
        ### Removing missing crops
        # detecions_iterator = tqdm(zip(self.info['videos'],self.info['frames']),desc='Validate detections')
        # selection = [os.path.exists('%s%s/%06d.jpg'%(self.data_path,v,f))
        #              for i, (v, f) in enumerate(detecions_iterator)]
        
        # self.info['videos'] = self.info['videos'][selection]
        # self.info['frames'] = self.info['frames'][selection]
        ### Determine length of our training set
        uni_videos = np.unique(self.info['videos'])
        if opt.Training['size'] is None:
            self.length = len(self.info['frames'])
        else:
            per_video = int(opt.Training['size']/len(uni_videos))
            #self.length = min(opt.Training['size'],len(self.info['frames']))
            selection = [np.where(self.info['videos']==v)[0][:per_video] 
                            for v in uni_videos]#np.random.permutation(len(self.info['frames']))
            selection = np.concatenate(selection)
            self.info['videos'] = self.info['videos'][selection]
            self.info['frames'] = self.info['frames'][selection]
            self.length = len(self.info['frames'])
        
        ### Load fc6 features into the RAM
        #self.fc6 = np.zeros((self.length, opt.Network['feature_size']))

        data_iter = tqdm(uni_videos,position=2)
        data_iter.set_description('Load Posture Representation')
        
        self.fc6 = []
        for i, v in enumerate(data_iter):
            frames = self.info['frames'][self.info['videos']==v]
            features_file = np.load(opt.Paths['fc6'] + v + '.npz')
            selection = [np.where(features_file['frames']==frame)[0][0] 
                         for frame in frames 
                         if frame in features_file['frames']]
            self.fc6.append(features_file['fc6'][selection])
            # self.fc6.append(features_file['fc6'])
        
        self.fc6 = np.concatenate(self.fc6,0)
        assert len(self.fc6.shape)==2, 'Features not properly concatenated'
        assert self.fc6.shape[0]==self.info['videos'].shape[0],'Wrong number of features loaded: %d - %d'%(self.info['videos'].shape[0],self.fc6.shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video, frame = self.info['videos'][idx], self.info['frames'][idx]
        frames_sel = self.info['videos']==video
        frames = self.info['frames'][frames_sel]
        iframe = np.where(frames==frame)[0][0]
        F = len(frames)
        ### Load original image
        image_orig          = load_image(self.data_path,video,frame)

        rand_inter          = np.random.randint(2,min(4,F),1)[0]
        if (np.random.rand()<0.5 and iframe>=rand_inter) or iframe>F-rand_inter-1:
            rand_inter *= -1
        # if int(idx%200) > 100:
        #     rand_inter      *= -1
        image_inter         = load_image(self.data_path,video,frames[iframe+rand_inter])
        
        rand                = int(rand_inter/2)
        image_inter_truth   = load_image(self.data_path,video,frames[iframe+rand])
                    
        ### Load random images
        rand            = np.random.randint(0,F,1)[0]
        image_rand1     = load_image(self.data_path,video,frames[rand])

        rand            = np.random.randint(0,F,1)[0]
        image_rand2     = load_image(self.data_path,video,frames[rand])

        ### Downsample images
        image_orig          = transform.resize(image_orig, (128, 128), mode='constant').transpose((2, 0, 1))
        image_inter         = transform.resize(image_inter, (128, 128), mode='constant').transpose((2, 0, 1))
        image_inter_truth   = transform.resize(image_inter_truth, (128, 128), mode='constant').transpose((2, 0, 1))
        image_rand1         = transform.resize(image_rand1, (128, 128), mode='constant').transpose((2, 0, 1))
        image_rand2         = transform.resize(image_rand2, (128, 128), mode='constant').transpose((2, 0, 1))

        sample = {'image_orig': image_orig,'image_inter': image_inter, 
                  'image_inter_truth': image_inter_truth,'image_rand1':image_rand1, 
                  'image_rand2':image_rand2, 'fc6': self.fc6[idx],
                  'fc6_inter': self.fc6[frames_sel][iframe+rand_inter]}

        return sample

def load_image(path,v,f):
    name = '%s%s/%06d.jpg'%(path,v,f)
    if not os.path.exists(name): name = '%s%s/%d.jpg'%(path,v,f)
    im = io.imread(name)
    return im
    

"""======================================================="""
### Function to extract setup info from text file ###
def extract_setup_info(opt):

    baseline_setup = pd.read_table(opt.Paths['network_base_setup_file'], header=None)
    baseline_setup = [x for x in baseline_setup[0] if '=' not in x]
    sub_setups     = [x.split('#')[-1] for x in np.array(baseline_setup) if '#' in x]
    vals           = [x for x in np.array(baseline_setup)]
    set_idxs       = [i for i,x in enumerate(np.array(baseline_setup)) if '#' in x]+[len(vals)]
    settings = {}
    for i in range(len(set_idxs)-1):
        settings[sub_setups[i]] = [[y.replace(" ","") for y in x.split(':')] for x in vals[set_idxs[i]+1:set_idxs[i+1]]]

    d_opt = vars(opt)
    for key in settings.keys():
        d_opt[key] = {subkey:ast.literal_eval(x) for subkey,x in settings[key]}

    if opt.Paths['network_variation_setup_file'] == '':
        return [opt]


    variation_setup = pd.read_table(opt.Paths['network_variation_setup_file'], header=None)
    variation_setup = [x for x in variation_setup[0] if '=' not in x]
    sub_setups      = [x.split('#')[-1] for x in np.array(variation_setup) if '#' in x]
    vals            = [x for x in np.array(variation_setup)]
    set_idxs        = [i for i,x in enumerate(np.array(variation_setup)) if '#' in x]+[len(vals)]
    settings = {}
    for i in range(len(set_idxs)-1):
        settings[sub_setups[i]] = []
        for x in vals[set_idxs[i]+1:set_idxs[i+1]]:
            y = x.split(':')
            settings[sub_setups[i]].append([[y[0].replace(" ","")], ast.literal_eval(y[1].replace(" ",""))])
        settings

    all_c = []
    for key in settings.keys():
        sub_c = []
        for s_i in range(len(settings[key])):
            sub_c.append([[key]+list(x) for x in list(it.product(*settings[key][s_i]))])
        all_c.extend(sub_c)


    setup_collection = []
    training_options = list(it.product(*all_c))
    for variation in training_options:
        base_opt   = copy.deepcopy(opt)
        base_d_opt = vars(base_opt)
        for i,sub_variation in enumerate(variation):
            base_d_opt[sub_variation[0]][sub_variation[1]] = sub_variation[2]
            base_d_opt['iter_idx'] = i
        setup_collection.append(base_opt)

    return setup_collection


"""==================================================="""
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


"""===================================================="""
class Base_Loss(nn.Module):
    def __init__(self, weights=None):
        super(Base_Loss, self).__init__()
        self.weights = weights.type(torch.cuda.FloatTensor) if weights is not None else None
        self.loss    = nn.BCELoss(weight=self.weights)

    def forward(self, inp, target):
        return self.loss(inp, target)

"""===================================================="""
class Loss(nn.Module):
    def __init__(self, opt):
        super(Loss, self).__init__()
        self.loss       = nn.MSELoss(size_average=False)
        self.bs         = opt.Training['bs']
        self.alpha      = opt.Training['alpha']
        self.beta       = opt.Training['beta']
        self.gamma      = opt.Training['gamma']
        self.delta      = opt.Training['delta']

    def forward(self, recon_x, x, x_inter, x_inter_truth, z_app, z_app2, z_pos, mu_app, mu_pos, logvar_app, logvar_pos, vgg, z_dim):

        ### Perceptual Vgg loss of the reconstructed image with original
        recp_loss = 0
        for idx in range(4):
            out_recon = vgg(recon_x)[idx]
            recp_loss += self.loss(out_recon, vgg(x)[idx])/(out_recon.shape[1]*out_recon.shape[2]*out_recon.shape[3])
        BCE = recp_loss

        ### Perceptual Vgg loss of the reconstructed interpolated image with original future image
        recp_loss = 0
        for idx in range(4):
            out_recon = vgg(x_inter)[idx]
            recp_loss += self.loss(out_recon, vgg(x_inter_truth)[idx])/(out_recon.shape[1]*out_recon.shape[2]*out_recon.shape[3])
        BCE_inter = recp_loss

        ### Appearance constraint
        z_app_original = torch.autograd.Variable(z_app2.data, requires_grad=False)
        APP = self.loss(z_app, z_app_original)/z_dim

        ### KL divergence for both latent spaces
        KLD_element = mu_pos.pow(2).add_(logvar_pos.exp()).mul_(-1).add_(1).add_(logvar_pos)
        KLD_POS = torch.sum(KLD_element).mul_(-0.5)

        KLD_element = mu_app.pow(2).add_(logvar_app.exp()).mul_(-1).add_(1).add_(logvar_app)
        KLD_APP = torch.sum(KLD_element).mul_(-0.5)
        loss_dic = [BCE.item()/self.bs,  KLD_APP.item()/self.bs, KLD_POS.item()/self.bs, APP.item()/self.bs, BCE_inter.item()/self.bs]

        if BCE.item() < 170:
            return (BCE + self.alpha*BCE_inter + self.beta*KLD_APP + self.gamma*KLD_POS + self.delta*APP), (BCE.item()), loss_dic
        else:
            return (BCE + self.beta*KLD_APP + self.gamma*KLD_POS + self.delta*APP), (BCE.item()), loss_dic

"""==================================================="""
def appearance_trafo(model, opt):

    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor()
            ])
        }

    dic             = np.load(opt.Paths['dic'], encoding="latin1").item()
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

"""==================================================="""
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

"""==================================================="""
def save_graph(network_output, savepath, savename):
    from graphviz import Digraph
    def make_dot(var, savename, params=None):
        """ Produces Graphviz representation of PyTorch autograd graph
        Blue nodes are the Variables that require grad, orange are Tensors
        saved for backward in torch.autograd.Function
        Args:
            var: output Variable
            params: dict of (name, Variable) to add names to node that
                require grad (TODO: make optional)
        """
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',shape='box',align='left',fontsize='12',ranksep='0.1',height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()

        def size_to_str(size):
            return '('+(', ').join(['%d' % v for v in size])+')'

        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                elif hasattr(var, 'variable'):
                    u = var.variable
                    name = param_map[id(u)] if params is not None else ''
                    node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')
                else:
                    dot.node(str(id(var)), str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)

        add_nodes(var.grad_fn)
        print("Saving...")
        dot.save(savename)
        return dot

    if not os.path.exists(savepath+"/Network_Graphs"):
        os.makedirs(savepath+"/Network_Graphs")
    viz_graph = make_dot(network_output, savepath+"/Network_Graphs"+"/"+savename)
    print("Creating pdf...")
    viz_graph.view()
