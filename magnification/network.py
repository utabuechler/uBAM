
### Libraries
import torch.nn as nn, torch
from torchvision import models
from collections import namedtuple

################################################################################################
### NOTE: Divide and put into channels? ########################################################
################################################################################################
class ConvNet(nn.Module):

    def __init__(self, batchsize=64, input_shape=(3,227,227), seq_len = 8, gpu=True):
        super(ConvNet, self).__init__()

        self.batchsize = batchsize
        self.seq_len = seq_len
        self.gpu = gpu
        self.hidden = self.init_hidden(batchsize)

        self.conv = nn.Sequential()
        self.conv.add_module('conv1',nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0))
        self.conv.add_module('relu1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1',nn.MaxPool2d(kernel_size=3, stride=2))
        #self.conv.add_module('LRN1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv.add_module('lrn1',nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2',nn.Conv2d(96, 256, kernel_size=5, padding=2,groups=2),)
        self.conv.add_module('relu2',nn.ReLU(inplace=True))
        self.conv.add_module('pool2',nn.MaxPool2d(kernel_size=3, stride=2))
        #self.conv.add_module('LRN2',LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv.add_module('lrn2',nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3',nn.ReLU(inplace=True))

        self.conv.add_module('conv4',nn.Conv2d(384, 384, kernel_size=3, padding=1,groups=2))
        self.conv.add_module('relu4',nn.ReLU(inplace=True))

        self.conv.add_module('conv5',nn.Conv2d(384, 256, kernel_size=3, padding=1,groups=2))
        self.conv.add_module('relu5',nn.ReLU(inplace=True))
        self.conv.add_module('pool5',nn.MaxPool2d(kernel_size=3, stride=2))


        n_size = self._get_conv_output(input_shape)

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6',nn.Linear(n_size, 4096))
        self.fc6.add_module('relu6',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6',nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(4096,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))

    def forward(self, x):
        self.batchsize = x.size(0)/self.seq_len#for variable batchsize
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        return x

    def load_weights(self,pretrained_dict):
        #model.load_state_dict(torch.load('mytraining.pt'))
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        x = self.conv(x)
        return x

    def init_hidden(self,batchsize=256):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.gpu:
            #pass
            return (torch.autograd.Variable(torch.zeros(1, batchsize, 1024),requires_grad = False).cuda(),
                    torch.autograd.Variable(torch.zeros(1, batchsize, 1024),requires_grad = False).cuda())
        else:
            return (torch.autograd.Variable(torch.zeros(1, batchsize, 1024),requires_grad = False),
                    torch.autograd.Variable(torch.zeros(1, batchsize, 1024),requires_grad = False))

class VAE_FC6(nn.Module):
    def __init__(self, dic):
        super(VAE_FC6, self).__init__()
        
        self.nc = 3
        self.ngf = 128
        self.ndf = 128
        self.lstm_size = dic['feature_size']
        self.latent_variable_size = dic['z_dim']
        #encoder1
        self.en_app = nn.Sequential(
                    nn.Conv2d(self.nc,self. ndf, 4, 2, 1),
                    nn.BatchNorm2d(self.ndf),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1),
                    nn.BatchNorm2d(self.ndf*2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1),
                    nn.BatchNorm2d(self.ndf*4),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1),
                    nn.BatchNorm2d(self.ndf*8),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.ndf*8, self.ndf*8, 4, 2, 1),
                    nn.BatchNorm2d(self.ndf*8),
                    nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Linear(self.ndf*8*4*4, self.latent_variable_size)
        self.fc2 = nn.Linear(self.ndf*8*4*4, self.latent_variable_size)

        # encoder2
        self.en_pos = nn.Sequential(
                    nn.Linear(self.lstm_size, int(self.lstm_size/4)),
                    nn.BatchNorm1d(int(self.lstm_size/4)),
                    nn.LeakyReLU(0.2),
        )

        self.fc3 = nn.Linear(int(self.lstm_size/4), self.latent_variable_size)
        self.fc4 = nn.Linear(int(self.lstm_size/4), self.latent_variable_size)

        # decoder
        self.de_app = nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(self.ngf*8*2, self.ngf*8, 3, 1),
                    nn.BatchNorm2d(self.ngf*8, 1.e-3),
                    nn.LeakyReLU(0.2),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(self.ngf*8, self.ngf*4, 3, 1),
                    nn.BatchNorm2d(self.ngf*4, 1.e-3),
                    nn.LeakyReLU(0.2),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(self.ngf*4, self.ngf*2, 3, 1),
                    nn.BatchNorm2d(self.ngf*2, 1.e-3),
                    nn.LeakyReLU(0.2),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(self.ngf*2, self.ngf, 3, 1),
                    nn.BatchNorm2d(self.ngf, 1.e-3),
                    nn.LeakyReLU(0.2),
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(self.ngf, self.nc, 3, 1),
                    nn.Sigmoid(),
        )
        self.d1 = nn.Linear(self.latent_variable_size*2, self.ngf*8*2*4*4)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode_app(self, x):
        h5 = self.en_app(x)
        h5 = h5.view(-1, self.ndf*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def encode_pos(self, x):
        h1 = self.en_pos(x)
        return self.fc3(h1), self.fc4(h1)

    def reparametrize(self, mu, logvar):

        if not self.training:
            return mu
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        return self.de_app(h1)

    def get_latent_var(self, x, fc):
        mu_app, logvar_app = self.encode_app(x.view(-1, self.nc, self.ndf, self.ngf))
        mu_pos, logvar_pos = self.encode_pos(fc)
        return self.reparametrize(mu_app, logvar_app), self.reparametrize(mu_pos, logvar_pos)

    def forward(self, x1, fc):
        mu_app, logvar_app = self.encode_app(x1.view(-1, self.nc, self.ndf, self.ngf))
        mu_pos, logvar_pos = self.encode_pos(fc)
        z_app = self.reparametrize(mu_app, logvar_app); z_pos = self.reparametrize(mu_pos, logvar_pos)
        z = torch.cat((z_app, z_pos), dim=1)
        img = self.decode(z)
        return img, z_app, z_pos, mu_app, mu_pos, logvar_app, logvar_pos

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
