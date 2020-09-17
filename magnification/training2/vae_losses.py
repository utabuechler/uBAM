import torch, torch.nn as nn

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

