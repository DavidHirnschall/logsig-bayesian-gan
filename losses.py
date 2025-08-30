
from utils import *
from networks import *

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions doesn't compute the gradient w.r.t. targets" \
        "mark variables not requiring gradients"


class ComplementCrossEntropyLoss(torch.nn.Module):
    def __init__(self, except_index=None, weight=None, ignore_index=-100, size_average=True, reduce=True, device="cpu"):
        super(ComplementCrossEntropyLoss, self).__init__()
        self.except_index = except_index
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.reduce = reduce
        self.device = device

    def forward(self, input, target=None):
        # Use target if not None, else use self.except_index
        if target is not None:
            _assert_no_grad(target)
        else:
            assert self.except_index is not None
            target = torch.autograd.Variable(torch.LongTensor(input.data.shape[0]).fill_(self.except_index).to(self.device))
        result = torch.nn.functional.nll_loss(
        torch.log(1. - torch.nn.functional.softmax(input) + 1e-4), 
        target, weight=self.weight, 
        size_average=self.size_average, 
        ignore_index=self.ignore_index)
        return result
    
class NoiseLoss(torch.nn.Module):
  # need the scale for noise standard deviation
  # scale = noise  std
    def __init__(self, params, scale=None, observed=None, device = "cpu"):
        super(NoiseLoss, self).__init__()
        
        self.noises = []
        for param in params:
            noise = torch.normal(0, torch.ones_like(param)).to(device)
            noise.requires_grad = True
            self.noises.append(noise)
        if scale is not None:
            self.scale = scale
        else:
            self.scale = 1.
        self.observed = observed
        self.device = device

    def forward(self, params, scale=None, observed=None, seed=None, pr=False):
        if seed!=None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        # scale should be sqrt(2*alpha/eta), where eta is the learning rate and alpha is the strength of drag term
        if scale is None:
            scale = self.scale
        if observed is None:
            observed = self.observed

        assert scale is not None, "Please provide scale"
        noise_loss = 0.0
        
        if pr:
            print("noise:")
        for noise, var in zip(self.noises, params):
          # This is scale * z^T*v. Derivative wrt. v is scale*z. Noise ~ N(0,1)
            if pr:
                print(scale*torch.sum(Variable(noise)*var))
            noise_loss = noise_loss + scale*torch.sum(Variable(noise)*var)
        return noise_loss

class PriorLoss(torch.nn.Module):
  # negative log Gaussian prior
    def __init__(self, prior_std=1., observed=None, device = "cpu"):
        super(PriorLoss, self).__init__()
        self.observed = observed
        self.prior_std = prior_std
        self.device = device

    def forward(self, params, prior_std=1., observed=None, pr=False, g=1):
        if observed is None:
            observed = self.observed
        prior_loss = 0.0
        if pr:
            print("prior:")
        
            for name, var in params:
                if "linear" in name:
                    if prior_std=="recursive":
                        if len(var.size()) > 1:
                            fan_in = var.size()[0]
                            fan_out = var.size()[1]
                            sigma_prior_current = torch.tensor((g**2)*2/(fan_in+fan_out),device=self.device)
                            
                            prior_loss += (0.5*torch.sum(var*var)/sigma_prior_current + torch.log(torch.sqrt(sigma_prior_current)) )

        else:
            for var in params:
                if prior_std=="recursive":
                    if len(var.size()) > 1:
                        fan_in = var.size()[0]
                        fan_out = var.size()[1]
                        sigma_prior_current = torch.tensor((g**2)*2/(fan_in+fan_out),device=self.device)
                        prior_loss += (0.5*torch.sum(var*var)/sigma_prior_current + torch.log(torch.sqrt(sigma_prior_current)) )
                    
                else:
                    prior_loss += torch.sum(var*var/(prior_std*prior_std))
        prior_loss /= observed
        return prior_loss

   
def g_loss(D_fake):
    T = 1/math.sqrt(D_fake.shape[1]) * (D_fake[:,0]-torch.sum(D_fake[:,1:], dim=1))
    g_loss = torch.mean(T)
    return g_loss
 
def d_loss(D_fake, D_real):
    T_real = 1/math.sqrt(D_real.shape[1]) *(D_real[:,0]-torch.sum(D_real[:,1:], dim=1))
    T_fake = 1/math.sqrt(D_fake.shape[1]) *(D_fake[:,0]-torch.sum(D_fake[:,1:], dim=1))
    d_real_loss = torch.mean(T_real)
    d_fake_loss = -torch.mean(T_fake)
    
    d_loss = d_real_loss + d_fake_loss
    return d_loss, d_real_loss, d_fake_loss

def misclass(out, target):
    pred = torch.argmax(out, dim=1)
    m = sum(pred!=target)/(out.shape[0])
    return m

"""def F1_score(out, target, device="cpu", n_classes=None):
    if n_classes != None:
        if out.shape[1] > n_classes:
            out = out[:,1:]
        pred = torch.argmax(out, dim=1)
        num_classes = n_classes
    else:
        if out.shape[1] > 2: 
            out = out[:,1:]
        pred = torch.argmax(out, dim=1)
        num_classes=out.shape[1]
    return F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)(pred, target)
"""

def wauc(y_test, y_scores, classes=[0, 1]):
    
    y_test_bin = np.eye(len(classes))[y_test].astype(int)
    # Compute AUC for each class
    auc_per_class = []
    for i in range(y_test_bin.shape[1]):
        auc_w = roc_auc_score(y_test_bin[:, i], y_scores[:, i])
        auc_per_class.append(auc_w)
        
    # Compute weights (proportion of each class in y_true)
    class_counts = np.bincount(y_test)
    class_weights = class_counts / len(y_test)

    # Compute weighted AUC
    weighted_auc = np.sum(np.array(auc_per_class) * class_weights)
    return weighted_auc

