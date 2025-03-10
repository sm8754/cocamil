import torch
import torch.nn.functional as F

class ClassLoss(torch.nn.Module):
    def __init__(self, scale, lambda_g, lambda_js=0.1):
        super(ClassLoss, self).__init__()

        self.scale = scale
        self.lambda_g = lambda_g
        self.lambda_js = lambda_js

    def forward(self, cos_theta, cos_theta_m, target, score, js_div=None):
        loss_g = score ** 2

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss = F.cross_entropy(output, target, reduction='mean')

        if js_div is not None:
            loss_js = torch.mean(js_div)
            loss += self.lambda_js * loss_js

        return loss.mean() + self.lambda_g * loss_g.mean()