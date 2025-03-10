import torch
import torch.nn as nn
import re

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class ComplexLoss(nn.Module):

    def __init__(
            self,
            gamma=2,
    ):
        super().__init__()
        self.loss_foc = FocalLoss(gamma)
        self.factors = {0: ['hazy', 'blurry', 'clear'],
        1: ['extreme brightness', "normal brightness"],
        2: ['uniform stain', 'uneven stain'],
        3: ['purple', 'red', 'blue-purple'],
        4: ['small', 'medium', 'large']}

        sentence_format = "A {bl} WSI with {br}, {st}, a {c} coloring style, and showing {si}-sized tumor, is {adj} to classify"
        pattern = re.sub(r"\{.*?\}", r"(.+)", sentence_format)
        self.pattern = "^" + pattern + "$"

    def forward(self, text_similarities, targets):
        losses = []
        for s,target in enumerate(targets):
            match = re.match(self.pattern, target)
            extracted_values = match.groups()

            target_index = (self.factors[i].index(v) for i,v in enumerate(extracted_values))

            blur_target,brightness_target,stain_target,color_target,size_target=target_index
            text_similarity = text_similarities[s]
            blur_logits = text_similarity.sum(5).sum(4).sum(3).sum(2)
            brightness_logits = text_similarity.sum(5).sum(4).sum(3).sum(1)
            stain_logits = text_similarity.sum(5).sum(4).sum(2).sum(1)
            color_logits = text_similarity.sum(5).sum(3).sum(2).sum(1)
            size_logits  = text_similarity.sum(4).sum(3).sum(2).sum(1)

            loss1 = self.loss_foc(blur_logits, blur_target)
            loss2 = self.loss_foc(brightness_logits, brightness_target)
            loss3 = self.loss_foc(stain_logits, stain_target)
            loss4 = self.loss_foc(color_logits, color_target)
            loss5 = self.loss_foc(size_logits, size_target)

            losses.append((loss1 + loss2 + loss3 + loss4 + loss5)/5)

        return losses.mean()


