import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
from optimizer import create_optimizer
from loss import create_loss
from utils.utils import cross_entropy_torch
from models import CoCaMIL,CLIP
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

class ModelInterface(pl.LightningModule):

    def __init__(self, model, loss_arg, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.model = self.instancialize(CoCaMIL.COCAMIL)
        self.class_loss = create_loss(loss_arg,class_loss)
        self.complex_loss = create_loss(loss_arg,complex_loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        self.textes = self.generate_text()
        
        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(num_classes = 2),
                                                     torchmetrics.F1(num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        self.shuffle = kargs['data'].data_shuffle
        self.count = 0


    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def generate_text(self):
        self.blur = ['hazy','blurry','clear']
        self.brightness = ['extreme brightness', "normal brightness"]
        self.stain = ['uniform stain', 'uneven stain']
        self.color = ['purple', 'red', 'blue-purple']
        self.size = ['small','medium','large']

        text = torch.cat([CLIP.tokenize(f'A {bl} WSI with {br}, {st}, a {c} coloring style, and showing {si}-sized tumor, is easy to classify')
                          for bl, br, st, c, si in product(self.blur,self.brightness,self.stain,self.color,self.size)]).to('cuda')

    def training_step(self, batch, batch_idx):
        data, label, target_text = batch
        results_dict = self.model(data=data, text=self.textes,clip=True)
        logits =  results_dict['cos_theta']

        class_loss = self.class_loss(logits,results_dict['cos_theta_m'],label,results_dict['complexity_score'])
        text_similarity = results_dict['text_similarity'].view(-1, len(self.blur), len(self.brightness), len(self.stain), len(self.color),len(self.size))
        complex_loss = self.complex_loss(text_similarity,target_text)

        total_loss = class_loss + 0.5 * complex_loss

        Y_hat = torch.argmax(logits, dim=1)
        for Y, Y_hat_ in zip(label, Y_hat):
            Y = int(Y)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat_.item() == Y)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('class_loss', class_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('complex_loss', complex_loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': total_loss}

    def training_epoch_end(self, training_step_outputs):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch, batch_idx):
        data, label, target_text = batch
        results_dict = self.model(data=data, text=self.textes,clip=True)
        logits = results_dict['cos_theta']

        class_loss = F.cross_entropy(logits, label)
        complex_loss = self.complex_loss(results_dict['text_similarity'],target_text)

        total_loss = class_loss + 0.5 * complex_loss

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(logits, dim=1)

        for Y, Y_hat_ in zip(label, Y_hat):
            Y = int(Y)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat_.item() == Y)

        return {
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'label': label,
            'val_loss': total_loss
        }


    def validation_epoch_end(self, validation_step_outputs):
        Y_prob = torch.cat([x['Y_prob'] for x in validation_step_outputs], dim=0)
        Y_hat = torch.cat([x['Y_hat'] for x in validation_step_outputs], dim=0)
        target = torch.cat([x['label'] for x in validation_step_outputs], dim=0)
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.valid_metrics(Y_hat, target), on_epoch=True, prog_bar=True)
        self.log('val_auc', self.AUROC(Y_prob, target), on_epoch=True, prog_bar=True)
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = 0.0
            else:
                acc = float(correct) / count
            print(f"Class {c}: Acc = {acc:.4f}, Correct = {correct}/{count}")
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]


    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        data, label, _ = batch
        results_dict = self.model(data=data, text=self.textes,clip=True)
        logits = results_dict['cos_theta']
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(logits, dim=1)

        #---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim = 0)
        
        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        print()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')

    def instancialize(self, Model, **other_args):
        state_dict = CLIP.get_state(self.hparams.model.clip_mode)
        args1 = {
            "n_classes": self.hparams.model.n_classes,
            "l_margin": self.hparams.loss.l_margin,
            "u_margin": self.hparams.loss.u_margin,
            "dims": self.hparams.loss.dims,
            "easy_margin": self.hparams.loss.easy_margin,
            "context_length": state_dict["positional_embedding"].shape[0],
            "embed_dim": state_dict["text_projection"].shape[1],
            "vocab_size": state_dict["token_embedding.weight"].shape[0],
            "transformer_width": state_dict["ln_final.weight"].shape[0],
            "transformer_heads": transformer_width // 64,
            "transformer_layers": len(
                set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
        }
        args1.update(other_args)
        print("Instantiated model with args:", args1)
        return Model(**args1)