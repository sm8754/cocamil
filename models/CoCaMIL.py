import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class Loc_embedding(nn.Module):
    def __init__(self, dim=512):
        super(Loc_embedding, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TextEncoder(nn.Module):
    def __init__(self, context_length,embed_dim,
                 vocab_size,transformer_width,transformer_heads,transformer_layers, dtype=torch.float32):
        super(TextEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.dtype = dtype
        self.context_length = context_length

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class COCAMIL(nn.Module):
    def __init__(self, n_classes, l_margin, u_margin, dims, easy_margin,context_length,embed_dim,
                 vocab_size,transformer_width,transformer_heads,transformer_layers):
        super(COCAMIL, self).__init__()
        self.dims = dims
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.easy_margin = easy_margin
        self.pos_layer = Loc_embedding(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.dims)
        self.weight = nn.Parameter(torch.Tensor(n_classes, self.dims))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.encode_text = TextEncoder(context_length,embed_dim,
                 vocab_size,transformer_width,transformer_heads,transformer_layers)
        self._fc3 = nn.Linear(512, 512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(m.log(), p, reduction='none').sum(dim=1)
        kl_qm = F.kl_div(m.log(), q, reduction='none').sum(dim=1)
        js_div = 0.5 * (kl_pm + kl_qm)
        return js_div

    def comput_score(self,text_features,image_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        similarity = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity[0].topk(1)

        image_features = image_features.float()
        target_text_features = text_features[indices].float()

        image_features2 = F.softmax(image_features, dim=1)
        text_features2 = F.softmax(target_text_features, dim=1)
        js_div = self.js_divergence(text_features2, image_features2)

        return js_div,similarity[0]

    def _margin(self, alpha):
        margin = (self.u_margin - self.l_margin) * (1-alpha) + self.l_margin
        return margin

    def adjust_ratio(self,text_features,image_features,class_features):
        weight_norm = F.normalize(self.weight, dim=1)  # [n_classes, 512]
        cos_theta = torch.mm(F.normalize(class_features), weight_norm.T)  # [B, n_classes]
        cos_theta = cos_theta.clamp(-1, 1)

        score,similarity= self.comput_score(text_features,image_features)
        # Compute adaptive margin
        ada_margin = self._margin(score)  # [B, 1]
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        # Apply margin
        if self.easy_margin:
            cos_theta_m = torch.where((cos_theta > 0).to(torch.bool), cos_theta_m.to(torch.float32),
                                      cos_theta.to(torch.float32))
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where((cos_theta > threshold).to(torch.bool), cos_theta_m.to(torch.float32),
                                      (cos_theta - mm).to(torch.float32))
        return cos_theta_m, cos_theta,score,similarity

    def forward(self, **kwargs):

        image = kwargs['data'].float()
        
        h = self._fc1(image)

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1)

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)

        h = self.norm(h)[:,0]
        h2 = h.view(B, -1)

        class_features = self._fc2(h2)

        if clip:
            image_features = self._fc3(h2)
            text = kwargs['text']
            text_features = self.clip_model.encode_text(tokenize(text, context_length=77).to("cuda"))

            cos_theta_m, cos_theta, score,similarity = self.adjust_ratio(text_features,image_features,class_features)

            results_dict = {
                'cos_theta': cos_theta,
                'cos_theta_m': cos_theta_m,
                'complexity_score': score,
                'text_similarity': similarity
                }
        else:
            weight_norm = F.normalize(self.weight, dim=1)  # [n_classes, 512]
            cos_theta = torch.mm(F.normalize(class_features), weight_norm.T)  # [B, n_classes]
            cos_theta = cos_theta.clamp(-1, 1)

            results_dict = {
                'cos_theta': cos_theta
            }


        return results_dict



