import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip.model import Transformer, LayerNorm
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)

        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = [
                nn.Linear(dim, middle_dim),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
            ]
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        if tokenized_prompts.shape[0] == 1 and prompts.shape[0] > 1:
            tokenized_prompts = tokenized_prompts.expand(prompts.shape[0], -1)

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1).to(x.device)] @ self.text_projection 
        return x


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()

        ctx_init = "A photo of a X person"
        ctx_dim = 512
        n_ctx = 4  # "A photo of a"

        device = token_embedding.weight.device
        tokenized_prompts = clip.tokenize(ctx_init).to(device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + 1:, :])

        self.num_class = num_class
        self.token_ = token_embedding.to(device)
        self.dtype = dtype
        self.ctx_dim = ctx_dim

    def forward(self, bias):
        # bias: [B, 512]
        b = bias.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        bias = bias.unsqueeze(1)  # [B, 1, 512]

        prompts = torch.cat(
            [
                prefix,
                bias,
                suffix,
            ],
            dim=1,
        )
        return prompts


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.embed_dim = self.in_planes_proj
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        # keep old ones in case you still want auxiliary triplet from visual states
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # classifier / bn used on cross-modal output
        self.classifier_proj = nn.Linear(self.embed_dim, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        
        self.classifier_proj_cross = nn.Linear(self.embed_dim, self.num_classes, bias=False)
        self.classifier_proj_cross.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(self.embed_dim)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.bottleneck_proj_cross = nn.BatchNorm1d(self.embed_dim)
        self.bottleneck_proj_cross.bias.requires_grad_(False)
        self.bottleneck_proj_cross.apply(weights_init_kaiming)

        self.classifier_id_bge = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.normal_(self.classifier_id_bge.weight.data, std=0.001)
        nn.init.constant_(self.classifier_id_bge.bias.data, val=0.0)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(
            self.model_name,
            self.h_resolution,
            self.w_resolution,
            self.vision_stride_size
        )
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        # PromptSG freezes text encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('view number is : {}'.format(view_num))
        else:
            self.cv_embed = None

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(
            num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding
        )

        # inversion network
        self.img2text = IM2TEXT(
            embed_dim=self.embed_dim,
            middle_dim=512,
            output_dim=512 if self.embed_dim == 512 else self.embed_dim,
            n_layer=2
        )

        # cross-attn + transformer (PromptSG MIM)
        cmt_depth = getattr(cfg.MODEL, "CMT_DEPTH", 2)

        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            self.embed_dim // 64,
            batch_first=True
        )
        self.cross_modal_transformer = Transformer(
            width=self.embed_dim,
            layers=cmt_depth,
            heads=self.embed_dim // 64
        )

        scale = self.cross_modal_transformer.width ** -0.5
        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5

        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
        if self.cross_attn.in_proj_bias is not None:
            nn.init.constant_(self.cross_attn.in_proj_bias, 0.0)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.constant_(self.cross_attn.out_proj.bias, 0.0)

    def cross_former(self, text_tokens, image_tokens_k, image_tokens_v):
        # text_tokens: [B, 1, C]
        # image_tokens_k/v: [B, L, C]
        q = self.ln_pre_t(text_tokens)
        k = self.ln_pre_i(image_tokens_k)
        v = self.ln_pre_i(image_tokens_v)

        x = q + self.cross_attn(q, k, v, need_weights=False)[0]  # [B,1,C]
        x = x.permute(1, 0, 2)  # [1,B,C]
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # [B,1,C]
        x = self.ln_post(x)
        return x

    def forward(self, x=None, label=None, cam_label=None, view_label=None,
                eval_mode=None, dataset_flag=None):
        if x is None:
            raise ValueError("Input image x must not be None.")

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            image_feats_proj = image_features_proj.permute(1, 0, 2)  # [B,L,C]
            img_feature_last = F.adaptive_avg_pool2d(
                image_features_last, image_features_last.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature = F.adaptive_avg_pool2d(
                image_features, image_features.shape[2:4]
            ).view(x.shape[0], -1)
            img_feature_proj = image_feats_proj[:, 0]

        elif self.model_name == 'ViT-B-16':
            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None

            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            # image_feats_proj = image_features_proj  # [B,L,C]
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        # PromptSG: invert global visual embedding -> pseudo token
        token_features = self.img2text(img_feature_proj)

        # compose prompt and encode text
        prompts = self.prompt_learner(token_features)
        text_feature = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

        # normalized pair for SupCon / alignment
        text_feature_norm = F.normalize(text_feature, dim=-1)
        img_feature_proj_norm = F.normalize(img_feature_proj, dim=-1)

        # Multimodal Interaction Module
        cross_x = self.cross_former(
            text_feature.unsqueeze(1),   # [B,1,C]
            image_features_proj,            # [B,L,C]
            image_features_proj             # [B,L,C]
        )
        cross_feat = cross_x.squeeze(1)  # [B,C]
        cross_x_bn = self.bottleneck_proj_cross(cross_feat)   
        cls_score = self.classifier_proj_cross(cross_x_bn).float()

        feat = self.bottleneck(img_feature)
        feat_cls_score = self.classifier(feat).float()

        img_feat = self.bottleneck_proj(img_feature_proj)
        img_score = self.classifier_proj(img_feat).float()

        # image_logits = self.classifier_id_bge(img_feature_proj).float()
        # text_logits = self.classifier_id_bge(text_feature).float()
        if self.training:
            return {
                # "cls_score": [cls_score, feat_cls_score, image_logits, text_logits], # cho ID loss
                "cls_score": [cls_score, feat_cls_score, img_score], # cho ID loss
                "global_feat": [cross_feat, img_feature_proj, img_feature, img_feature_last, text_feature],  # cho Triplet loss
                "text_feat": text_feature_norm,         # cho SupCon
                "img_feat_proj": img_feature_proj_norm  # cho SupCon
            }
        else:
            if eval_mode == 'clipreid':
                if self.neck_feat == 'after':
                    return torch.cat([cross_x_bn, feat], dim=1)
                    # return torch.cat([cross_x_bn, feat, F.normalize(img_feature_proj, dim=1)], dim=1)
                else:
                    return cross_feat
            elif eval_mode == 'cross_cls':
                if dataset_flag == 'query':
                    # return cross_x_bn
                    return torch.cat([cross_x_bn], dim=1)
                else:
                    return torch.cat([img_feat], dim=1)
                    # return torch.cat([cross_x_bn], dim=1)
                    

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(
        state_dict or model.state_dict(),
        h_resolution,
        w_resolution,
        vision_stride_size
    )
    return model