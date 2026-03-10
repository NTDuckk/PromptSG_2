
import os
import torch
import torch.nn as nn
from typing import Optional
from timm.models.layers import trunc_normal_

from .clip.model import LayerNorm, Transformer
from .clip import clip


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


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # ipdb.set_trace()
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = self.transformer([x])
        x = outputs[0]
        att = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD   # x,att
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature


class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.bn(self.fc_out(x))


class PromptComposer(nn.Module):
    def __init__(self, composed_template, simplified_template, dtype, token_embedding):
        super().__init__()
        self.dtype = dtype
        # Calculate n_ctx from the position of placeholder 'X' in the template
        words = composed_template.split()
        n_ctx = words.index('X')

        tokenized_prompts = clip.tokenize(composed_template)
        tokenized_simplified = clip.tokenize(simplified_template)
        device = token_embedding.weight.device
        tokenized_prompts = tokenized_prompts.to(device)
        tokenized_simplified = tokenized_simplified.to(device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
            simplified_embedding = token_embedding(tokenized_simplified).type(dtype)

        self.register_buffer('tokenized_prompts', tokenized_prompts)
        self.register_buffer('tokenized_simplified', tokenized_simplified)
        self.register_buffer('token_prefix', embedding[:, :n_ctx + 1, :])
        self.register_buffer('token_suffix', embedding[:, n_ctx + 2:, :])
        self.register_buffer('simplified_prompts', simplified_embedding)

    def forward(self, bias: torch.Tensor):
        b = bias.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        bias = bias.unsqueeze(1)
        prompts = torch.cat([prefix, bias, suffix], dim=1)
        return prompts

    def get_composed_tokens(self, batch_size, device):
        return self.tokenized_prompts.to(device).expand(batch_size, -1)

    def get_simplified_prompts(self, batch_size, device):
        return self.simplified_prompts.to(device).expand(batch_size, -1, -1)

    def get_simplified_tokens(self, batch_size, device):
        return self.tokenized_simplified.to(device).expand(batch_size, -1)


class PromptSGReID(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        self.inference_prompt_mode = 'simplified'

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        else:
            raise NotImplementedError('PromptSG patch currently supports ViT-B-16 and RN50 shapes only.')

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        self.base_model = load_clip_to_cpu(cfg, self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(device)

        self.text_encoder = TextEncoder(self.base_model)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        else:
            self.cv_embed = None

        self.img2text = IM2TEXT(
            embed_dim=self.in_planes_proj,
            middle_dim=self.in_planes_proj,
            output_dim=self.in_planes_proj,
            n_layer=cfg.MODEL.PROMPTSG.INVERSION_LAYERS,
            dropout=cfg.MODEL.PROMPTSG.INVERSION_DROPOUT,
        )

        self.cross_attn = nn.MultiheadAttention(
            self.in_planes_proj,
            self.in_planes_proj // 64,
            batch_first=True,
        )
        self.cross_modal_transformer = Transformer(
            width=self.in_planes_proj,
            layers=cfg.MODEL.PROMPTSG.CMT_DEPTH,
            heads=self.in_planes_proj // 64,
        )
        self.ln_pre_t = LayerNorm(self.in_planes_proj)
        self.ln_pre_i = LayerNorm(self.in_planes_proj)
        self.ln_post = LayerNorm(self.in_planes_proj)

        scale = self.cross_modal_transformer.width ** -0.5
        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.prompt_composer = PromptComposer(
            cfg.MODEL.PROMPTSG.COMPOSED_TEMPLATE,
            cfg.MODEL.PROMPTSG.SIMPLE_TEMPLATE,
            self.base_model.dtype,
            self.base_model.token_embedding,
        )

        self.classifier = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def set_inference_prompt_mode(self, mode: str):
        mode = mode.lower().strip()
        if mode not in {'simplified', 'composed'}:
            raise ValueError(f'Unsupported inference prompt mode: {mode}')
        self.inference_prompt_mode = mode

    def encode_visual_tokens(self, x, cam_label=None, view_label=None):
        visual_out = self.base_model(x)
        if isinstance(visual_out, (tuple, list)):
            visual_tokens = visual_out[0]
            intermediate_hidden = visual_out[2] if len(visual_out) > 2 else []
        else:
            visual_tokens = visual_out
            intermediate_hidden = []

        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(1)

        return {
            'visual_tokens': visual_tokens.float(),
            'global_proj': visual_tokens[:, 0, :].float(),
            'intermediate_hidden': intermediate_hidden,
        }

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False,
        )[0]
        x = x.permute(1, 0, 2)
        x = self.cross_modal_transformer(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x[0].unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        return x

    def encode_text_from_image(self, global_proj):
        pseudo_prompt = self.img2text(global_proj)
        prompts = self.prompt_composer(pseudo_prompt.to(self.prompt_composer.dtype))
        tokenized = self.prompt_composer.get_composed_tokens(prompts.shape[0], prompts.device)
        text_feature = self.text_encoder(prompts, tokenized)
        return text_feature.float(), pseudo_prompt.float()

    def encode_text_simplified(self, batch_size, device):
        prompts = self.prompt_composer.get_simplified_prompts(batch_size, device).type(self.prompt_composer.dtype)
        tokenized = self.prompt_composer.get_simplified_tokens(batch_size, device)
        text_feature = self.text_encoder(prompts, tokenized)
        return text_feature.float()

    def forward_train(self, x, pids=None, cam_label=None, view_label=None):
        visual = self.encode_visual_tokens(x, cam_label=cam_label, view_label=view_label)
        text_feature, pseudo_prompt = self.encode_text_from_image(visual['global_proj'])
        cross_x = self.cross_former(text_feature.unsqueeze(1), visual['visual_tokens'], visual['visual_tokens'])
        fused = cross_x.squeeze(1)
        fused_bn = self.bottleneck(fused)
        cls_score = self.classifier(fused_bn)

        # Multi-layer triplet: fused feature + CLS tokens from preceding 2 ViT layers
        triplet_feats = [fused]
        intermediate = visual.get('intermediate_hidden', [])
        # intermediate contains last 3 ViT layer outputs (LND format)
        # Use the first 2 (preceding layers), skip the last (already used via cross-modal)
        for h in intermediate[:-1]:
            h_nld = h.permute(1, 0, 2).float()  # LND -> NLD
            cls_token = h_nld[:, 0, :]           # CLS token from this layer
            triplet_feats.append(cls_token)

        return {
            'cls_score': cls_score,
            'triplet_feat': fused,
            'triplet_feats': triplet_feats,
            'global_image': visual['global_proj'],
            'text_feat': text_feature,
            'fused_feat': fused,
            'fused_bn': fused_bn,
            'pseudo_prompt': pseudo_prompt,
        }

    def forward_infer(self, x, cam_label=None, view_label=None, prompt_mode: Optional[str] = None):
        mode = (prompt_mode or self.inference_prompt_mode).lower().strip()
        visual = self.encode_visual_tokens(x, cam_label=cam_label, view_label=view_label)

        if mode == 'composed':
            text_feature, _ = self.encode_text_from_image(visual['global_proj'])
        elif mode == 'simplified':
            text_feature = self.encode_text_simplified(x.shape[0], x.device)
        else:
            raise ValueError(f'Unsupported prompt mode: {mode}')

        cross_x = self.cross_former(text_feature.unsqueeze(1), visual['visual_tokens'], visual['visual_tokens'])
        fused = cross_x.squeeze(1)
        fused_bn = self.bottleneck(fused)

        if self.neck_feat == 'after':
            return fused_bn
        return fused

    def forward(self, x=None, label=None, cam_label=None, view_label=None, prompt_mode=None, get_image=False, get_text=False):
        if self.training:
            return self.forward_train(x, label, cam_label=cam_label, view_label=view_label)
        return self.forward_infer(x, cam_label=cam_label, view_label=view_label, prompt_mode=prompt_mode)

    def get_param_groups(self, visual_lr, new_lr, weight_decay):
        visual_params = []
        new_params = []

        for p in self.base_model.parameters():
            if p.requires_grad:
                visual_params.append(p)

        for module in [self.img2text, self.cross_attn, self.cross_modal_transformer, self.classifier, self.bottleneck]:
            for p in module.parameters():
                if p.requires_grad:
                    new_params.append(p)

        for module in [self.ln_pre_t, self.ln_pre_i, self.ln_post]:
            for p in module.parameters():
                if p.requires_grad:
                    new_params.append(p)

        if isinstance(self.cv_embed, nn.Parameter) and self.cv_embed.requires_grad:
            new_params.append(self.cv_embed)

        visual_param_ids = {id(p) for p in visual_params}
        dedup_new = [p for p in new_params if id(p) not in visual_param_ids]

        return [
            {'params': visual_params, 'lr': visual_lr, 'weight_decay': weight_decay},
            {'params': dedup_new, 'lr': new_lr, 'weight_decay': weight_decay},
        ]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        cleaned = {}
        for k, v in param_dict.items():
            cleaned[k.replace('module.', '')] = v
        self.load_state_dict(cleaned, strict=False)
        print('Loading pretrained model from {}'.format(trained_path))


def load_clip_to_cpu(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size):
    model_name_or_path = getattr(cfg.MODEL, 'PRETRAIN_PATH', '') or backbone_name
    image_size = tuple(cfg.INPUT.SIZE_TRAIN)
    model, model_cfg = clip.build_CLIP_from_openai_pretrained(
        model_name_or_path,
        image_size=image_size,
        stride_size=vision_stride_size,
    )
    print('Loading CLIP backbone from {}'.format(model_cfg['model_path']))
    return model


def make_model(cfg, num_class, camera_num, view_num):
    return PromptSGReID(num_class, camera_num, view_num, cfg)
