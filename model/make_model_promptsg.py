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
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        text_feature = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature


class IM2TEXT(nn.Module):
    """
    Reuse the inversion-network style MLP that already exists in your codebase.
    Input: global projected image embedding (512)
    Output: one pseudo token s* in the CLIP token space (512)
    """

    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
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
        return self.fc_out(x)


class PromptComposer(nn.Module):
    """
    Compose PromptSG prompt:
      composed   : "A photo of a X person." / vehicle
      simplified : "A photo of a person." / vehicle
    Here X is replaced by the pseudo-token s* generated from the image.
    """

    def __init__(self, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name in ['VehicleID', 'veri']:
            composed_template = 'A photo of a X vehicle.'
            simplified_template = 'A photo of a vehicle.'
        else:
            composed_template = 'A photo of a X person.'
            simplified_template = 'A photo of a person.'

        self.dtype = dtype

        tokenized_composed = clip.tokenize(composed_template)
        tokenized_simplified = clip.tokenize(simplified_template)

        with torch.no_grad():
            composed_embedding = token_embedding(tokenized_composed).type(dtype)
            simplified_embedding = token_embedding(tokenized_simplified).type(dtype)

        # For "A photo of a X person." the placeholder token is after the first 5 embeddings.
        self.register_buffer('tokenized_composed', tokenized_composed)
        self.register_buffer('tokenized_simplified', tokenized_simplified)
        self.register_buffer('token_prefix', composed_embedding[:, :5, :])
        self.register_buffer('token_suffix', composed_embedding[:, 6:, :])
        self.register_buffer('simplified_prompts', simplified_embedding)

    def compose(self, pseudo_token: torch.Tensor):
        b = pseudo_token.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        prompts = torch.cat([prefix, pseudo_token.unsqueeze(1), suffix], dim=1)
        return prompts

    def get_composed_tokens(self, batch_size, device):
        return self.tokenized_composed.to(device).expand(batch_size, -1)

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

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to('cuda')

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
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

        # Reuse the user's requested inversion-network block.
        self.img2text = IM2TEXT(
            embed_dim=self.in_planes_proj,
            middle_dim=self.in_planes_proj,
            output_dim=self.in_planes_proj,
            n_layer=2,
            dropout=0.1,
        )

        # Reuse the user's requested cross-attention stack.
        self.cross_attn = nn.MultiheadAttention(
            self.in_planes_proj,
            self.in_planes_proj // 64,
            batch_first=True,
        )
        self.cross_modal_transformer = Transformer(
            width=self.in_planes_proj,
            layers=2,
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

        self.prompt_composer = PromptComposer(cfg.DATASETS.NAMES, clip_model.dtype, clip_model.token_embedding)

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

    def _get_cv_embed(self, cam_label=None, view_label=None):
        if self.cv_embed is None:
            return None
        if cam_label is not None and view_label is not None and self.camera_num > 0 and self.view_num > 0:
            return self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        if cam_label is not None and self.camera_num > 0:
            return self.sie_coe * self.cv_embed[cam_label]
        if view_label is not None and self.view_num > 0:
            return self.sie_coe * self.cv_embed[view_label]
        return None

    def encode_visual_tokens(self, x, cam_label=None, view_label=None):
        cv_embed = self._get_cv_embed(cam_label=cam_label, view_label=view_label)
        x11, x12, xproj = self.image_encoder(x, cv_embed)

        if self.model_name == 'RN50':
            raise NotImplementedError('RN50 PromptSG path is not implemented in this patch.')

        return {
            'visual_tokens': xproj.float(),
            'cls_x11': x11[:, 0, :].float(),
            'cls_x12': x12[:, 0, :].float(),
            'global_proj': xproj[:, 0, :].float(),
        }

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False,
        )[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x[0].unsqueeze(0)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x

    def encode_text_from_image(self, global_proj):
        pseudo_token = self.img2text(global_proj)
        prompts = self.prompt_composer.compose(pseudo_token.to(self.prompt_composer.dtype))
        tokenized = self.prompt_composer.get_composed_tokens(prompts.shape[0], prompts.device)
        text_feature = self.text_encoder(prompts, tokenized)
        return text_feature.float(), pseudo_token.float()

    def encode_text_simplified(self, batch_size, device):
        prompts = self.prompt_composer.get_simplified_prompts(batch_size, device).type(self.prompt_composer.dtype)
        tokenized = self.prompt_composer.get_simplified_tokens(batch_size, device)
        text_feature = self.text_encoder(prompts, tokenized)
        return text_feature.float()

    def forward_train(self, x, pids=None, cam_label=None, view_label=None):
        visual = self.encode_visual_tokens(x, cam_label=cam_label, view_label=view_label)
        text_feature, pseudo_token = self.encode_text_from_image(visual['global_proj'])
        cross_x = self.cross_former(text_feature.unsqueeze(1), visual['visual_tokens'], visual['visual_tokens'])
        fused = cross_x.squeeze(1)
        fused_bn = self.bottleneck(fused)
        cls_score = self.classifier(fused_bn)

        return {
            'cls_score': cls_score,
            'triplet_feats': [visual['cls_x11'], visual['cls_x12'], fused],
            'global_image': visual['global_proj'],
            'text_feat': text_feature,
            'fused_feat': fused,
            'fused_bn': fused_bn,
            'pseudo_token': pseudo_token,
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
        frozen = set(id(p) for p in self.text_encoder.parameters())

        for p in self.image_encoder.parameters():
            if p.requires_grad and id(p) not in frozen:
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

        return [
            {'params': visual_params, 'lr': visual_lr, 'weight_decay': weight_decay},
            {'params': new_params, 'lr': new_lr, 'weight_decay': weight_decay},
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


from .clip import clip as _clip_module


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = _clip_module._MODELS[backbone_name]
    model_path = _clip_module._download(url)

    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = _clip_module.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def make_model(cfg, num_class, camera_num, view_num):
    return PromptSGReID(num_class, camera_num, view_num, cfg)
