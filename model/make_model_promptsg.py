import math
import torch
import torch.nn as nn

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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class InversionNetwork(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, v):
        x = self.act(self.fc1(v))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        return x


class PromptComposer(nn.Module):
    def __init__(self, clip_model, prompt_mode: str):
        super().__init__()
        self.prompt_mode = prompt_mode
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype

        composed_str = "A photo of a X person."
        simplified_str = "A photo of a person."

        tokenized_composed = clip.tokenize(composed_str)
        tokenized_simplified = clip.tokenize(simplified_str)
        tokenized_x = clip.tokenize("X")
        x_token_id = tokenized_x[0, 1].item()
        x_pos = (tokenized_composed[0] == x_token_id).nonzero(as_tuple=False)
        if x_pos.numel() == 0:
            raise ValueError("Cannot locate placeholder token in composed prompt")
        self.x_pos = int(x_pos[0].item())

        self.register_buffer("tokenized_composed", tokenized_composed)
        self.register_buffer("tokenized_simplified", tokenized_simplified)
        self.embed_composed = None
        self.embed_simplified = None

    def _ensure_embeddings(self):
        if self.embed_composed is None:
            with torch.no_grad():
                self.embed_composed = self.token_embedding(self.tokenized_composed).type(self.dtype)
                self.embed_simplified = self.token_embedding(self.tokenized_simplified).type(self.dtype)
            self.register_buffer("embed_composed", self.embed_composed)
            self.register_buffer("embed_simplified", self.embed_simplified)

    def forward(self, s_star: torch.Tensor):
        self._ensure_embeddings()
        b = s_star.shape[0]
        if self.prompt_mode == 'simplified':
            tokenized = self.tokenized_simplified.expand(b, -1)
            prompts = self.embed_simplified.expand(b, -1, -1)
            return prompts, tokenized

        s_star = s_star.to(dtype=self.embed_composed.dtype)

        tokenized = self.tokenized_composed.expand(b, -1)
        prefix = self.embed_composed[:, :self.x_pos, :].expand(b, -1, -1)
        suffix = self.embed_composed[:, self.x_pos + 1 :, :].expand(b, -1, -1)
        prompts = torch.cat([prefix, s_star.unsqueeze(1), suffix], dim=1)
        return prompts, tokenized


class CrossAttentionGuidance(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, text_feat: torch.Tensor, patch_tokens: torch.Tensor):
        q = text_feat.unsqueeze(1)
        out, weights = self.attn(q, patch_tokens, patch_tokens, need_weights=True, average_attn_weights=False)
        w = weights.mean(dim=1).squeeze(1).squeeze(1)
        return out.squeeze(1), w


class PromptSGModel(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.prompt_mode = cfg.MODEL.PROMPTSG.PROMPT_MODE
        if self.model_name != 'ViT-B-16':
            raise NotImplementedError('Only ViT-B-16 is supported in PromptSGModel in this repo')

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to(cfg.MODEL.DEVICE)

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.prompt_composer = PromptComposer(clip_model, cfg.MODEL.PROMPTSG.PROMPT_MODE)
        self.inversion = InversionNetwork(dim=512)

        self.cross_guidance = CrossAttentionGuidance(embed_dim=512, num_heads=cfg.MODEL.PROMPTSG.CROSS_ATTN_HEADS)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=cfg.MODEL.PROMPTSG.CROSS_ATTN_HEADS,
            dim_feedforward=2048,
            activation='gelu',
            batch_first=True,
        )
        self.post_blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.MODEL.PROMPTSG.POST_CA_BLOCKS)

        self.bottleneck = nn.BatchNorm1d(512)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(512, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        with torch.no_grad():
            dummy = torch.zeros(1, 512)
            prompts, tokenized = self.prompt_composer(dummy)
            prompts = prompts.to(device=cfg.MODEL.DEVICE)
            tokenized = tokenized.to(device=cfg.MODEL.DEVICE)
            self.register_buffer("_text_feat_cached", self.text_encoder(prompts, tokenized).detach().cpu())

    def forward(self, x, label=None):
        x10, x11, x12, xproj = self.image_encoder(x, return_intermediate=True)

        v = xproj[:, 0]
        patches = xproj[:, 1:]

        if self.prompt_mode == 'simplified':
            text_feat = self._text_feat_cached.to(device=x.device).expand(x.shape[0], -1)
        else:
            s_star = self.inversion(v)
            prompts, tokenized = self.prompt_composer(s_star)
            prompts = prompts.to(device=x.device)
            tokenized = tokenized.to(device=x.device)
            with torch.no_grad():
                text_feat = self.text_encoder(prompts, tokenized)

        _, patch_weights = self.cross_guidance(text_feat, patches)
        patches = patches * patch_weights.unsqueeze(-1)

        seq = torch.cat([xproj[:, :1], patches], dim=1)
        seq = self.post_blocks(seq)
        v_final = seq[:, 0]

        x10p = (self.image_encoder.ln_post(x10) @ self.image_encoder.proj)[:, 0]
        x11p = (self.image_encoder.ln_post(x11) @ self.image_encoder.proj)[:, 0]

        feat_bn = self.bottleneck(v_final)

        if self.training:
            cls_score = self.classifier(feat_bn)
            return cls_score, [x10p, x11p, v_final], v_final, text_feat

        return v_final

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def make_model(cfg, num_class, camera_num=None, view_num=None):
    return PromptSGModel(num_classes=num_class, cfg=cfg)
