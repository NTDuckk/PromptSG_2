import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


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
        # x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class InversionNetwork(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim, affine=False)
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

        self.composed_str = "A photo of a X person"
        self.simplified_str = "A photo of a person"
        
       # register buffers once (empty => chưa khởi tạo)
        self.register_buffer("tokenized_composed", torch.empty(0, dtype=torch.long))
        self.register_buffer("tokenized_simplified", torch.empty(0, dtype=torch.long))
        self.register_buffer("embed_composed", torch.empty(0))
        self.register_buffer("embed_simplified", torch.empty(0))

        self.x_pos = None
        
    def _ensure_tokenization(self):
        if self.tokenized_composed.numel() == 0:
            import model.clip.clip as clip_module

            dev = self.token_embedding.weight.device  # cùng device với embedding

            tokenized_composed = clip_module.tokenize([self.composed_str]).to(dev)
            tokenized_simplified = clip_module.tokenize([self.simplified_str]).to(dev)

            tokenized_x = clip_module.tokenize(["X"]).to(dev)
            x_token_id = tokenized_x[0, 1].item()

            x_pos = (tokenized_composed[0] == x_token_id).nonzero(as_tuple=False)
            if x_pos.numel() == 0:
                raise ValueError("Cannot locate placeholder token in composed prompt")

            # chỉ gán (buffer đã tồn tại từ __init__)
            self.tokenized_composed = tokenized_composed
            self.tokenized_simplified = tokenized_simplified
            self.x_pos = int(x_pos[0].item())

    def _ensure_embeddings(self):
        self._ensure_tokenization()
        if self.embed_composed.numel() == 0:
            with torch.no_grad():
                embed_composed = self.token_embedding(self.tokenized_composed).type(self.dtype)
                embed_simplified = self.token_embedding(self.tokenized_simplified).type(self.dtype)

            self.embed_composed = embed_composed
            self.embed_simplified = embed_simplified


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

class LayerNorm(nn.LayerNorm):
    """LayerNorm that is safe to use with fp16 (casts to fp32 for normalization)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.float())
        return ret.to(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class PostCABlock(nn.Module):
    """ViT-style Encoder block: PreLN -> SelfAttn -> MLP"""
    def __init__(self, d_model=512, nhead=8, mlp_ratio=4.0, drop_path=0.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_drop,
            batch_first=True
        )

        self.ln2 = LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            QuickGELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(proj_drop),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # Self-attn
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path(attn_out)

        # FFN
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class MultimodalInteractionModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int = 2,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        reweight: str = "mul_mean1",
    ):
        super().__init__()
        self.reweight = reweight

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )

        self.q_ln = LayerNorm(embed_dim)
        self.kv_ln = LayerNorm(embed_dim)

        self.post_blocks = nn.ModuleList(
            [
                PostCABlock(
                    d_model=embed_dim,
                    nhead=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, text_feat, patch_tokens, cls_token, return_cls_states=False):
        B, M, D = patch_tokens.shape

        q = self.q_ln(text_feat).unsqueeze(1)   # (B,1,D)
        kv = self.kv_ln(patch_tokens)           # (B,M,D)

        _, attn_w = self.cross_attn(
            query=q, key=kv, value=kv,
            need_weights=True,
            average_attn_weights=False
        )
        attn_map = attn_w.mean(dim=1)  # (B,1,M)

        # reweight patches
        if self.reweight == "mul_mean1":
            scale = (attn_map.squeeze(1) * M).unsqueeze(-1)  # (B,M,1)
            patch_rw = patch_tokens * scale
        elif self.reweight == "mul":
            patch_rw = patch_tokens * attn_map.transpose(1, 2)
        elif self.reweight == "residual":
            scale = (attn_map.squeeze(1) * M).unsqueeze(-1)
            patch_rw = patch_tokens * (1.0 + scale)
        else:
            raise ValueError(f"Unknown reweight mode: {self.reweight}")

        # seq before post blocks
        seq = torch.cat([cls_token, patch_rw], dim=1)  # (B,1+M,D)

        cls_states = [seq[:, 0, :]]  # state0 (before blocks)
        for blk in self.post_blocks:
            seq = blk(seq)
            cls_states.append(seq[:, 0, :])  # state after each block

        if return_cls_states:
            return seq, attn_map, cls_states  # len = 1 + num_blocks
        return seq, attn_map

class PromptSGModel(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super().__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.prompt_mode = cfg.MODEL.PROMPTSG.PROMPT_MODE
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        
        # KHÔNG THAY ĐỔI - Giữ nguyên như CLIP-ReID
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024  # CLIP ResNet50 có projected feature 1024
        
        # Classifiers - GIỮ NGUYÊN
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        # Bottlenecks - GIỮ NGUYÊN
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # Load CLIP model
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        # Encoders
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        # PromptSG modules
        self.prompt_composer = PromptComposer(clip_model, cfg.MODEL.PROMPTSG.PROMPT_MODE)
        self.inversion = InversionNetwork(dim=512)  # Luôn là 512 vì CLIP text encoder output 512

        # Multimodal Interaction Module (MIM)
        self.mim = MultimodalInteractionModule(
            embed_dim=512,  # Luôn là 512 cho CLIP
            num_heads=cfg.MODEL.PROMPTSG.CROSS_ATTN_HEADS,
            num_blocks=cfg.MODEL.PROMPTSG.POST_CA_BLOCKS
        )
        
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder.eval()

        # Cache for simplified prompt
        self._text_feat_cached = None

    def _ensure_text_features(self):
        if self._text_feat_cached is None:
            self.prompt_composer._ensure_embeddings()
            with torch.no_grad():
                prompts = self.prompt_composer.embed_simplified  # (1,L,512)
                tokenized = self.prompt_composer.tokenized_simplified  # (1,L)
                text = self.text_encoder(prompts, tokenized)  # (1,512)
            self._text_feat_cached = text.detach().cpu()


    def forward(self, x = None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None):
        """
        Forward pass of PromptSG model
        """
        # Get text features only
        if get_text:
            if self.prompt_mode == 'simplified':
                self._ensure_text_features()
                text_features = self._text_feat_cached.to(device=x.device).expand(x.shape[0], -1)
            else:
                # For composed prompt, need to generate pseudo token first
                # image_features_last, image_features, image_features_proj = self.image_encoder(x)
                features_intermediate, features_final, features_proj= self.image_encoder(x)
                v = features_proj[:, 0] if self.model_name == 'ViT-B-16' else features_proj[0]
                s_star = self.inversion(v)
                prompts, tokenized = self.prompt_composer(s_star)
                # with torch.no_grad():
                text_features = self.text_encoder(prompts, tokenized)
            return text_features

        # Get image features only
        if get_image:
            # image_features_last, image_features, image_features_proj = self.image_encoder(x)
            features_intermediate, features_final, features_proj= self.image_encoder(x)
            if self.model_name == 'RN50':
                return features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return features_proj[:, 0]

        # Main forward pass for training/inference
        # Get image features from CLIP visual encoder
        
        # image_features_last, image_features, image_features_proj = self.image_encoder(x)
        features_intermediate, features_final, features_proj = self.image_encoder(x)

        # Extract features based on backbone type
        if self.model_name == 'ViT-B-16':
            # ViT-B/16: [CLS] token + patch tokens
            # img_feature_last = image_features_last[:, 0]  # Last layer CLS token (768)
            # img_feature = image_features[:, 0]  # Intermediate CLS token (768)
            # img_feature_proj = image_features_proj[:, 0]  # Projected CLS token (512)
         
            CLS_intermediate = features_intermediate[:, 0]  # Intermediate CLS token (768)
            CLS_final = features_final[:, 0]  # Last layer CLS token (768)
            CLS_proj = features_proj[:, 0]  # Projected CLS token (512)
            
            # Patches for cross-attention (exclude CLS token)
            patches = features_proj[:, 1:]  # (batch, num_patches, 512)
             
            # CLS token for final sequence
            cls_token = features_proj[:, :1]  # (batch, 1, 512)
            
        elif self.model_name == 'RN50':
            # ResNet50: global feature + spatial features
            # Đoạn này cần kiểm tra kỹ output của CLIP ResNet50
            # Theo CLIP-ReID: image_features_proj[0] là global feature (1024)
            #                image_features_proj[1] là spatial features (1024, h, w)
            #                image_features_proj[2] là projected global feature (512)
            #                image_features_proj[3] là projected spatial features (512, h, w)
            # Theo CLIP-ReID: features_intermediate[0] là global feature (1024)
            #                features_final[0] là spatial features (1024, h, w)
            #                features_proj[0] là projected global feature (1024)
            #                features_proj[1] là projected spatial features (512, h, w)
            
            # Global features
            # img_feature_last = F.avg_pool2d(image_features_last, image_features_last.shape[2:]).view(x.shape[0], -1)  # (batch, 2048)
            # img_feature = F.avg_pool2d(image_features, image_features.shape[2:]).view(x.shape[0], -1)  # (batch, 2048)
            # img_feature_proj = image_features_proj[0]  # Global projected feature (1024)
            CLS_intermediate = F.avg_pool2d(features_intermediate, features_intermediate.shape[2:]).view(x.shape[0], -1)  # (batch, 2048)
            CLS_final = F.avg_pool2d(features_final, features_final.shape[2:]).view(x.shape[0], -1)  # (batch, 2048)
            CLS_proj = features_proj[0]  # Global projected feature (1024)
            
            # Với ResNet50, chúng ta cần dùng projected spatial features (512) cho cross-attention
            # image_features_proj[3] có shape (batch, 512, h, w)
            # if len(image_features_proj) > 3:
            #     b, c, h, w = image_features_proj[3].shape  # c = 512
            #     patches = image_features_proj[3].view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, 512)
            # features_proj[1] có shape (batch, 512, h, w)
            if len(features_proj) > 1:
                b, c, h, w = features_proj[1].shape  # c = 512
                patches = features_proj[1].view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, 512)
                
                # Tạo CLS token từ projected global feature (cần project từ 1024 xuống 512)
                if not hasattr(self, 'resnet_projection'):
                    self.resnet_projection = nn.Linear(1024, 512).to(x.device)
                # cls_token = self.resnet_projection(img_feature_proj).unsqueeze(1)  # (batch, 1, 512)
                cls_token = self.resnet_projection(CLS_proj).unsqueeze(1)  # (batch, 1, 512)
            else:
                # Fallback: dùng spatial features và project
                # b, c, h, w = image_features_proj[1].shape  # c = 1024
                # patches = image_features_proj[1].view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, 1024)
                b, c, h, w = features_final.shape  # c = 1024
                patches = features_final.view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, 1024)
                
                # Project patches từ 1024 xuống 512
                if not hasattr(self, 'patch_projection'):
                    self.patch_projection = nn.Linear(1024, 512).to(x.device)
                patches = self.patch_projection(patches)  # (batch, h*w, 512)
                
                # Tạo CLS token
                if not hasattr(self, 'resnet_projection'):
                    self.resnet_projection = nn.Linear(1024, 512).to(x.device)
                # cls_token = self.resnet_projection(img_feature_proj).unsqueeze(1)  # (batch, 1, 512)
                cls_token = self.resnet_projection(CLS_proj).unsqueeze(1)  # (batch, 1, 512)
        
        # Get global visual embedding for inversion network
        # Dùng projected feature cho inversion (luôn là 512)
        if self.model_name == 'ViT-B-16':
            v = CLS_proj  # Đã là 512
        elif self.model_name == 'RN50':
            # Cần project từ 1024 xuống 512
            if not hasattr(self, 'inversion_projection'):
                self.inversion_projection = nn.Linear(1024, 512).to(x.device)
            v = self.inversion_projection(CLS_proj)  # (batch, 512)

        # Generate text features based on prompt mode
        if self.prompt_mode == 'simplified':
            self._ensure_text_features()
            device = x.device if x is not None else next(self.parameters()).device
            text_feat = self._text_feat_cached.to(device).expand(x.shape[0], -1)
        else:
            s_star = self.inversion(v)  # Generate pseudo token (512)
            prompts, tokenized = self.prompt_composer(s_star)
            text_feat = self.text_encoder(prompts, tokenized)  # (batch, 512)

        # ========== Multimodal Interaction Module (MIM) ==========
        # sequence, attn_map = self.mim(text_feat, patches, cls_token)
        sequence, attn_map, cls_states = self.mim(text_feat, patches, cls_token, return_cls_states=True)
        
        # Extract final representation from first token (text-enhanced or CLS)
        # v_final = sequence[:, 0, :]   # (batch, 512)
        triplet_feats = [cls_states[-1], cls_states[-2], cls_states[-3]]
        v_final = cls_states[-1]
        
        # ========== Bottleneck Layers ==========
        # Dùng img_feature (768/2048) cho bottleneck chính
        # Dùng v_final (512) cho bottleneck projection

        # feat = self.bottleneck(img_feature)  # (batch, in_planes)
        feat = self.bottleneck(CLS_final) #CLS_final: CLS x12 - 768

        # Với ResNet50, cần project v_final từ 512 lên 1024 cho bottleneck_proj
        if self.model_name == 'RN50':
            if not hasattr(self, 'final_projection'):
                self.final_projection = nn.Linear(512, 1024).to(x.device)
            feat_proj_input = self.final_projection(v_final)
        else:
            feat_proj_input = v_final  # ViT: giữ nguyên 512
            
        feat_proj = self.bottleneck_proj(feat_proj_input)

        # ========== Output ==========
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            
            # Return format: cls_score, triplet_feats, image_feat, text_feat
            # cls_score: [cls_score, cls_score_proj] 
            # triplet_feats: [img_feature_last, img_feature, v_final]
            # image_feat: v_final (final representation)
            # text_feat: text features from cross-attention

            #processor đang nhận: cls_score, triplet_feats, image_feat, text_feat, target.  
            # return [cls_score, cls_score_proj], [CLS_intermediate, CLS_final, v_final], v_final, text_feat
            return [cls_score_proj], triplet_feats, v, text_feat

        else:
            if self.neck_feat == 'after':
                # Concatenate features after bottleneck
                return torch.cat([feat, feat_proj], dim=1)
            else:
                # Concatenate original image feature với v_final
                # Với ResNet50, cần project v_final từ 512 lên 1024
                if self.model_name == 'RN50':
                    if not hasattr(self, 'concat_projection'):
                        self.concat_projection = nn.Linear(512, 1024).to(x.device)
                    v_final_concat = self.concat_projection(v_final)
                else:
                    v_final_concat = v_final
                return torch.cat([CLS_final, v_final_concat], dim=1)

    def load_param(self, trained_path):
        """Load pretrained parameters"""
        param_dict = torch.load(trained_path, map_location='cpu')
        for key in param_dict:
            new_key = key.replace('module.', '')
            if new_key in self.state_dict():
                self.state_dict()[new_key].copy_(param_dict[key])


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    for key in state_dict.keys():
        print(key)

    # Hoặc tìm keys chứa "text_projection"
    text_proj_keys = [k for k in state_dict.keys() if "text_projection" in k]
    print("Text projection keys test:", text_proj_keys)

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def make_model(cfg, num_class, camera_num, view_num):
    model = PromptSGModel(num_class, camera_num, view_num, cfg)
    return model
