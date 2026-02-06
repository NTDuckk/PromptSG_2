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

    def forward(self, prompts, tokenized_prompts, return_tokens: bool = False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        tokens_proj = x @ self.text_projection              # (B, L, D)
        eot_idx = tokenized_prompts.argmax(dim=-1)          # (B,)
        pooled = tokens_proj[torch.arange(tokens_proj.size(0), device=tokens_proj.device), eot_idx]  # (B, D)

        if not return_tokens:
            return pooled
        return pooled, tokens_proj, eot_idx



class InversionNetwork(nn.Module):
    """
    f_theta: v (CLIP joint embedding dim) -> s* (token embedding width)
    Paper: 3-layer MLP, hidden=512, BN after last state.
    """
    def __init__(self, v_dim: int, token_dim: int = 512, hidden: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(v_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, token_dim)
        self.bn = nn.BatchNorm1d(token_dim, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, v):
        x = self.act(self.fc1(v))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        return x



class FixedPromptComposer(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype
        
        # Tokenize template parts
        self.template = "A photo of a {} person"
        self.prefix = "A photo of a"
        self.suffix = "person"
        
        # Tokenize từng phần
        import model.clip.clip as clip_module
        prefix_tokens = clip_module.tokenize([self.prefix])[0, 1:-1].tolist()  # exclude SOT and EOT
        suffix_tokens = clip_module.tokenize([self.suffix])[0, 1:-1].tolist()
        
        # CLIP special tokens
        self.sot_token = clip_module.tokenize([""])[0, 0].item()
        self.eot_token = clip_module.tokenize([""])[0, -1].item()
        
        # Tạo token sequence: [SOT] + prefix + [X] + suffix + [EOT]
        self.composed_str = self.template.format("X")
        self.token_ids = clip_module.tokenize([self.composed_str])
        
        # Find position of X in tokenized sequence
        prefix_str = self.composed_str[:self.composed_str.find("X")]
        prefix_ids = clip_module.tokenize([prefix_str])
        self.x_pos = prefix_ids.shape[1] - 1  # Token position of X
        
        fixed_ids = self.token_ids.clone()
        fixed_ids[0, self.x_pos] = clip_module.tokenize(["person"])[0, 1].item()  # Replace X with person token
        fixed_ids = fixed_ids.to(self.token_embedding.weight.device)  # Move to same device as embedding
        with torch.no_grad():
            fixed_emb = self.token_embedding(fixed_ids).type(self.dtype)

        # QUAN TRỌNG: cắt graph để không bị backward qua graph cũ ở iter sau
        fixed_emb = fixed_emb.detach()

        # Nên lưu dạng buffer để:
        # (1) model.to(device) sẽ tự move nó
        # (2) không có grad
        self.register_buffer("fixed_embeddings", fixed_emb)
        self.register_buffer("token_ids", self.token_ids)

    
    def forward(self, s_star):
        """
        s_star: (B, D) pseudo token
        """
        B = s_star.size(0)
        L = self.fixed_embeddings.size(1)
        
        # Tạo prompts bằng cách thay thế embedding tại vị trí X
        prompts = self.fixed_embeddings.expand(B, L, -1).clone()
        prompts[:, self.x_pos, :] = s_star
        
        tokenized = self.token_ids.expand(B, -1)
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

class PromptSGInteraction(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, 
                 drop_path=0.0, reweight="mul_mean1", eps=1e-6):
        super().__init__()
        self.reweight = reweight
        self.eps = eps
        
        # Cross-attention: text query -> visual patches (Equation 7)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True)
        self.cross_norm = LayerNorm(embed_dim)
        
        # 2 transformer blocks (self-attention on visual tokens)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True),
                'norm2': LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    QuickGELU(),
                    nn.Dropout(attn_drop),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    nn.Dropout(attn_drop),
                ),
                'drop_path': DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            })
            for _ in range(2)
        ])

    def _reweight_patches(self, patches, attn_map):
        """Apply attention map to reweight patches (Equation 7 implementation)"""
        B, M, D = patches.shape
        
        if self.reweight == "mul_mean1":
            scale = attn_map * M  # Scale to maintain magnitude
            patches_reweighted = patches * scale.transpose(1, 2)
        elif self.reweight == "mul":
            patches_reweighted = patches * attn_map.transpose(1, 2)
        elif self.reweight == "residual":
            patches_reweighted = patches * (1.0 + attn_map.transpose(1, 2) * M)
        else:
            raise ValueError(f"Unknown reweight mode: {self.reweight}")
            
        return patches_reweighted

    def forward(self, visual_tokens, text_tokens, return_cls_states=False):
        """
        visual_tokens: [CLS] + patches (B, 1+M, D)
        text_tokens: text embedding (could be EOT or full sequence)
        """
        B, L, D = visual_tokens.shape
        
        # Separate CLS token and patches
        cls_token = visual_tokens[:, :1, :]  # (B, 1, D)
        patches = visual_tokens[:, 1:, :]    # (B, M, D)
        
        # Cross-attention: text attends to patches
        # text_tokens as query, patches as key/value
        attn_out, attn_weights = self.cross_attn(
            text_tokens, patches, patches,
            need_weights=True, average_attn_weights=False
        )
        
        # Compute attention map (average over heads and text tokens)
        # attn_weights: (B, num_heads, L_text, M_patches)
        attn_map = attn_weights.mean(dim=1)  # (B, L_text, M)
        if attn_map.size(1) > 1:  # If multiple text tokens, average them
            attn_map = attn_map.mean(dim=1, keepdim=True)  # (B, 1, M)
        
        # Normalize attention map
        attn_map = attn_map / (attn_map.sum(dim=-1, keepdim=True) + self.eps)
        
        # Reweight patches using attention map
        patches_reweighted = self._reweight_patches(patches, attn_map)
        
        # Recombine with CLS token
        visual = torch.cat([cls_token, patches_reweighted], dim=1)
        visual = self.cross_norm(visual)
        
        # Store CLS states for triplet loss
        cls_states = [visual[:, 0, :]]
        
        # 2 transformer blocks (self-attention only on visual)
        for block in self.blocks:
            # Self-attention
            residual = visual
            x = block['norm1'](visual)
            x, _ = block['attn'](x, x, x, need_weights=False)
            visual = residual + block['drop_path'](x)
            
            # MLP
            residual = visual
            x = block['norm2'](visual)
            x = block['mlp'](x)
            visual = residual + block['drop_path'](x)
            
            cls_states.append(visual[:, 0, :])
        
        if return_cls_states:
            return visual, attn_map, cls_states
        return visual, attn_map


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
        self.prompt_composer = FixedPromptComposer(clip_model)
        self.inversion = InversionNetwork(v_dim=512, token_dim=512)  # CLIP joint embedding dim 512 -> token dim 512

        # Multimodal Interaction Module
        self.interaction = PromptSGInteraction(
            embed_dim=512,
            num_heads=cfg.MODEL.PROMPTSG.CROSS_ATTN_HEADS,
            mlp_ratio=getattr(cfg.MODEL.PROMPTSG, "MLP_RATIO", 4.0),
            attn_drop=getattr(cfg.MODEL.PROMPTSG, "ATTN_DROPOUT", 0.0),
            drop_path=getattr(cfg.MODEL.PROMPTSG, "DROP_PATH", 0.0),
            reweight=getattr(cfg.MODEL.PROMPTSG, "REWEIGHT_MODE", "mul_mean1"),
        )
        
        # Loại bỏ các mode không cần thiết
        # Giữ lại coattn_text_mode để chọn dùng EOT hay full sequence
        self.coattn_text_mode = getattr(cfg.MODEL.PROMPTSG, "COATTN_TEXT_MODE", "eot")
        
        # ========== ADD THESE PROJECTIONS ==========
        # For ResNet50 specific projections
        if self.model_name == 'RN50':
            self.inversion_projection = nn.Linear(1024, 512)  # Project from 1024 to 512 for inversion
            self.resnet_projection = nn.Linear(1024, 512)    # Project CLS token from 1024 to 512
            self.patch_projection = nn.Linear(1024, 512)     # Project patches from 1024 to 512
            self.final_projection = nn.Linear(512, 1024)     # Project back from 512 to 1024 for bottleneck_proj
            self.concat_projection = nn.Linear(512, 1024)    # For inference concatenation
            
            # Initialize these layers
            self.inversion_projection.apply(weights_init_kaiming)
            self.resnet_projection.apply(weights_init_kaiming)
            self.patch_projection.apply(weights_init_kaiming)
            self.final_projection.apply(weights_init_kaiming)
            self.concat_projection.apply(weights_init_kaiming)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        self.text_encoder.eval()

        # Cache for simplified prompt
        self._text_cache = None
        self._text_feat_cached = None

    def _ensure_text_features(self):
        if self._text_cache is None:
            self.prompt_composer._ensure_embeddings()
            with torch.no_grad():
                prompts = self.prompt_composer.embed_simplified
                tokenized = self.prompt_composer.tokenized_simplified
                pooled, tokens, eot_idx = self.text_encoder(prompts, tokenized, return_tokens=True)

            pooled_cpu = pooled.detach().cpu()
            tokens_cpu = tokens.detach().cpu()
            eot_idx_cpu = eot_idx.detach().cpu()
            eot_token_cpu = tokens_cpu[torch.arange(tokens_cpu.size(0)), eot_idx_cpu].unsqueeze(1)  # (1,1,D)

            self._text_cache = {
                "pooled": pooled_cpu,
                "tokens": tokens_cpu,
                "eot_idx": eot_idx_cpu,
                "eot_token": eot_token_cpu,
            }
            self._text_feat_cached = pooled_cpu



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
                features_intermediate, features_final, features_proj = self.image_encoder(x)
                if self.model_name == 'ViT-B-16':
                    v = features_proj[:, 0]  # Already 512
                else:  # RN50
                    v = self.inversion_projection(features_proj[0])  # Project from 1024 to 512
                s_star = self.inversion(v)
                prompts, tokenized = self.prompt_composer(s_star)
                # with torch.no_grad():
                text_features = self.text_encoder(prompts, tokenized)
            return text_features

        # Get image features only
        if get_image:
            features_intermediate, features_final, features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return features_proj[:, 0]

        # Main forward pass for training/inference
        # Get image features from CLIP visual encoder
        features_intermediate, features_final, features_proj = self.image_encoder(x)

        device = x.device
        B = x.size(0)
        
        # Extract features based on backbone type
        if self.model_name == 'ViT-B-16':
            # ViT-B/16: [CLS] token + patch tokens
            CLS_intermediate = features_intermediate[:, 0]  # Intermediate CLS token (768)
            CLS_final = features_final[:, 0]  # Last layer CLS token (768)
            CLS_proj = features_proj[:, 0]  # Projected CLS token (512)
            
            # Patches for cross-attention (exclude CLS token)
            patches = features_proj[:, 1:]  # (batch, num_patches, 512)
            
            # Visual tokens for interaction
            visual_tokens = features_proj  # (B, 1+M, 512)
            
            # Get global visual embedding for inversion network
            v = CLS_proj  # Already 512
            
        elif self.model_name == 'RN50':
            # ResNet50: global feature + spatial features
            CLS_intermediate = F.avg_pool2d(features_intermediate, features_intermediate.shape[2:]).view(x.shape[0], -1)  # (batch, 2048)
            CLS_final = F.avg_pool2d(features_final, features_final.shape[2:]).view(x.shape[0], -1)  # (batch, 2048)
            CLS_proj = features_proj[0]  # Global projected feature (1024)
            
            # Get global visual embedding for inversion network (project to 512)
            v = self.inversion_projection(CLS_proj)  # (batch, 512)
            
            # Prepare patches for cross-attention
            if len(features_proj) > 1:
                b, c, h, w = features_proj[1].shape  # c = 512
                patches = features_proj[1].view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, 512)
                cls_token = self.resnet_projection(CLS_proj).unsqueeze(1)  # (batch, 1, 512)
            else:
                # Fallback: use spatial features and project
                b, c, h, w = features_final.shape  # c = 1024
                patches = features_final.view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, 1024)
                patches = self.patch_projection(patches)  # (batch, h*w, 512)
                cls_token = self.resnet_projection(CLS_proj).unsqueeze(1)  # (batch, 1, 512)
            
            # Visual tokens for interaction
            visual_tokens = torch.cat([cls_token, patches], dim=1)  # (B, 1+M, 512)

        # Generate text features (always composed mode)
        s_star = self.inversion(v)  # Generate pseudo token (512)
        prompts, tokenized = self.prompt_composer(s_star)
        pooled, tokens, eot_idx = self.text_encoder(prompts, tokenized, return_tokens=True)
        text_feat = pooled
        
        # Prepare text tokens for interaction - DETACH to prevent graph reuse errors
        if self.coattn_text_mode == "full":
            text_tokens = tokens.detach()
        elif self.coattn_text_mode == "eot":
            text_tokens = tokens[torch.arange(tokens.size(0), device=tokens.device), eot_idx].unsqueeze(1).detach()
        
        # ========== Multimodal Interaction Module ==========
        # Ensure text_tokens has correct shape
        if text_tokens.dim() == 2:
            text_tokens = text_tokens.unsqueeze(1)  # (B, 1, D) if it's pooled feature
        
        # Call interaction module
        v_tokens_out, attn_map, cls_states = self.interaction(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            return_cls_states=True
        )
        
        # Get final representation from CLS token
        v_final = cls_states[-1]  # CLS token after 2 transformer blocks
        
        # ========== Triplet Loss States ==========
        triplet_feats = [
            cls_states[0],  # Original CLS (after cross-attention reweight)
            cls_states[1],  # After block 1
            cls_states[2]   # After block 2 (final)
        ]
        
        # Prepare features for bottleneck layers
        if self.model_name == 'RN50':
            feat_proj_input = self.final_projection(v_final)  # Project from 512 to 1024
        else:
            feat_proj_input = v_final  # ViT: keep as 512
        
        # ========== Bottleneck Layers ==========
        feat = self.bottleneck(CLS_final)  # CLS_final: CLS x12 - 768
        feat_proj = self.bottleneck_proj(feat_proj_input)

        # ========== Output ==========
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], triplet_feats, v, text_feat
        else:
            if self.neck_feat == 'after':
                # Concatenate features after bottleneck
                return torch.cat([feat, feat_proj], dim=1)
            else:
                # Concatenate original image feature with v_final
                if self.model_name == 'RN50':
                    v_final_concat = self.concat_projection(v_final)  # Project from 512 to 1024
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
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    for key in state_dict.keys():
        print(key)

    # Hoặc tìm keys chứa "text_projection"
    text_proj_keys = [k for k in state_dict.keys() if "text_projection" in k]
    embed_dim = state_dict["text_projection"].shape[1]
    print("embed_dim test: ", embed_dim)
    print("Text projection keys test:", text_proj_keys)

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def make_model(cfg, num_class, camera_num, view_num):
    model = PromptSGModel(num_class, camera_num, view_num, cfg)
    return model
