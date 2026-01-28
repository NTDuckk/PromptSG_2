import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
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


class CrossAttentionGuidance(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, text_feat: torch.Tensor, patch_tokens: torch.Tensor):
        q = text_feat.unsqueeze(1)
        _, attn_weights  = self.attn(q, patch_tokens, patch_tokens, need_weights=True, average_attn_weights=False)
        attn_map = attn_weights.mean(dim=1)  # (batch, 1, num_patches)
        attn_map = attn_map.squeeze(1)      # (batch, num_patches)
        return attn_map


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
        
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        
        # Classifiers
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        # Bottlenecks
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
        self.inversion = InversionNetwork(dim=512)

        # Multimodal Interaction Module (MIM)
        self.cross_guidance = CrossAttentionGuidance(embed_dim=512, num_heads=cfg.MODEL.PROMPTSG.CROSS_ATTN_HEADS)
        
        # Post cross-attention transformer blocks (2 layers as in paper)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=cfg.MODEL.PROMPTSG.CROSS_ATTN_HEADS,
            dim_feedforward=2048,
            activation='gelu',
            batch_first=True,
        )
        self.post_blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.MODEL.PROMPTSG.POST_CA_BLOCKS)
        
        # Cache for simplified prompt
        self._text_feat_cached = None

    def _ensure_text_features(self):
        """Cache text features for simplified prompt to avoid recomputation"""
        if self._text_feat_cached is None:
            with torch.no_grad():
                # Create simplified prompt: "A photo of a person"
                dummy_pseudo_token = torch.zeros(1, 512, device=next(self.parameters()).device)
                prompts, tokenized = self.prompt_composer(dummy_pseudo_token)
                self._text_feat_cached = self.text_encoder(prompts, tokenized).detach().cpu()

    def forward(self, x, label=None, get_image=False, get_text=False, cam_label=None, view_label=None):
        """
        Forward pass of PromptSG model
        
        Args:
            x: input images
            label: labels for training
            get_image: if True, return image features only
            get_text: if True, return text features only
            cam_label: camera labels
            view_label: view labels
        """
        # Get text features only
        if get_text:
            if self.prompt_mode == 'simplified':
                self._ensure_text_features()
                text_features = self._text_feat_cached.to(device=x.device).expand(x.shape[0], -1)
            else:
                # For composed prompt, need to generate pseudo token first
                image_features_last, image_features, image_features_proj = self.image_encoder(x)
                v = image_features_proj[:, 0] if self.model_name == 'ViT-B-16' else image_features_proj[0]
                s_star = self.inversion(v)
                prompts, tokenized = self.prompt_composer(s_star)
                with torch.no_grad():
                    text_features = self.text_encoder(prompts, tokenized)
            return text_features

        # Get image features only
        if get_image:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        # Main forward pass for training/inference
        # Get image features from CLIP visual encoder
        image_features_last, image_features, image_features_proj = self.image_encoder(x)
        
        # Extract features based on backbone type
        if self.model_name == 'ViT-B-16':
            # ViT-B/16: [CLS] token + patch tokens
            img_feature_last = image_features_last[:, 0]  # Last layer CLS token
            img_feature = image_features[:, 0]  # Intermediate CLS token
            img_feature_proj = image_features_proj[:, 0]  # Projected CLS token
            
            # Patches for cross-attention (exclude CLS token)
            patches = image_features_proj[:, 1:]  # (batch, num_patches, embed_dim)
            
            # CLS token for final sequence
            cls_token = image_features_proj[:, :1]  # (batch, 1, embed_dim)
            
        elif self.model_name == 'RN50':
            # ResNet50: global feature + spatial features
            # Global features
            img_feature_last = F.avg_pool2d(image_features_last, image_features_last.shape[2:]).squeeze()
            img_feature = F.avg_pool2d(image_features, image_features.shape[2:]).squeeze()
            img_feature_proj = image_features_proj[0]  # Global projected feature
            
            # For ResNet, we need to create "patches" from spatial features
            # image_features_proj[1] has shape (batch, 1024, h, w)
            b, c, h, w = image_features_proj[1].shape
            patches = image_features_proj[1].view(b, c, -1).permute(0, 2, 1)  # (batch, h*w, 512)
            
            # Create CLS token from global average pooling
            cls_token = img_feature_proj.unsqueeze(1)  # (batch, 1, 512)
        
        # Get global visual embedding for inversion network
        v = img_feature_proj

        # Generate text features based on prompt mode
        if self.prompt_mode == 'simplified':
            # Use fixed prompt: "A photo of a person"
            self._ensure_text_features()
            text_feat = self._text_feat_cached.to(device=x.device).expand(x.shape[0], -1)
        else:
            # Use composed prompt with pseudo token
            s_star = self.inversion(v)  # Generate pseudo token
            prompts, tokenized = self.prompt_composer(s_star)
            with torch.no_grad():
                text_feat = self.text_encoder(prompts, tokenized)

        # ========== Multimodal Interaction Module (MIM) ==========
        # Cross-attention: text features (Q) attend to visual patches (K, V)
        attention_map = self.cross_guidance(text_feat, patches)  # (batch, num_patches)
        
        # Re-weight patches with attention weights (Equation in paper)
        # This is the semantic guidance: patches aligned with text get higher weights
        patches_weighted = patches * attention_map.unsqueeze(-1)
        
        # ========== Construct Sequence for Transformer Blocks ==========
        # As in paper: CLS token + re-weighted patches
        seq = torch.cat([cls_token, patches_weighted], dim=1)
        
        # ========== Post Cross-Attention Transformer Blocks ==========
        # 2 transformer blocks as described in paper (after Eq. 7)
        seq = self.post_blocks(seq)
        
        # Extract final representation from CLS token (index 0)
        v_final = seq[:, 0]  # (batch, embed_dim)
        
        # ========== Bottleneck Layers ==========
        feat = self.bottleneck(v_final)
        feat_proj = self.bottleneck_proj(v_final)

        # ========== Output ==========
        if self.training:
            # Training mode: return classification scores and features
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            
            # Return format based on original ReID code
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, v_final], v_final
        else:
            # Inference mode
            if self.neck_feat == 'after':
                # Concatenate features after bottleneck
                return torch.cat([feat, feat_proj], dim=1)
            else:
                # Concatenate original image feature with final representation
                return torch.cat([img_feature, v_final], dim=1)

    def load_param(self, trained_path):
        """Load pretrained parameters"""
        param_dict = torch.load(trained_path, map_location='cpu')
        for key in param_dict:
            # Remove 'module.' prefix if exists (from DataParallel)
            new_key = key.replace('module.', '')
            if new_key in self.state_dict():
                self.state_dict()[new_key].copy_(param_dict[key])

def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    from .clip import clip
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def make_model(cfg, num_class, camera_num, view_num):
    model = PromptSGModel(num_class, camera_num, view_num, cfg)
    return model
