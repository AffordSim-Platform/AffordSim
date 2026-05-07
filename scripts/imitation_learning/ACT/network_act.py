"""
ACT (Action Chunking with Transformers) .

: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
(Tony Z. Zhao et al., RSS 2023)

:
1. ResNet18Encoder:  ResNet18 + SpatialSoftmax 
2. ACTPolicy: CVAE (Conditional Variational Autoencoder) + Transformer
   - : CVAE encoder  (obs, action)  to  latent z
   - : z ~ N(0, I)
   - Transformer decoder:  obs + z ,  action chunk
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================================
#  ( BC  SpatialSoftmax + ResNet18)
# ============================================================================

class SpatialSoftmax(nn.Module):
    """Spatial Softmax:  (B, C, H, W)  (B, C*2) ."""

    def __init__(self, height, width, num_channels):
        super().__init__()
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("xx", xx.reshape(1, 1, height, width))
        self.register_buffer("yy", yy.reshape(1, 1, height, width))

    def forward(self, features):
        B, C, H, W = features.shape
        weights = F.softmax(features.reshape(B, C, -1), dim=-1).reshape(B, C, H, W)
        expected_x = (weights * self.xx.to(features.device)).sum(dim=[2, 3])
        expected_y = (weights * self.yy.to(features.device)).sum(dim=[2, 3])
        return torch.cat([expected_x, expected_y], dim=-1)


class ResNet18Encoder(nn.Module):
    """
    ResNet18 encoder with Spatial Softmax.

    : (B, H, W, C) float [0, 1]
    : (B, out_dim)
    """

    def __init__(self, in_channels=3, out_dim=256, finetune_from="all"):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Spatial Softmax on layer4 output (512 channels, 8x8 for 256x256 input)
        self.spatial_softmax = SpatialSoftmax(8, 8, 512)
        self.fc = nn.Linear(1024, out_dim)

        # ImageNet normalization
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 
        for param in self.parameters():
            param.requires_grad = False

        unfreeze_layers = {
            "layer4": [self.layer4],
            "layer3": [self.layer3, self.layer4],
            "layer2": [self.layer2, self.layer3, self.layer4],
            "all": [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4],
        }
        for layer in unfreeze_layers.get(finetune_from, [self.layer3, self.layer4]):
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.spatial_softmax.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, image):
        # image: (B, H, W, C) float [0, 1]  to  (B, C, H, W), ImageNet normalize
        x = image.permute(0, 3, 1, 2)
        x = (x - self.img_mean.to(x.device)) / self.img_std.to(x.device)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.spatial_softmax(x)
        x = self.fc(x)
        return x


# ============================================================================
# Positional Encoding
# ============================================================================

class SinusoidalPositionEncoding(nn.Module):
    """."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)].to(x.device)


# ============================================================================
# ACT Policy (CVAE + Transformer)
# ============================================================================

class ACTPolicy(nn.Module):
    """
    ACT (Action Chunking with Transformers) .

    :
      1. :  ResNet18 + SpatialSoftmax
      2. CVAE Encoder ():
         - : [CLS] + action_tokens ()
         - Cross-attention to obs_tokens ( + )
         - CLS token  to  mu, log_var  to  z ~ N(mu, var)
      3. Transformer Decoder:
         - : action queries (learnable) + z ( query)
         - Cross-attention to obs_tokens
         - : action chunk (C, action_dim)

    Loss:
      - L1 reconstruction loss (action prediction)
      - KL divergence (latent regularization)
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 7,
        action_chunk_size: int = 8,
        visual_feat_dim: int = 256,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 7,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # ---  ---
        self.front_encoder = ResNet18Encoder(in_channels=3, out_dim=visual_feat_dim)
        self.wrist_encoder = ResNet18Encoder(in_channels=3, out_dim=visual_feat_dim)

        # ---  ---
        #  hidden_dim ( token)
        self.img_proj = nn.Linear(visual_feat_dim, hidden_dim)
        #  (1  token)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # --- CVAE Encoder () ---
        #  action  hidden_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        # Action positional encoding
        self.encoder_pos = SinusoidalPositionEncoding(hidden_dim, max_len=action_chunk_size + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # CVAE: CLS  to  mu, log_var
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, latent_dim)

        # --- Transformer Decoder ---
        # Latent z  to  hidden_dim ( action queries )
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Learnable action queries
        self.action_queries = nn.Parameter(torch.zeros(1, action_chunk_size, hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)

        # Decoder positional encoding
        self.decoder_pos = SinusoidalPositionEncoding(hidden_dim, max_len=action_chunk_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # : hidden_dim  to  action_dim
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # 
        self._init_weights()

    def _init_weights(self):
        for m in [self.img_proj, self.state_proj, self.action_proj,
                  self.latent_mu, self.latent_log_var, self.latent_proj,
                  self.action_head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_obs(self, front, wrist, state):
        """ token .

        Args:
            front: (B, H, W, C) float [0, 1]
            wrist: (B, H, W, C) float [0, 1]
            state: (B, state_dim)

        Returns:
            obs_tokens: (B, 3, hidden_dim)  [front_token, wrist_token, state_token]
        """
        feat_front = self.front_encoder(front)  # (B, visual_feat_dim)
        feat_wrist = self.wrist_encoder(wrist)  # (B, visual_feat_dim)

        front_token = self.img_proj(feat_front).unsqueeze(1)  # (B, 1, hidden_dim)
        wrist_token = self.img_proj(feat_wrist).unsqueeze(1)  # (B, 1, hidden_dim)
        state_token = self.state_proj(state).unsqueeze(1)     # (B, 1, hidden_dim)

        obs_tokens = torch.cat([front_token, wrist_token, state_token], dim=1)
        return obs_tokens

    def _encode_cvae(self, obs_tokens, actions):
        """CVAE Encoder:  (obs, action)  latent z.

        Args:
            obs_tokens: (B, N_obs, hidden_dim)
            actions: (B, C, action_dim)

        Returns:
            mu: (B, latent_dim)
            log_var: (B, latent_dim)
        """
        B = actions.shape[0]

        #  action  hidden_dim
        action_tokens = self.action_proj(actions)  # (B, C, hidden_dim)

        # [CLS, action_1, ..., action_C]
        cls = self.cls_token.expand(B, -1, -1)
        encoder_input = torch.cat([cls, action_tokens], dim=1)  # (B, 1+C, hidden_dim)
        encoder_input = self.encoder_pos(encoder_input)

        # Self-attention (CLS  action )
        #  encoder,  cross-attention to obs
        # ACT : encoder  [CLS, action_tokens],  cross-attention
        encoder_output = self.cvae_encoder(encoder_input)

        # CLS token  to  latent distribution
        cls_output = encoder_output[:, 0]  # (B, hidden_dim)
        mu = self.latent_mu(cls_output)
        log_var = self.latent_log_var(cls_output)

        return mu, log_var

    def _reparameterize(self, mu, log_var):
        """."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, obs_tokens, z):
        """Transformer Decoder:  obs + z  action chunk.

        Args:
            obs_tokens: (B, N_obs, hidden_dim)  tokens
            z: (B, latent_dim) latent code

        Returns:
            actions: (B, C, action_dim)  action chunk
        """
        B = obs_tokens.shape[0]

        # Latent z  to  hidden_dim,  action query 
        z_proj = self.latent_proj(z).unsqueeze(1)  # (B, 1, hidden_dim)

        # Action queries + z
        queries = self.action_queries.expand(B, -1, -1) + z_proj  # (B, C, hidden_dim)
        queries = self.decoder_pos(queries)

        # Cross-attention: queries attend to obs_tokens
        decoded = self.transformer_decoder(queries, obs_tokens)  # (B, C, hidden_dim)

        #  action
        actions = self.action_head(decoded)  # (B, C, action_dim)
        return actions

    def forward(self, front, wrist, state, actions=None):
        """.

        :
            front, wrist, state, actions  to  (pred_actions, mu, log_var)
        :
            front, wrist, state, actions=None  to  pred_actions (z ~ N(0, I))

        Args:
            front: (B, H, W, C) float [0, 1]
            wrist: (B, H, W, C) float [0, 1]
            state: (B, state_dim)
            actions: (B, C, action_dim) or None

        Returns:
            : (pred_actions, mu, log_var)
            : pred_actions
        """
        # 
        obs_tokens = self._encode_obs(front, wrist, state)

        if actions is not None:
            # : CVAE encode  to  reparameterize  to  decode
            mu, log_var = self._encode_cvae(obs_tokens, actions)
            z = self._reparameterize(mu, log_var)
            pred_actions = self._decode(obs_tokens, z)
            return pred_actions, mu, log_var
        else:
            # : z ~ N(0, I)
            B = front.shape[0]
            z = torch.zeros(B, self.latent_dim, device=front.device)
            pred_actions = self._decode(obs_tokens, z)
            return pred_actions

    def get_action(self, front, wrist, state):
        """,  action chunk.

        Args:
            front: (1, H, W, C)
            wrist: (1, H, W, C)
            state: (1, state_dim)

        Returns:
            (1, C, action_dim) action chunk
        """
        with torch.no_grad():
            return self.forward(front, wrist, state, actions=None)
