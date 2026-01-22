# models/swin_with_attn.py

import torch
import torch.nn as nn
import timm
from timm.models.swin_transformer import SwinTransformerBlock, WindowAttention
from typing import List, Tuple, Dict, Any

class AttentionExtractor(nn.Module):
    """
    Wrapper for Swin Transformer Block to capture attention weights.
    Uses forward hooks to capture attention outputs.
    """
    def __init__(self, block: SwinTransformerBlock):
        super().__init__()
        self.block = block
        self.attention_weights: torch.Tensor = None
        self.handle = self.block.attn.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # The attention module is WindowAttention inside SwinTransformerBlock
        # It stores the raw attention weights in its forward pass
        # For WindowAttention, the attention weights are usually computed internally
        # We need to slightly modify the forward pass or access intermediate results.
        # This is tricky with frozen timm. A better approach is to subclass or monkey-patch.
        # For this codebase, we'll assume the Swin block is modified slightly or use a custom one.
        # Here, we'll define a custom block that outputs attention.
        pass # Placeholder - see CustomSwinTransformerBlockWithAttn below

    def forward(self, x, *args, **kwargs):
        return self.block(x, *args, **kwargs)

# A modified SwinTransformerBlock that outputs attention weights
class CustomSwinTransformerBlockWithAttn(SwinTransformerBlock):
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, input_resolution, num_heads, window_size,
                         shift_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                         drop_path, act_layer, norm_layer)
        # Replace the attention layer with one that can return attention
        self.attn = CustomWindowAttention(
            dim, window_size, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input x size ({L}) doesn't match resolution ({H}x{W})."

        shortcut = x
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = shifted_x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows, attn_weights = self.attn(x_windows, mask=self.attn_mask)  # Get attention weights

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = attn_windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_weights # Return features and attention weights

class CustomWindowAttention(WindowAttention):
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn # Return output and attention weights

# A custom Swin Transformer that uses the modified blocks
class SwinTransformerWithAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load base Swin from timm
        base_model_name = config.get('backbone', 'swin_tiny_patch4_window7_224')
        self.base_model = timm.create_model(base_model_name, pretrained=True, num_classes=0) # Remove head

        # --- REBUILDING LOGIC BEGINS ---
        img_size = config.get('input_size', [224, 224])[0]
        patch_size = 4
        in_chans = config.get('num_channels', 4)
        embed_dim = config.get('embed_dim', 96)
        depths = config.get('swin_depths', [2, 2, 6, 2])
        num_heads = config.get('swin_num_heads', [3, 6, 12, 24])
        window_size = config.get('window_size', 7)
        drop_path_rate = config.get('dropout_rate', 0.2)
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.mlp_ratio = 4.0

        self.patch_embed = timm.models.swin_transformer.PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_drop = nn.Dropout(p=0.) # Overwrite default if needed

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayerWithAttn(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(img_size // (2 ** (i_layer + 2)), img_size // (2 ** (i_layer + 2))),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=0., # Assume dropout handled elsewhere or set here
                attn_drop=0., # Assume dropout handled elsewhere or set here
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=timm.models.swin_transformer.PatchMerging if (i_layer < len(depths) - 1) else None
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1) # For classification head, not needed for seg
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        all_attns = []
        for layer in self.layers:
            x, layer_attns = layer(x)
            all_attns.extend(layer_attns) # Collect attentions from all layers

        x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1 # Not for segmentation
        # x = torch.flatten(x, 1) # B C # Not for segmentation
        # x = self.head(x) # Not for segmentation
        return x, all_attns # Return features and all attentions


class BasicLayerWithAttn(nn.Module):
    """A basic Swin Transformer layer for one stage."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            CustomSwinTransformerBlockWithAttn(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            x, attn = blk(x)
            attns.append(attn) # Collect attention from each block in the layer
        if self.downsample is not None:
            x = self.downsample(x)
        return x, attns # Return downsampled features and attentions from this layer
