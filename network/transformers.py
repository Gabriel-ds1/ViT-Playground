import torch
import torch.nn as nn
from .conv_stem import ConvStem
from .patch_embedding import PatchEmbedding
from .utils import DropPath

class MultiHeadSelfAttention(nn.Module):
    """
    args:
        embed_dim: width of token vectors throughout Transformer stack (similar to PatchEmbeddings embed_dim)
        num_heads: how many parallel "attention heads" you want. Splits embedding into smaller chunks, each attend separately, then recombine.
        attn_drop: dropout rate for attention layer
        proj_drop: dropout rate for projection layer
    returns:
        tensor of shape (B, N + 1, embed_dim) if add_cls token else (B, N, embed_dim) 
            -B: batch and N: number of patches per image
        
    """
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        # size of each individual heads sub-embedding, so if embed_dim=768 and num_heads=12, each head would have 64-dimensional "slice" of the full 768 dim vector.
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # queries, keys, and values (scaled-dot-product attention), 3 linear layers bundled into one.
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias = False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x:torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        """
        weights of q -> projects input embeddings into queries
        weights of k -> projects into keys
        weights of v -> projects into values
        these projections are where the layer "learns" how to best represent each patch "how much should patch i attend to patch j?"

        Dot product attention -> 'importance scores', the weight you give to patch j when building updated representation of patch i
        ^ -> by using weights of queries and keys, the model figures out what features of the patches matter for comparing them-- thus which patches are "important to each other"

        Values and weighted sum -> attn @ v  -- weights of values (v) decides what information you actually carry over from each patch into that weighted sum

        Multi-head -> multiple 'views' of importance -- splitting into num_heads lets the model learn *different* comparison functions in parallel

        Final projection -> self.proj (with its own weights and biases) mixes all the heads' outputs back together and reshapes for next transformer block.
        """
        B, N, C = x.shape # batch, tokens, embeddings
        # apply qkv to input
        qkv = self.qkv(x) # (B, N, 3*C)
        # reshape and permute so you can slice it into q, k, and v of shape (B, num_heads, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, N, head_dim)

        # compute raw similarity score betweene very query and every key within each head, then scale down to keep variance stable
        # so that softmax gradients dont vanish or explode
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, heads, N, N)
        # turn each row into a probability distribution over "which key each query should attend to"
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # dropout random attention weights to regularize

        # multiply ^ weights by the values (v) to get weighted sum of each query-i.e. the attention output for each token, per head.
        out = attn @ v # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C) # (B, N, embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn
        return out
    
class TransformerEncoderBlock(nn.Module):
    """
    A single transformer encoder block for ViT, consisting of multi-head self-attention,
    pre-layer normalization, MLP, and optional stochastic depth.

    Args:
        embed_dim (int): Dimensionality of token embeddings.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP hidden layer size as multiple of embed_dim.
        drop (float): Dropout rate after attention and MLP.
        attn_drop (float): Dropout rate within attention mechanism.
        drop_path_prob (float): Probability of dropping entire residual branch (stochastic depth).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0, drop_path_prob: float = 0.0):
        super().__init__()
        # LayerNorm before attention
        self.norm1 = nn.LayerNorm(embed_dim)
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, drop)
        # Stochastic Depth (DropPath)
        # stochastic depth is a regularization trick where, during training, entire residual branches are randomly dropped with some probability
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), 
                                 nn.LayerNorm(hidden_dim),
                                 nn.SiLU(),
                                 nn.Dropout(drop),
                                 nn.Linear(hidden_dim, embed_dim),
                                 nn.Dropout(drop))
    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        # optionally return attention alongside output to later visualize *where* the model looked.
        if return_attn:
            attn_out, attn_map = self.attn(self.norm1(x), return_attn=True)
        else:
            attn_out = self.attn(self.norm1(x), return_attn=False)
            attn_map = None

        # 1) Multi-head attention block with pre-norm + residual + (stochastic) drop-path (adding x (x + ..) makes this residual)
        x = x + self.drop_path(attn_out) # x + (normalize x and apply multi head attention into placeholder drop_path)
        # 2) MLP block with pre-norm + residual + (stochastic) drop-path (adding x (x + ..) makes this residual)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # ^^^x + (normalize x and apply MLP into placeholder drop_path)
        if return_attn:
            return x, attn_map
        return x

class ViT(nn.Module):
    """
    Vision Transformer model for image classification, with optional convolutional stem, 
    learnable positional encoding, and stack of transformer encoder blocks.

    Args:
        img_size (int): Input image size (assumes square images).
        patch_size (int): Size of each image patch.
        in_channels (int): Number of image channels.
        num_classes (int): Number of target classes for classification head.
        embed_dim (int): Token embedding dimension.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of self-attention heads per block.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dim.
        drop_rate (float): Dropout rate throughout.
        attn_drop_rate (float): Dropout rate for attention.
        use_conv_stem (bool): If True, use convolutional stem before patch embedding.
        conv_stem_dim (int): Channel dimension output by conv stem.
    """
    def __init__(self, *, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, num_classes: int = 100, embed_dim: int = 768, 
                 depth: int = 12, num_heads: int = 8, mlp_ratio: float = 4.0, drop_rate: float = 0.1, attn_drop_rate: float = 0.0, use_conv_stem=True, conv_stem_dim=64) -> None:
        super().__init__()

        # for capturing activations
        self.activations = {}
        # Hook function generator
        def _hook(name):
            def fn(module, inp, out):
                self.activations[name] = out
            return fn

        self.use_conv_stem = use_conv_stem # flag to train using conv stem before converting to patches for ViT
        if self.use_conv_stem:
            self.conv_stem = ConvStem(in_channels=in_channels, hidden_dim=conv_stem_dim)
            conv_out_channels = conv_stem_dim
            img_size = 8
            patch_size = 1
        else:
            self.conv_stem = None
            conv_out_channels = in_channels

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=conv_out_channels, embed_dim=embed_dim,
                                          add_cls_token=True, drop_rate=drop_rate)
        # number of patches per image, +1 computes correct length so that it matches with the +1 of cls.
        total_patches = self.patch_embed.num_patches + 1

        # Override pos_embed with correct size
        # this is a new pos_embed (separate from PatchEmbedding) so that we have one *single* 
        # positional-encoding table that covers exactly the sequence length used throughout the model. (patches + CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim)) # ensures we learn one vector per position (0 = CLS, 1...64 = patches) for the full stack of transformer blocks below
        self.dropout = nn.Dropout(drop_rate)

        # hook the sequence
        self.dropout.register_forward_hook(_hook('after_pos_drop'))

        # Transformer blocks, if depth=12, stacks 12 TransformerEncoder blocks
        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio = mlp_ratio, drop=drop_rate, 
                                                             attn_drop=attn_drop_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # Classification head. Extracts the CLS token's final embedding and passes it through. 
        self.head = nn.Linear(embed_dim, num_classes, bias=True) # (B, num_classes) , worth trying out bias=False

        #hook the final norm
        self.norm.register_forward_hook(_hook('post_norm'))
        # hook pre-head (right before head)
        self.head.register_forward_hook(_hook('pre_head'))

        # Register hooks on modules we want to inspect
        self.patch_embed.proj.register_forward_hook(_hook('patch_projection'))
        for i, blk in enumerate(self.blocks):
            blk.attn.qkv.register_forward_hook(_hook(f'attn_qkv_block{i}')) # may be overkill, but can test out anyways for now
            blk.attn.register_forward_hook(_hook(f'attn_out_block{i}'))
            blk.mlp.register_forward_hook(_hook(f'mlp_out_block{i}'))

        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights (per ViT paper defaults).
        """
        # Weight initialization as in original ViT paper
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # fills tensor in place with values from a truncated normal distribution (keeps weights relatively small)
        nn.init.xavier_uniform_(self.head.weight) # xavier initialization for classification head
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor, return_attn: bool = False, return_tokens: bool = False):
        """
        Compute feature representations for input images, returning embeddings, and optionally
        attention maps and intermediate tokens for visualization.

        Args:
            x (Tensor): Input images of shape (batch, channels, height, width).
            return_attn (bool): If True, collect and return all attention maps from each block.
            return_tokens (bool): If True, collect and return all intermediate token representations.

        Returns:
            Tensor or tuple: Final token sequences, plus attention maps and all tokens if requested.
                - If neither optional flag is set: returns only the final tokens.
                - If return_attn and/or return_tokens are True: returns (final_tokens, [...attn_maps...], [...tokens...]).
        """
        # token_seqs structure:
        # [0]: after patch embedding
        # [1]: after pos_embed+dropout
        # [2]: after block 0
        # ...
        # [N]: after final block (depth-1)
        # [N+1]: after final LayerNorm

        # ----- attn_maps list structure -----
        # attn_maps[0]: attention from Transformer block 0
        # attn_maps[1]: attention from Transformer block 1
        # ...
        # attn_maps[depth-1]: attention from last Transformer block (block depth-1)
        # these are only for the transformer's main blocks.

        all_tokens = []
        all_attn_maps = [] # collect attention maps if needed

        # first train using conv stem
        if self.use_conv_stem:
            x = self.conv_stem(x)

        x = self.patch_embed(x) # (B, num_patches+1, D)

        if return_tokens:
            all_tokens.append(x)

        x = x + self.pos_embed # add learned positional encodings
        x = self.dropout(x)
        if return_tokens:
            all_tokens.append(x)

        for blk in self.blocks: # depth x (attention + MLP)
            if return_attn:
                # make sure block returns (out, attn_map)
                x, attn_map = blk(x, return_attn=True)
                all_attn_maps.append(attn_map)
            else:
                x = blk(x, return_attn=False)
            if return_tokens:
                all_tokens.append(x)

        x = self.norm(x) # final LayerNorm
        if return_tokens:
            all_tokens.append(x)

        # decide what to return
        outputs = []
        if return_attn:
            outputs.append(all_attn_maps)
        if return_tokens:
            outputs.append(all_tokens)

        if outputs:
            return (x, *outputs)
        return x

    def forward(self, x: torch.Tensor, return_attn: bool = False, return_tokens: bool = False):
        """
        Forward pass of the ViT model, producing class logits and optionally attention maps and token embeddings.

        Args:
            x (Tensor): Input images, shape (batch, channels, height, width).
            return_attn (bool): If True, also return all attention maps from transformer blocks.
            return_tokens (bool): If True, also return all intermediate and final token embeddings.

        Returns:
            Tensor or tuple: 
                - If no flags: logits (batch, num_classes).
                - If return_attn: (logits, attn_maps).
                - If return_tokens: (logits, token_seqs).
                - If both: (logits, attn_maps, token_seqs).
        """
        features = self.forward_features(x, return_attn, return_tokens)
        if return_attn and return_tokens:
            x_final, attn_maps, token_seqs = features
        elif return_attn:
            x_final, attn_maps = features
            token_seqs = None
        elif return_tokens:
            x_final, token_seqs = features
            attn_maps = None
        else:
            x_final = features
            attn_maps = token_seqs = None

        cls_token = x_final[:, 0] # (B, embed_dim) , takes only the CLS token
        logits = self.head(cls_token) # (B, num_classes), the logits

        # return the right tuple
        ret = [logits]
        if return_attn:
            ret.append(torch.stack(attn_maps))
        if return_tokens:
            ret.append(token_seqs)
        return tuple(ret) if len(ret) > 1 else logits