import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    Uses nn.Unfold + Linear projection for flexibility.
    Adds positional embedding and optional cls token.

    Shape Example:
        if input (2, 3, 16, 16) -> batch size, channels, width, height
        and if PatchEmbedding(img_size=16, patch_size=2, in_channels=3, embed_dim=768)
        tensor with cls ends up as -> (2, 64+1, 768) -> batch size, num_patches + cls, embed dim
        *num_patches here is 64 because we have 2 batches of 2x2 patches (img_size=16 / 2 = 8. 8 * 2(batches) = 64. 64 + 1 (cls token) = 65)

    """
    def __init__(self, img_size: int = 8, patch_size: int = 1, in_channels: int = 64, embed_dim: int = 768, add_cls_token: bool = True, drop_rate: float = 0.0):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by the patch size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2 # floor division, squared. this tells us how many whole patches fit along each dimension.

        # Unfold image into (B, C*P*P, N) where N = num_patches
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        # Linear projection from flattened patch to embedding dim
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim, bias = True) # may be worth testing with bias = False

        self.add_cls_token = add_cls_token
        total_patches = self.num_patches + 1 if add_cls_token else self.num_patches
        if add_cls_token:
            # Classification token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches, embed_dim))
        self.dropout = nn.Dropout(p=drop_rate)
        self._init_weights()

    def _init_weights(self):
        # Weight initialization as in original ViT paper
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # fills tensor in place with values from a truncated normal distribution (keeps weights relatively small)
        if self.add_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight) # xavier initialization for projection weights
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x: Tensor shape (B, C, H, W)
        Returns:
            Tensor shape (B, N + 1, embed_dim) if add_cls_token else (B, N, embed_dim)
        Note*: here we use B for batch size, and N for num_patches.
        """
        B, C, H, W = x.shape
        # unfold -> (B, C*P*P, num_patches * num_patches)
        patches = self.unfold(x)
        # transpose -> (B, num_patches, C*P*P)
        patches = patches.transpose(1, 2)
        # projection -> (B, num_patches, embed_dim)
        x = self.proj(patches)
        # optionally prepend cls token
        """
        cls (classification token) -> borrowed from BERT/Vision-Transformer designs
        1) creates *learnable* vector (cls_token)
        2) *expand* it so that each batch gets its own copy and *prepend* it to the sequence of patch embeddings.
        3) at the *end* of the Transformer encoder, takes the *output* corresponding to that first "cls" position as a summary representation of the *entire* image.
        Shape Example:
            Suppose num_patches = 64, embed_dim = 768, B = 4
            x.shape  # → (4, 64, 768)   before adding CLS
            # with CLS token:
            x.shape  # → (4, 65, 768)
        """
        if self.add_cls_token:
            # expand cls_token so each batch gets its own copy
            cls_token = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
            x = torch.cat((cls_token, x), dim=1) # (B, num_patches+1, embed_dim)

        # add positional embedding
        """
        Positional embedding gives the transformer a sense of *where* each patch belongs in the image.
        """
        #x = x + self.pos_embed # (B, num_patches+1, embed_dim)
        x = self.dropout(x)
        return x