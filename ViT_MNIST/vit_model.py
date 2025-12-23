import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2)  # Flattening the output
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels, hidden_dims=None):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)

        # If hidden_dims is provided (list), use it. Otherwise use fixed hidden_dim for all layers
        if hidden_dims is None:
            hidden_dims = [hidden_dim] * num_encoders
        
        # Verify length
        if len(hidden_dims) != num_encoders:
            raise ValueError(f"hidden_dims length ({len(hidden_dims)}) must match num_encoders ({num_encoders})")

        self.encoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=h_dim,
                dropout=dropout, 
                activation=activation, 
                batch_first=True, 
                norm_first=True
            ) for h_dim in hidden_dims
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
        
        # Store config for reconstruction
        self.config = {
            'num_patches': num_patches,
            'img_size': img_size,
            'num_classes': num_classes,
            'patch_size': patch_size,
            'embed_dim': embed_dim,
            'num_encoders': num_encoders,
            'num_heads': num_heads,
            'hidden_dim': hidden_dim, # Base hidden dim
            'dropout': dropout,
            'activation': activation,
            'in_channels': in_channels,
            'hidden_dims': hidden_dims # Actual hidden dims
        }

    def forward(self, x):
        x = self.embeddings_block(x)
        for layer in self.encoder_blocks:
            x = layer(x)
        x = self.mlp_head(x[:, 0, :])
        return x

    def get_layer_ffn_activations(self, x):
        """
        Returns a dictionary of activations for the FFN hidden layer of each encoder block.
        We hook into the first Linear layer of the FFN (linear1 in TransformerEncoderLayer).
        """
        activations = {}
        hooks = []
        
        def get_hook(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks
        for i, layer in enumerate(self.encoder_blocks):
            hooks.append(layer.linear1.register_forward_hook(get_hook(f"encoder_{i}")))

        # Run forward
        with torch.no_grad():
            self.forward(x)
        
        # Remove hooks
        for h in hooks:
            h.remove()
            
        return activations
