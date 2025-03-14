
#model 1, transformer encoder after convtds

"""
'test/loss': 0.8595742583274841,
'test/CER': 26.172466278076172

self.model = nn.Sequential(
    # (T, N, bands=2, C=16, freq)
    SpectrogramNorm(channels=self.NUM_BANDS * ELECTRODE_CHANNELS),
    # (T, N, bands=2, mlp_features[-1])
    MultiBandRotationInvariantMLP(
        in_features=in_features,
        mlp_features=mlp_features,
        num_bands=self.NUM_BANDS,
    ),
    # (T, N, num_features)
    nn.Flatten(start_dim=2),
    TDSConvEncoder(
        num_features=num_features,
        block_channels=block_channels,
        kernel_width=kernel_width,
    ),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=num_features,  # Embedding size
            nhead=4,  # Number of attention heads
            dim_feedforward=num_features * 4,  # Feedforward network size
            dropout=0.1
        ),
        num_layers=2  # Number of Transformer layers
    ),
    # (T, N, num_classes)
    nn.Linear(num_features, charset().num_classes),
    nn.LogSoftmax(dim=-1),
)
"""

#model 2, transformer encoder before convtds per band

"""
'test/loss': 0.809109091758728,
'test/CER': 25.58893394470215,

self.model = nn.Sequential(
    # (T, N, bands=2, C=16, freq)
    SpectrogramNorm(channels=self.NUM_BANDS * ELECTRODE_CHANNELS),
    # (T, N, bands=2, mlp_features[-1])
    MultiBandRotationInvariantMLP(
        in_features=in_features,
        mlp_features=mlp_features,
        num_bands=self.NUM_BANDS,
    ),
    PerBandTransformer(
        d_model=mlp_features[-1], 
        nhead=4, 
        dim_feedforward=mlp_features[-1] * 4, 
        num_layers=2
    ),
    # (T, N, num_features)
    nn.Flatten(start_dim=2),
    TDSConvEncoder(
        num_features=num_features,
        block_channels=block_channels,
        kernel_width=kernel_width,
    ),
    # (T, N, num_classes)
    nn.Linear(num_features, charset().num_classes),
    nn.LogSoftmax(dim=-1),
)

class PerBandTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=0.1
            ), 
            num_layers=num_layers
        )

    def forward(self, x):
        # x shape: (T, N, bands, features)
        bands = torch.unbind(x, dim=2)  # List of tensors [(T, N, features) for each band]
        bands = [self.transformer(band) for band in bands]  # Process each band separately
        return torch.stack(bands, dim=2)  # Re-stack along the bands dimension
"""

#model 2, n_head=16 

""" result: overfit more than n_head=4
'test/loss': 0.8230770230293274,
'test/CER': 25.848281860351562,
"""

#model 2, dim_feedforward=mlp_features[-1] * 2

""" result: underfit more than dim_feedforward=mlp_features[-1] * 4
'test/loss': 0.8386710286140442,
'test/CER': 26.66954803466797,
"""

#model 3 transformer encoder before mlp with flattened bands

""" best result with transformers so far
'test/loss': 0.7799743413925171,
'test/CER': 24.854116439819336,

self.model = nn.Sequential(
    # (T, N, bands=2, C=16, freq)
    SpectrogramNorm(channels=self.NUM_BANDS * ELECTRODE_CHANNELS),

    nn.Flatten(start_dim=2),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=in_features*2,  # Embedding size
            nhead=4,  # Number of attention heads
            dim_feedforward=in_features,  # Feedforward network size
            dropout=0.1
        ),
        num_layers=2  # Number of Transformer layers
    ),
    ReshapeBands(),

    # (T, N, bands=2, mlp_features[-1])
    MultiBandRotationInvariantMLP(
        in_features=in_features,
        mlp_features=mlp_features,
        num_bands=self.NUM_BANDS,
    ),
    # (T, N, num_features)
    nn.Flatten(start_dim=2),
    TDSConvEncoder(
        num_features=num_features,
        block_channels=block_channels,
        kernel_width=kernel_width,
    ),

    # (T, N, num_classes)
    nn.Linear(num_features, charset().num_classes),
    nn.LogSoftmax(dim=-1),
)

class ReshapeBands(nn.Module):
    def forward(self, x):
        # Assuming input is (T, N, 2 * in_features), reshape it back to (T, N, 2, in_features)
        T, N, features = x.shape
        return x.view(T, N, 2, features // 2)
"""

#model 3, n_head=16 
""" result: the lowest validation so far but out of memory error for the testing portion
val/loss            0.7237744331359863
val/CER            22.352680206298828
"""

#model 3, n_head=16, num_layers=1 
""" result: still out of memory error for the testing portion
val/loss            0.7776759266853333
val/CER            24.280017852783203
"""

#model 3, n_head=4, num_layers=4 
""" result: large overfitting
'test/loss': 0.8677679896354675,
'test/CER': 27.6204891204834,
"""

#model 4, combine model 1 and 3
"""
'test/loss': 0.8499680161476135,
'test/CER': 26.712772369384766,

self.model = nn.Sequential(
    # (T, N, bands=2, C=16, freq)
    SpectrogramNorm(channels=self.NUM_BANDS * ELECTRODE_CHANNELS),

    nn.Flatten(start_dim=2),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=in_features*2,  # Embedding size
            nhead=4,  # Number of attention heads
            dim_feedforward=in_features,  # Feedforward network size
            dropout=0.1
        ),
        num_layers=2  # Number of Transformer layers
    ),
    ReshapeBands(),

    # (T, N, bands=2, mlp_features[-1])
    MultiBandRotationInvariantMLP(
        in_features=in_features,
        mlp_features=mlp_features,
        num_bands=self.NUM_BANDS,
    ),
    # (T, N, num_features)
    nn.Flatten(start_dim=2),
    TDSConvEncoder(
        num_features=num_features,
        block_channels=block_channels,
        kernel_width=kernel_width,
    ),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=num_features,  # Embedding size
            nhead=4,  # Number of attention heads
            dim_feedforward=num_features * 4,  # Feedforward network size
            dropout=0.1
        ),
        num_layers=2  # Number of Transformer layers
    ),

    # (T, N, num_classes)
    nn.Linear(num_features, charset().num_classes),
    nn.LogSoftmax(dim=-1),
)
"""