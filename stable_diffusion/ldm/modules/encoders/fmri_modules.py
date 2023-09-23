import torch.nn as nn


class FMRI2VisualEmbedder(nn.Module):
    def __init__(self, input_size = 567, output_size = 4096, hidden_size = 256, dropout_prob = 0.5):
        super().__init__()
        self.fmri2vae = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # First fully connected layer
            nn.ReLU(),                           # ReLU activation function
            nn.Dropout(dropout_prob),            # Dropout layer
            nn.Linear(hidden_size, hidden_size),  # First fully connected layer
            nn.ReLU(),                           # ReLU activation function
            nn.Dropout(dropout_prob),            # Dropout layer
            nn.Linear(hidden_size, output_size)  # Second fully connected layer
        )
    
    def forward(self, fmri):
        emb_vae = self.fmri2vae(fmri)
        emb_vae = emb_vae.reshape(fmri.shape[0],4,32,32) # hard-coded to SD-v-1.5
        return emb_vae
