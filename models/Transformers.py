## Encoder architecture

#VIT

#Resnet + Encoder

import torch
import torch.nn as nn
import models.ResNet_1D as resnets


class PositionalEncoding(nn.Module):

    """Positional Encoding as defined for 2D."""

    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()

        #empty positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        #Add positional encodings - batch, seq_length, embedding_size
        return x + self.pe[:, :x.size(1), :].to(x.device)  
    


class ECG_Transformer_Encoder(nn.Module):
    def __init__(self, num_classes = 5, input_dim = 12, max_len=5000,  embed_dim = 256, num_heads=4, num_layers=4, enc_dropout = 0.1, enc_feature_extracture = 'none'):
        super(ECG_Transformer_Encoder, self).__init__()

        self.enc_feature_extracture = enc_feature_extracture
        #If we do not use a network to extract features, we need to adapt the input dimensions to the transformer dimensions
        if enc_feature_extracture == 'none':
            self.embedding = nn.Conv1d(in_channels = input_dim, out_channels=embed_dim, kernel_size = 3, stride = 1, padding = 1)  # Project ECG channels to embed_dim

        self.pos_encoding = PositionalEncoding(embed_dim, max_len) #Build positional encoding vector
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward = 4 * embed_dim, dropout = enc_dropout, batch_first=True)
        self.full_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling over time
        self.fc = nn.Linear(embed_dim, num_classes)  # Classification head

    def forward(self, x):

        #print(x.shape)
        
        #If we are usibg just the transformer
        if self.enc_feature_extracture == 'none':
            x = self.embedding(x)  # Shape: (batch, seq_len, embed_dim)

        #print(x.shape)
        #B,C,SeqLen - > B, SeqLen, C
        x = x.permute(0,2,1)
        x = self.pos_encoding(x)
        x = self.full_encoder(x)  # (batch_size, seq_len, embed_dim)
        #print(x.shape)
        x = x.permute(0, 2, 1)  # Change to (batch, embed_dim, seq_len) for pooling

        x = self.avg_pool(x).squeeze(-1)  # Apply global average pooling -> (batch, embed_dim)
        
        out = self.fc(x)  # Final classification
        return out
    


class ResTransformer(nn.Module):
    def __init__(self, in_channels, out_classes, signal_len, 
                resnet_type="resnet18", 
                num_heads=4, 
                num_layers=4, 
                enc_dropout = 0.1):
        
        super(ResTransformer, self).__init__()

        

        # Dynamically select the ResNet model
        if hasattr(resnets, resnet_type):
            #calls the chosen resnet and inits it base on our parameters
            resnet = getattr(resnets, resnet_type)(in_channels, out_classes, signal_len)
        else:
            raise ValueError(f"Invalid resnet_type: {resnet_type}. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101' or 'resnet152'.")

        # Remove the last classification layer and the average pooling, to preserve all the temporal information
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # Determine the embedding size based on ResNet architecture
        resnet_output_dim = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "resnet152": 2048
        }
        embed_dim = resnet_output_dim.get(resnet_type, 2048)  # Default to 2048 if not found

        #This already contains the classification layer
        #The resnet18 is responsible for extracting the features
        self.encoder = ECG_Transformer_Encoder(num_classes = out_classes, 
                                               input_dim = embed_dim, 
                                               max_len = signal_len,  
                                               embed_dim = embed_dim, 
                                               num_heads= num_heads, 
                                               num_layers=num_layers, 
                                               enc_dropout = enc_dropout, 
                                               enc_feature_extracture = 'resnet18')

    def forward(self, x):

        # Extract feature maps from ResNet
        x = self.resnet(x) 

        #Encoder + classification
        out = self.encoder(x)  
        
        return out