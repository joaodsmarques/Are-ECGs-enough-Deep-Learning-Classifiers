import torch
import torch.nn as nn
import models.ResNet_1D as resnets


class AttResNet(nn.Module):
    def __init__(self, in_channels, out_classes, signal_len, resnet_type="resnet18", num_heads=8, reduction = "none"):
        super(AttResNet, self).__init__()

        self.in_channels = in_channels
        self.signal_len = signal_len
        self.reduction = reduction

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

        # Multihead Attention Layer
        #embed_dim = n_channels
        self.attention = nn.MultiheadAttention(embed_dim = embed_dim, num_heads=num_heads, batch_first=True)

        flattened_size = self.calculate_flattened_size()


        # Final classifier
        if self.reduction == "none":
            self.classifier = nn.Linear(flattened_size, out_classes)

        else:
            self.classifier = nn.Linear(embed_dim, out_classes)

    def calculate_flattened_size(self):
        """Helper function to compute the flattened feature size after self.features."""
        # It makes a dummy 
        with torch.no_grad():
            # Create a dummy input with the given signal length and channels
            dummy_input = torch.zeros(1, self.in_channels, self.signal_len)
            features_output = self.resnet(dummy_input)

            return features_output.view(1, -1).size(1)

    def forward(self, x):
        # Extract feature maps from ResNet
        features = self.resnet(x)  # Shape: (batch_size, embed_dim, 1, 1)
        
        
        #Get in the format batch, seql length, embedded dim
        features = features.permute(0, 2, 1)
        
        # Apply multihead self-attention (Q, K, V are the same)
        attn_output, _ = self.attention(features, features, features)  # Shape: (batch_size, 1, embed_dim)
        

        #flattens the vector into batch, data
        # Instead of this, we can apply mean if we have a small dataset an too much data
        if self.reduction == "none":
            attn_output = attn_output.flatten(start_dim = 1)
        #mean reduction
        else:
            attn_output = attn_output.mean(dim=1)
        
        # Flatten and classify
        output = self.classifier(attn_output)  # Shape: (batch_size, num_classes)

        return output
    