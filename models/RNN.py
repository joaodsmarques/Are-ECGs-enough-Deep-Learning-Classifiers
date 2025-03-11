#Code adapted from https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html

import torch
import torch.nn as nn
import models.ResNet_1D as resnets

class ECG_RNN(nn.Module):
    def __init__(self, n_channels, hidden_size, num_layers, num_classes = 5, rnn_type = "lstm", dropout = 0.3, bidirectional = False):
        super(ECG_RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.n_directions = 2 if bidirectional else 1

        #Choose the model based on the user selection
        if rnn_type == "lstm":
            self.lstm = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            print('lstm')

        elif rnn_type == "gru":
            self.gru = nn.GRU(n_channels, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            print('gru')

        #Default rnn
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(n_channels, hidden_size, num_layers, batch_first=True, dropout=dropout)
            print('rnn')

        else:
             raise Exception("Invalid type of rnn.")

        self.fc = nn.Linear(hidden_size * self.n_directions, num_classes)

    def forward(self, x):

        #Get initial h and c for LSTM
        h0 = torch.zeros(self.num_layers * self.n_directions, x.size(0), self.hidden_size).to(x.device)
        
        #Make it like batch, sequance length, channels
        x = torch.permute(x, (0,2,1))
      
        #LSTM 
        if self.rnn_type == "lstm":
            c0 = torch.zeros(self.num_layers * self.n_directions, x.size(0), self.hidden_size).to(x.device)
            
            out, _ = self.lstm(x, (h0, c0))

        # GRU model
        elif self.rnn_type == "gru":
            out, _ = self.gru(x, h0)

        # Default RNN
        elif self.rnn_type == "rnn":
            out, _ = self.rnn(x, h0)

        else:
            raise Exception("Invalid type of rnn.")

        #Get just the last iteration
        out = out[:, -1, :]

        #Classify
        out = self.fc(out)  

        return out


class CRNN(nn.Module):
    def __init__(self, in_channels, out_classes, signal_len, 
                 resnet_type="resnet18", 
                 hidden_size = 256, 
                 num_layers = 1, 
                 rnn_type = "lstm", 
                 dropout = 0.3, 
                 bidirectional = False):
        
        super(CRNN, self).__init__()

        self.in_channels = in_channels
        self.signal_len = signal_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.n_directions = 2 if bidirectional else 1

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

        # Types of RNNs implemented
        #Choose the model based on the user selection
        if rnn_type == "lstm":
            self.lstm = nn.LSTM(embed_dim , hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            print('lstm')

        elif rnn_type == "gru":
            self.gru = nn.GRU(embed_dim , hidden_size, num_layers, batch_first=True, dropout=dropout)
            print('gru')

        #Default rnn
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
            print('rnn')

        else:
             raise Exception("Invalid type of rnn.")

        # Final classifier
        self.fc = nn.Linear(hidden_size * self.n_directions, out_classes)

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
        x = self.resnet(x)  
        
        #Get initial h and c for LSTM
        h0 = torch.zeros(self.num_layers * self.n_directions, x.size(0), self.hidden_size).to(x.device)
        
        #Make it like batch, sequance length, channels
        x = torch.permute(x, (0,2,1))
      
        #LSTM 
        if self.rnn_type == "lstm":
            c0 = torch.zeros(self.num_layers * self.n_directions, x.size(0), self.hidden_size).to(x.device)
            
            out, _ = self.lstm(x, (h0, c0))

        # GRU model
        elif self.rnn_type == "gru":
            out, _ = self.gru(x, h0)

        # Default RNN
        elif self.rnn_type == "rnn":
            out, _ = self.rnn(x, h0)

        else:
            raise Exception("Invalid type of rnn.")

        #Get just the last iteration
        out = out[:, -1, :]

        #Classify
        out = self.fc(out)  

        return out