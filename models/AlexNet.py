import torch
# Original code from https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html
# Pytorch adaptation to 1D


class AlexNet(torch.nn.Module):
    def __init__(self,in_channels, out_classes, signal_length, p_dropout):
        super(AlexNet, self).__init__()

        self.input_channels = in_channels
        self.signal_len = signal_length
        self.p_dropout = p_dropout

        self.features = torch.nn.Sequential(

            torch.nn.Conv1d(in_channels,64,kernel_size=11,stride=4,padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3,stride=2),

            torch.nn.Conv1d(64, 192, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=3, stride=2),

            torch.nn.Conv1d(192, 384, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(384),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(256),

            torch.nn.Conv1d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
           
            torch.nn.MaxPool1d(kernel_size=3, stride=2),
            torch.nn.AdaptiveAvgPool1d(6)
        )

        #gets the output size of the feature space
        flattened_size = self.calculate_flattened_size()

        self.classifier = torch.nn.Sequential(

           torch.nn.Dropout(self.p_dropout),
           torch.nn.Linear(flattened_size,4096),
           torch.nn.ReLU(inplace=True),

           torch.nn.Dropout(self.p_dropout),
           torch.nn.Linear(4096, 4096),
           torch.nn.ReLU(inplace=True),
           torch.nn.Linear(4096, out_classes)
       )

    def calculate_flattened_size(self):
        """Helper function to compute the flattened feature size after self.features."""
        with torch.no_grad():
            # Create a dummy input with the given signal length and channels
            dummy_input = torch.zeros(1, self.input_channels, self.signal_len)
            features_output = self.features(dummy_input)
            return features_output.view(1, -1).size(1)

    def forward(self,x):

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
