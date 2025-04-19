from torch import nn
from torch.nn import functional as F



class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, kernel_size=[], no_kernels=[], fc1_size=512, conv_activation='ReLU', use_batch_norm=True,dropout=0.5):
        
        super(SimpleCNN, self).__init__()
        self.conv_activation = conv_activation
        self.fc1_size = fc1_size
        self.kernel_size = kernel_size
        self.no_kernels = no_kernels
        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm  
        self.dropout=dropout  # Dropout probability
        # Flag to enable/disable Batch Normalization
        # Define convolutional layers with optional Batch Normalization
        self.conv1 = nn.Conv2d(input_channels, no_kernels[0], kernel_size=kernel_size[0], stride=1, padding=kernel_size[0] // 2)
        self.bn1 = nn.BatchNorm2d(no_kernels[0]) if use_batch_norm else None  # Optional Batch Norm
        self.conv2 = nn.Conv2d(no_kernels[0], no_kernels[1], kernel_size=kernel_size[1], stride=1, padding=kernel_size[1] // 2)
        self.bn2 = nn.BatchNorm2d(no_kernels[1]) if use_batch_norm else None
        self.conv3 = nn.Conv2d(no_kernels[1], no_kernels[2], kernel_size=kernel_size[2], stride=1, padding=kernel_size[2] // 2)
        self.bn3 = nn.BatchNorm2d(no_kernels[2]) if use_batch_norm else None
        self.conv4 = nn.Conv2d(no_kernels[2], no_kernels[3], kernel_size=kernel_size[3], stride=1, padding=kernel_size[3] // 2)
        self.bn4 = nn.BatchNorm2d(no_kernels[3]) if use_batch_norm else None
        self.conv5 = nn.Conv2d(no_kernels[3], no_kernels[4], kernel_size=kernel_size[4], stride=1, padding=kernel_size[4] // 2)
        self.bn5 = nn.BatchNorm2d(no_kernels[4]) if use_batch_norm else None

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(no_kernels[4] * 7 * 7, fc1_size)
        self.fc2 = nn.Linear(fc1_size, num_classes)
        
        self.dropout_layer=nn.Dropout(p=self.dropout) if self.dropout>0 else None# Optional Dropout

    def forward(self, x):
        x = self.pool(self.apply_batch_norm(self.conv1(x), self.bn1))  # Apply Batch Norm if enabled
        x = self.pool(self.apply_batch_norm(self.conv2(x), self.bn2))
        x = self.pool(self.apply_batch_norm(self.conv3(x), self.bn3))
        x = self.pool(self.apply_batch_norm(self.conv4(x), self.bn4))
        x = self.pool(self.apply_batch_norm(self.conv5(x), self.bn5))
        x = x.view(-1, self.no_kernels[4] * 7 * 7)  # Flatten the tensor
        x = self.activation(self.fc1(x))
        if self.dropout_layer is not None:
            x=self.dropout_layer(x)  # Apply Dropout
        x = self.fc2(x)
        return x

    def apply_batch_norm(self, x, bn_layer):
        if self.use_batch_norm and bn_layer is not None:
            return self.activation(bn_layer(x))  # Apply Batch Norm and activation
        else:
            return self.activation(x)  # Skip Batch Norm and apply activation

    def activation(self, x):
        if self.conv_activation == 'ReLU':
            return F.relu(x)
        elif self.conv_activation == 'Sigmoid':
            return F.sigmoid(x)
        elif self.conv_activation == 'Tanh':
            return F.tanh(x)
        elif self.conv_activation == 'GELU':
            return F.gelu(x)
        elif self.conv_activation == 'SiLU':
            return F.silu(x)
        elif self.conv_activation == 'Mish':
            return F.mish(x)
        elif self.conv_activation == 'ELU':
            return F.elu(x)
        elif self.conv_activation == 'SELU':
            return F.selu(x)
        elif self.conv_activation == 'LeakyReLU':
            return F.leaky_relu(x)
        else:
            raise ValueError("Invalid activation function specified")