import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class BaseModel(nn.Module):
    def summary(self, input_size=None):
        return torchinfo.summary(
            self,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "params_percent"],
        )


class Model1(BaseModel):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Model2(BaseModel):
    def __init__(self):
        super(Model2, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.ReLU()
        )

        self.oblock = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 10, 1),
            nn.Conv2d(10, 10, 7, 7),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x


class Model3(BaseModel):
    def __init__(self):
        super(Model3, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock3 = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.Conv2d(16, 32, 1),
            nn.Conv2d(32, 10, 1),
            nn.Conv2d(10, 10, 7, 7),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x


class Model4(BaseModel):
    def __init__(self):
        super(Model4, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(
            nn.BatchNorm2d(8), nn.Conv2d(8, 4, 1), nn.MaxPool2d(2, 2)
        )

        self.cblock2 = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(
            nn.BatchNorm2d(12), nn.Conv2d(12, 8, 1), nn.MaxPool2d(2, 2)
        )

        self.cblock3 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 1),
            nn.Conv2d(10, 10, 7, 7),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x


class Model5(BaseModel):
    def __init__(self):
        super(Model5, self).__init__()
        DROP = 0.05
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(
            nn.BatchNorm2d(8), nn.Conv2d(8, 4, 1), nn.MaxPool2d(2, 2)
        )

        self.cblock2 = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(
            nn.BatchNorm2d(12), nn.Conv2d(12, 8, 1), nn.MaxPool2d(2, 2)
        )

        self.cblock3 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 1),
            nn.Conv2d(10, 10, 7, 7),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x


class Model6(BaseModel):
    def __init__(self):
        super(Model6, self).__init__()
        DROP = 0.05
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(
            nn.BatchNorm2d(8), nn.Conv2d(8, 4, 1), nn.MaxPool2d(2, 2)
        )

        self.cblock2 = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(
            nn.BatchNorm2d(12), nn.Conv2d(12, 8, 1), nn.MaxPool2d(2, 2)
        )

        self.cblock3 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 12, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x


class Model7(BaseModel):
    def __init__(self):
        super(Model7, self).__init__()
        DROP = 0.03
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 10, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(
            nn.BatchNorm2d(10), nn.Conv2d(10, 8, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        self.cblock2 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 10, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 14, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(
            nn.BatchNorm2d(14), nn.Conv2d(14, 10, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        self.cblock3 = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 14, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Conv2d(14, 22, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.BatchNorm2d(22),
            nn.Conv2d(22, 20, 1),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 10, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x
