import torch
import torch.nn as nn
import torchvision.models as models

class ResNetMultiTaskFootball(nn.Module):
    def __init__(self, num_home_phases, num_away_phases, num_home_forms, num_away_forms):
        super().__init__()
        base_model = models.resnet18(pretrained=True)

        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # (B, 512, 1, 1)

        self.shared_fc = nn.Sequential(
            nn.Flatten(),         # (B, 512)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Output heads
        self.head_hp = nn.Linear(256, num_home_phases)
        self.head_ap = nn.Linear(256, num_away_phases)
        self.head_hf = nn.Linear(256, num_home_forms)
        self.head_af = nn.Linear(256, num_away_forms)

    def forward(self, x):
        x = self.backbone(x)     # (B, 512, 1, 1)
        x = self.shared_fc(x)    # (B, 256)

        return self.head_hp(x), self.head_ap(x), self.head_hf(x), self.head_af(x)
