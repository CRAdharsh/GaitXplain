import torch
import torch.nn as nn


class TemporalCNNBiLSTM(nn.Module):
    def __init__(self, input_features=32, hidden_size=64,
                 num_gait_classes=7, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.binary_head = nn.Linear(64, 2)
        self.gait_head = nn.Linear(64, num_gait_classes)

    def forward(self, x):
        # x: batch x time x features
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)

        attention_weights = torch.softmax(
            self.attention(x), dim=1
        )
        context = torch.sum(x * attention_weights, dim=1)

        representation = self.shared(context)

        binary_output = self.binary_head(representation)
        gait_output = self.gait_head(representation)

        return {
            "binary": binary_output,
            "gait": gait_output,
            "attention": attention_weights.squeeze(-1)
        }


if __name__ == "__main__":
    model = TemporalCNNBiLSTM()
    test_input = torch.randn(4, 32, 32)

    output = model(test_input)

    print("MODEL CREATED SUCCESSFULLY")
    print("Input shape:", tuple(test_input.shape))
    print("Binary output:", tuple(output["binary"].shape))
    print("Gait output:", tuple(output["gait"].shape))
    print("Attention:", tuple(output["attention"].shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

