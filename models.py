import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_cnn(model, train_loader, val_loader, epochs, device, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    return history


class ConditionalEncoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4 + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4 + num_classes, latent_dim)

    def forward(self, x, y_onehot):
        h = self.conv(x).flatten(1)
        h = torch.cat([h, y_onehot], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, z, y_onehot):
        h = torch.cat([z, y_onehot], dim=1)
        h = self.fc(h).view(-1, 128, 4, 4)
        return self.deconv(h)


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = ConditionalEncoder(latent_dim, num_classes)
        self.decoder = ConditionalDecoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y_onehot):
        mu, logvar = self.encoder(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y_onehot), mu, logvar

    @torch.no_grad()
    def sample(self, labels, device):
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float().to(device)
        z = torch.randn(len(labels), self.decoder.fc.in_features - self.num_classes, device=device)
        return self.decoder(z, y_onehot)


def cvae_loss(recon, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss, kl_loss
