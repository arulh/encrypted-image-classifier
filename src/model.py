import torchvision.models
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics

class NeuralNet(pl.LightningModule):

    def __init__(self):
        super(NeuralNet, self).__init__()
        # Load a pre-trained model (e.g., ResNet-18) as the backbone
        self.backbone = torchvision.models.resnet18(pretrained=True)
        # Freeze all parameters in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer with a new one for binary classification
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, 2)

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)


    def forward(self, x):
        # Forward pass through the backbone
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # Extract images and labels from the batch
        x, y = batch
        # Forward pass
        outputs = self.forward(x)
        # Compute the loss (binary cross-entropy)
        loss = F.cross_entropy(outputs, y)
        # Log training loss
        self.log('train_loss', loss)

        self.train_acc(outputs, y)
        self.log('train_acc', self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        # Extract images and labels from the batch
        x, y = batch
        # Forward pass
        outputs = self.forward(x)
        # Compute the loss (binary cross-entropy)
        loss = F.cross_entropy(outputs, y)
        # Log validation loss
        self.log('val_loss', loss)

        self.val_acc(outputs, y)
        self.log('val_acc', self.val_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Forward pass
        outputs = self.forward(x)
        # Compute the loss (binary cross-entropy)
        loss = F.cross_entropy(outputs, y)
        # Log validation loss
        self.log('test_loss', loss)

        self.test_acc(outputs, y)
        self.log('test_acc', self.test_acc)
    
    def predict_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer