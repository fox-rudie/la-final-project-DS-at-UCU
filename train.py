import os

import pytorch_lightning as pl
import torch
import torchmetrics
import yaml
from torchvision import transforms
from torchvision.datasets import CIFAR10

from core.model import ViT

pl.seed_everything(42)
torch.backends.cudnn.deterministic = True


class ModelLightning(pl.LightningModule):
    def __init__(self, config):
        super(ModelLightning, self).__init__()
        self.model = ViT(
            image_size=config['image_size'],
            num_classes=config['num_classes'],
            dim=config['model_dim'],
            depth=config['depth'],
            heads=config['heads'],
            feedforward_dim=config['feedforward_dim'],
            attention_type=config['attention_type']
        )
        self.train_acc = torchmetrics.Accuracy('multiclass', config['num_classes'])
        self.valid_acc = torchmetrics.Accuracy('multiclass', config['num_classes'])

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def compute_loss(y_hat, y):
        return torch.nn.functional.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=False)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('valid_loss', loss, prog_bar=False)
        self.valid_acc(y_hat, y)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    with open('config.yaml', 'r') as config_file:
        cfg = yaml.safe_load(config_file)

    # load data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                         [0.24703223, 0.24348513, 0.26158784])])

    train_dataset = CIFAR10(root=cfg['data_rootdir'], train=True, transform=transform, download=True)
    valid_dataset = CIFAR10(root=cfg['data_rootdir'], train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                               drop_last=True, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False,
                                               drop_last=False, pin_memory=False)

    # configure model
    model = ModelLightning(config=cfg['model'])

    # configure trainer
    train_cfg = cfg['training']

    os.makedirs(os.path.join(train_cfg['output_dir'], 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(train_cfg['output_dir'], 'logs'), exist_ok=True)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=train_cfg['output_dir'], name='logs')
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(train_cfg['output_dir'], 'ckpts'),
                                              filename='{epoch}-{train_loss:.3f}-{valid_loss:.3f}',
                                              save_on_train_epoch_end=True, save_top_k=3, monitor='valid_loss')
    early_stop = pl.callbacks.EarlyStopping(monitor='valid_loss', patience=5, check_on_train_epoch_end=True,
                                            verbose=False)
    trainer = pl.Trainer(accelerator='gpu', devices=1, precision=train_cfg['precision'],
                         max_epochs=train_cfg['max_epochs'], callbacks=[checkpoint, early_stop], logger=tb_logger)

    # perform training
    trainer.fit(model, train_loader, valid_loader)
