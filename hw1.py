import argparse
import os
import torch

import torch.nn.functional as F

from torch import nn
from load_data import DataGenerator
from dnc import DNC
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import neptune.new as neptune


class MANN(nn.Module):

    def __init__(self, num_classes, samples_per_class, model_size=128, input_size=784):
        super(MANN, self).__init__()
        
        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
    
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_size = input_size
        self.layer1 = torch.nn.LSTM(num_classes + input_size, 
                                    model_size, 
                                    batch_first=True)
        self.layer2 = torch.nn.LSTM(model_size,
                                    num_classes,
                                    batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: tensor
                A tensor of shape [B, K+1, N, 784] of flattened images
            
            labels: tensor:
                A tensor of shape [B, K+1, N, N] of ground truth labels
        Returns:
            
            out: tensor
            A tensor of shape [B, K+1, N, N] of class predictions
        """
        device = next(self.layer1.parameters()).device
        b, k, n, d = input_images.shape
        k -= 1
        
        train_images = input_images[:,:-1].reshape(b, n*k, d)
        train_labels = input_labels[:,:-1].reshape(b, n*k, n)
        
        train_inps = torch.cat([train_images, train_labels], dim=2)
        
        test_images = input_images[:,-1:].reshape(b, n, d)
        # test_labels = input_labels[:,-1:].reshape(b, n, n)
        
        test_labels = torch.zeros((b,n,n)).to(device)
        test_inps = torch.cat([test_images, test_labels], dim=2)
        
        all_inps = torch.cat([train_inps, test_inps], dim=1)
        
        o1, _ = self.layer1(all_inps)
        out = self.layer2(o1)[0].reshape(b, k+1, n, n) # (batch_size, n, k+1, d)
        
        return out
        


    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: tensor
                A tensor of shape [B, K+1, N, N] of network outputs
            
            labels: tensor
                A tensor of shape [B, K+1, N, N] of class labels
                
        Returns:
            scalar loss
        """
        b,k,n,_ = preds.shape
        k -= 1
        
        labels = labels[:,-1].reshape(-1,n) # (B*N,N)
        preds = preds[:,-1].reshape(-1,n)
        
        labels = torch.argmax(labels, dim=1).long() # (B*N,)
        
        return self.loss(preds, labels)     



def train_step(images, labels, model, optim):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def main(config):
    device = torch.device("cuda")
    tensorboard_on = False
    neptune_on = True

    if tensorboard_on:
        writer = SummaryWriter(config.logdir)
    
    if neptune_on:
        api_token = ''
        run = neptune.init(project='kareem-elsawah/meta-learning', api_token=api_token, source_files=["*.py"])

    # Download Omniglot Dataset
    if not os.path.isdir('./omniglot_resized'):
        gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                            dest_path='./omniglot_resized.zip',
                                            unzip=True)
    assert os.path.isdir('./omniglot_resized')

    # Create Data Generator
    data_generator = DataGenerator(config.num_classes, 
                                   config.num_samples, 
                                   device=device)

    # Create model and optimizer
    model = MANN(config.num_classes, config.num_samples, 
                 model_size=config.model_size)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    for step in tqdm(range(config.training_steps), position=0, leave=False):
        images, labels = data_generator.sample_batch('train', config.meta_batch_size)
        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch('test', 
                                                         config.meta_batch_size)
            pred, test_loss = model_eval(images, labels, model)
            pred = torch.reshape(pred, [-1, 
                                        config.num_samples + 1, 
                                        config.num_classes, 
                                        config.num_classes])
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            labels = torch.argmax(labels[:, -1, :, :], axis=2)
            if neptune_on:
                run['train_loss'].log(train_loss.cpu().numpy().reshape(-1)[0], step=step)
                run['test_loss'].log(test_loss.cpu().numpy().reshape(-1)[0], step=step)
                run['meta_test_acc'].log(pred.eq(labels).double().mean().item(), step=step)

            if tensorboard_on:
                writer.add_scalar('Train Loss', train_loss.cpu().numpy(), step)
                writer.add_scalar('Test Loss', test_loss.cpu().numpy(), step)
                writer.add_scalar('Meta-Test Accuracy', 
                                pred.eq(labels).double().mean().item(),
                                step)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--logdir', type=str, 
                        default='run/log')
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--model_size', type=int, default=128)
    main(parser.parse_args())