import numpy as np
import time
import h5py
import argparse
import os.path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import NNModel
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

## input hyper-paras
parser = argparse.ArgumentParser(description = "nueral networks")
parser.add_argument("-mode", dest="mode", type=str, default='train', help="train or test")
parser.add_argument("-num_epoches", dest="num_epoches", type=int, default=40, help="num of epoches")

parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int, default=100, help="dim of hidden neurons")
parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int, default=100, help="dim of hidden neurons")
parser.add_argument("-learning_rate", dest ="learning_rate", type=float, default=0.001, help = "learning rate")
parser.add_argument("-decay", dest ="decay", type=float, default=0.5, help = "learning rate")
parser.add_argument("-batch_size", dest="batch_size", type=int, default=100, help="batch size")
parser.add_argument("-dropout", dest ="dropout", type=float, default=0.4, help = "dropout prob")
parser.add_argument("-rotation", dest="rotation", type=int, default=10, help="image rotation")

parser.add_argument("-activation", dest="activation", type=str, default='relu', help="activation function")
parser.add_argument("-MC", dest='MC', type=int, default=10, help="number of monte carlo")
parser.add_argument("-channel_out1", dest='channel_out1', type=int, default=64, help="number of channels")
parser.add_argument("-channel_out2", dest='channel_out2', type=int, default=64, help="number of channels")
parser.add_argument("-k_size", dest='k_size', type=int, default=4, help="size of filter")
parser.add_argument("-pooling_size", dest='pooling_size', type=int, default=2, help="size for max pooling")
parser.add_argument("-stride", dest='stride', type=int, default=1, help="stride for filter")
parser.add_argument("-max_stride", dest='max_stride', type=int, default=2, help="stride for max pooling")
parser.add_argument("-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint")

args = parser.parse_args()

def _load_data(DATA_PATH, batch_size):
    ## for training
    train_transform = transforms.Compose([transforms.RandomRotation(args.rotation),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root=DATA_PATH, download=True, train=True, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## for testing
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = torchvision.datasets.CIFAR10(root=DATA_PATH, download=True, train=False, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, test_loader

def adjust_learning_rate(learning_rate, optimizer, epoch, decay):
    """Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
    lr = learning_rate
    if (epoch > 5):
        lr = 0.001
    if (epoch >= 10):
        lr = 0.0001
    if (epoch > 20):
        lr = 0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def _compute_accuracy(y_pred, y_batch):
	return (y_pred==y_batch).sum().item()

def __save_checkpoint(ckp_path, model, epoches, optimizer):
    checkpoint = {'epoch': epoches,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, ckp_path)

def _load_checkpoint(ckp_path, model, optimizer):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoches = checkpoint['epoch']
    return epoches 

def _test_model(model, test_loader, device):
    pred_vec = []
    ground_truth = []
    model.eval()
    with torch.no_grad():
        accuracy = 0
        for batch_id, (x_batch,y_labels) in enumerate(test_loader):
            x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
            output_y = model(x_batch)
            _, y_pred = torch.max(output_y.data, 1)
            accuracy += _compute_accuracy(y_pred, y_labels)
            pred_vec.append(y_pred)
            ground_truth.append(y_labels)
        pred_vec = torch.cat(pred_vec).cpu()
        ground_truth = torch.cat(ground_truth).cpu()
        print("Recall: "+ str(recall_score(ground_truth, pred_vec, average='macro')))
        print("Precision: "+ str(precision_score(ground_truth, pred_vec, average='macro')))
        print("F1: "+ str(f1_score(ground_truth, pred_vec, average='macro')))  
        print("Test Accuracy: "+str(accuracy/(len(test_loader)*args.batch_size)))

def main():
    use_cuda = torch.cuda.is_available() ## if have gpu or cpu 
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.cuda.manual_seed(72)

    ## initialize hyper-parameters
    num_epoches = args.num_epoches
    decay = args.decay
    learning_rate = args.learning_rate

    ## Load data
    DATA_PATH = "./data/"
    train_loader, test_loader=_load_data(DATA_PATH, args.batch_size)

    ## Model initialization 
    model = NNModel()
    model.to(device)

    ## loss function
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    epochs = 0
    ## load checkpoint
    #epochs = _load_checkpoint("./checkpoint/checkpoint.pt", model, optimizer)

    ## Tensorboard
    writer = SummaryWriter()

    ## Create checkpoint path
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    ## Model Training
    if args.mode == 'train':
        model = model.train()
        for epoch in range(epochs, num_epoches): 
            ## learning rate
            adjust_learning_rate(learning_rate, optimizer, epoch, decay)
            accuracy = 0
            running_loss = 0.0
            for batch_id, (x_batch,y_labels) in enumerate(train_loader):
                x_batch,y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)

                ## feed input data x into model
                output_y = model(x_batch)

                ## loss_function
                loss = loss_function(output_y, y_labels)

                ## back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ## get accuracy
                _, y_pred = torch.max(output_y.data, 1)
                accuracy += _compute_accuracy(y_pred, y_labels)
                running_loss += loss.item()

            writer.add_scalar('Loss/train', running_loss/(len(train_loader)*args.batch_size), epoch+1)
            print("Epoch "+str(epoch+1)+" Training Accuracy: "+str(accuracy/(len(train_loader)*args.batch_size))+" Avg Loss: "+str(running_loss/(len(train_loader)*args.batch_size)))

            ## save checkpoint
            __save_checkpoint("./checkpoint/checkpoint.pt", model, epoch+1, optimizer)
    
    _test_model(model, test_loader, device)

    writer.close()

if __name__ == '__main__':
	time_start = time.time()
	main()
	time_end = time.time()
	print("running time: ", (time_end - time_start)/60.0, "mins")