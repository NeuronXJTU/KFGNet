import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import scoreNet
import numpy as np
import argparse
from Dataset import VideoDataset
from utils import *


parser = argparse.ArgumentParser(description='RNN')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=12)
parser.add_argument('--use-gpus', default='1', type=str)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--data-dir', type=str, default='../data3')
parser.add_argument('--output-dir', type=str, default='OutTrain')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=1e-8)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--val-interval', type=int, default=2)
parser.add_argument('--save-model-interval', type=int, default=4)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpus
fold_k = args.fold
val_interval = args.val_interval
save_model_interval = args.save_model_interval
learning_rate = args.lr
weight_decay = args.weight_decay
batch_size = args.batch_size
epochs = args.epochs
num_workers = args.num_workers
data_dir = args.data_dir
output_dir = args.output_dir

# print('\n\n----------------------RNN-Fold-{}------------------------'.format(fold_k))
# print('Batch_Size: \t\t{}'.format(batch_size))
# print('Learning_Rate: \t\t{}'.format(learning_rate))
# print('Weigh_Decay: \t\t{}'.format(weight_decay))
# print('Epochs: \t\t{}'.format(epochs))
# print('Val_Interval: \t\t{}'.format(val_interval))
# print('Data_DIR: \t\t{}'.format(data_dir))
# print('Out_DIR: \t\t{}'.format(output_dir))
# print('Using   \t\t{} GPU(s).'.format(torch.cuda.device_count()))
# print('----------------------RNN-Fold-{}------------------------\n'.format(fold_k))

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device('cuda')


def train(epoch, model, train_loader, criterion, optimizer, scheduler, fwriter):
    epoch_loss = 0
    epoch_corrects = 0
    acc = 0
    acc_result = np.zeros((1, 33))
    model.train()

    print('\n--------------------Training-epoch-{:04}---------------------'.format(epoch+1))
    
    for batch_idx, (videos, labels) in enumerate(train_loader):

        videos, labels = videos.to(device), labels.to(device)

        output = model(videos)

        preds = output.data.max(1)[1]

        corrects = torch.sum( abs(labels.data.max(1)[1] - preds) < 3).item()
        acc += torch.sum(labels.data.max(1)[1] == preds).item()
        acc_list = [torch.sum( abs(labels.data.max(1)[1] - preds) <= i).item() for i in range(33)]
        acc_array = np.array(acc_list)
        acc_result += acc_array

        loss = criterion(output, labels)

        epoch_loss += loss

        epoch_corrects += corrects

        optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        # print(preds)
        # print(labels.data.max(1)[1])
        
        partial_epoch = epoch + (batch_idx + 1) / len(train_loader)
        print('Train Epoch: {:5.2f} \tLoss: {:.4f} \tAcc: {:.4f}'.format(partial_epoch, loss.item() / labels.size(0), corrects / labels.size(0)))

        fwriter.write('{:5.2f},{:.4f}\n'.format(partial_epoch, loss.item()))
        fwriter.flush()
    
    scheduler.step()

    epoch_acc = epoch_corrects / len(train_loader.dataset)
    acc /= len(train_loader.dataset)
    acc_result /= len(train_loader.dataset)
    print('Epoch Acc: {:.2f}'.format(epoch_acc))
    print('Train set: Acc: {:.4f}'.format(acc))
    print(acc_result)


def validate(epoch, model, val_loader, criterion, fwriter):
    loss = 0
    acc = 0
    acc_result = np.zeros((1, 33))

    model.eval()

    print('-------------------Validating-epoch-{:04}--------------------'.format(epoch+1))

    with torch.no_grad():
        for i, (videos, labels) in enumerate(val_loader):

            videos, labels = videos.to(device), labels.to(device)

            output = model(videos)

            preds = F.softmax(output, dim=1).data.max(1)[1]

            acc += torch.sum(labels.data.max(1)[1] == preds).item()
            acc_list = [torch.sum( abs(labels.data.max(1)[1] - preds) <= i).item() for i in range(33)]
            acc_array = np.array(acc_list)
            acc_result += acc_array

            loss += criterion(output, labels)

        acc /= len(val_loader.dataset)
        acc_result /= len(val_loader.dataset)
        loss /= len(val_loader.dataset)
        
        print('Val set: Average Loss: {:.4f}'.format(loss))
        print('Val set: Acc: {:.4f}'.format(acc))
        print(acc_result)


    fwriter.write('{},{:.4f},{:.4f}\n'.format(epoch + 1, loss, acc))
    fwriter.flush()

    return acc

def test(epoch, model, test_loader, criterion, fwriter):
    loss = 0
    acc = 0
    acc_result = np.zeros((1, 33))

    model.eval()

    print('-------------------Testing-epoch-{:04}--------------------'.format(epoch+1))

    with torch.no_grad():
        for i, (videos, labels) in enumerate(test_loader):

            videos, labels = videos.to(device), labels.to(device)

            output = model(videos)

            preds = F.softmax(output, dim=1).data.max(1)[1]

            acc += torch.sum(labels.data.max(1)[1] == preds).item()
            acc_list = [torch.sum( abs(labels.data.max(1)[1] - preds) <= i).item() for i in range(33)]
            acc_array = np.array(acc_list)
            acc_result += acc_array

            loss += criterion(output, labels)

        acc /= len(test_loader.dataset)
        acc_result /= len(test_loader.dataset)
        loss /= len(test_loader.dataset)
        
        print('Test set: Average Loss: {:.4f}'.format(loss))
        print('Test set: Acc: {:.4f}'.format(acc))
        print(acc_result)


    fwriter.write('{},{:.4f},{:.4f}\n'.format(epoch + 1, loss, acc))
    fwriter.flush()

    return acc


def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)


def collate_fn(data_and_label):
    v_data = [i[0] for i in data_and_label]
    label = [i[1] for i in data_and_label]
    v_data = pad_sequence(v_data, batch_first=True)
    label = pad_sequence(label, batch_first=True)

    return v_data, label


def main():
    model = scoreNet()

    model = model.to(device)

    check_dir(output_dir)
    
    fwriter_train = open(output_dir + '/train.csv', 'w')
    fwriter_val = open(output_dir + '/val.csv', 'w')
    fwriter_test = open(output_dir + '/test.csv', 'w')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [10], gamma = 0.1, last_epoch=-1)

    weights_init(model)

    train_dataset = VideoDataset(data_dir='../data_for_miccai/train')
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
    
    val_dataset = VideoDataset(data_dir='../data_for_miccai/val')
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)

    test_dataset = VideoDataset(data_dir='../data_for_miccai/test')
    test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)



    train_time, val_time, test_time = 0, 0, 0

    criterion = nn.MSELoss()

    max_acc = 0

    for epoch in range(epochs):
        start = tic()
        
        train(epoch, model, train_loader, criterion, optimizer, scheduler, fwriter_train)
        train_time += toc(start)

        if (epoch + 1) % val_interval == 0:
            start = tic()
            acc = validate(epoch, model, val_loader, criterion, fwriter_val)
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), output_dir + '/RNN-fold{}-best.ckpt'.format(fold_k))
            val_time += toc(start)
        
        if (epoch + 1) % val_interval == 0:
            start = tic()
            acc = test(epoch, model, test_loader, criterion, fwriter_test)
            test_time += toc(start)

        if (epoch + 1) % save_model_interval == 0:
            torch.save(model.state_dict(), output_dir + '/RNN-fold{}-{}.ckpt'.format(fold_k, epoch + 1))

        print('\nTrain time accumulated: {:.2f}s, Val time accumulated: {:.2f}s.\n'.format(train_time, val_time)) 


if __name__ == '__main__':
    main()
