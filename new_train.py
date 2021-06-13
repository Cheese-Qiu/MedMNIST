import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from medmnist.new_model import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO

def main(flag, input_root, output_root, num_epoch, download):
    flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
    Class = flag_to_class[flag]
    info = INFO[flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    lr = 0.001
    batch_size = 128
    path = os.path.join(output_root, 'ckp_%s'%(flag))
    if not os.path.exists(path):
        os.makedirs(path)

    train_transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = Class(root=input_root, split='train', transform=train_transform, download=download)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Class(root=input_root,split='val',transform=val_transform,download=download)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=True)
    test_dataset = Class(root=input_root,split='test',transform=test_transform,download=download)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    val_auc_list = []
    for i in range(num_epoch):
        train(model, optimizer, criterion, train_loader, device, task)
        validate(model, val_loader, device, val_auc_list, task, path, i)
    list = np.array(val_auc_list)
    print('Best epoch: %s' % (list.argmax()))
    best_model_path = os.path.join(path, 'ckpt_%d_auc_%.5f.pth' % (list.argmax(), list[list.argmax()]))

    model.load_state_dict(torch.load(best_model_path)['net'])
    test(model, 'train', train_loader, device, flag, task, output_root=output_root)
    test(model, 'val', val_loader, device, flag, task, output_root=output_root)
    test(model, 'val', val_loader, device, flag, task, output_root=output_root)

def train(model, optimizer, criterion, train_loader, device, task):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()


def validate(model, val_loader, device, val_auc_list, task, dir_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = torch.nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = torch.nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
}

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, split, data_loader, device, flag, task, output_root=None):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = torch.nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = torch.nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='dermamnist', type=str)
    parser.add_argument('--input_root', default='./input', type=str)
    parser.add_argument('--output_root', default='./output_test', type=str)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--download', default=False, help='whether download the dataset or not', type=bool)

    args = parser.parse_args()
    data_name = args.data_name
    input_root = args.input_root
    output_root = args.output_root
    num_epoch = args.num_epoch
    download = args.download
    main(data_name,input_root,output_root,num_epoch=num_epoch,download=download)