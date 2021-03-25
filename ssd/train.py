import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from model import SSD300, MultiBoxLoss
from dataset import BCCDDataset
from utils import save_checkpoint, AverageMeter, clip_gradient


class TrainParams:
    workers = 2
    batch_size = 8
    data_folder = './'
    keep_difficult = True
    n_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    batch_size = 8
    print_freq = 3
    lr = 0.001
    momentum = 0.9
    weight_decay = 5e-4
    grad_clip = None


def main_train():
    cudnn.benchmark = True

    if TrainParams.checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=TrainParams.n_classes)
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(
            params=[{'params': biases, 'lr': 2 * TrainParams.lr}, {'params': not_biases}],
            lr=TrainParams.lr, momentum=TrainParams.momentum, weight_decay=TrainParams.weight_decay)
    else:
        checkpoint = torch.load(TrainParams.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(TrainParams.device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(TrainParams.device)

    train_dataset = BCCDDataset(TrainParams.data_folder, split='train', keep_difficult=TrainParams.keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TrainParams.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=TrainParams.workers,
                                               pin_memory=True)

    epochs = start_epoch + 1

    for epoch in range(start_epoch, epochs):
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=TrainParams.device
        )

        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch, device, grad_clip=None, print_freq=3):
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels
