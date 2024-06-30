import argparse, os
import torch
import math, random
from pretrain.losses import HFL_loss, grad_loss, image_loss
from model import Frepa_ViT, Frepa_SwinT
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from visualization.view_2D import plot_parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pretrain.utils import TrainSetLoader, do_datalist

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--vgg_loss", default=True, help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.9
                    , help='Learning Rate decay')


def train():
    opt = parser.parse_args()
    cuda = opt.cuda
    print("=> use gpu id: '{}'".format(opt.gpus))
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = Frepa_ViT(in_chans=3, mid_chans=128)
    # model.load_state_dict(torch.load("/data/Train_and_Test/MAE/pretrain_ViT.pth"))

    model = model.to('cuda')

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    # print(model)

    print("===> Setting Optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)

        data_set = TrainSetLoader(do_datalist("/media/chuy/rescaled_data"))
        data_loader = DataLoader(dataset=data_set, num_workers=opt.threads,
                                 batch_size=opt.batchSize, shuffle=True)
        trainor(data_loader, optimizer, model, epoch)
        scheduler.step()
        # seg_scheduler.step()


def trainor(data_loader, optimizer, model, epoch):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0
    loss_1 = image_loss().cuda()
    loss_2 = grad_loss().cuda()
    loss_3 = HFL_loss().cuda()

    for iteration, (raw, gt) in enumerate(data_loader):
        # print(raw.shape, mask.shape)
        pre = model(raw)
        loss = loss_1(pre, gt) + loss_2(pre, gt) + 0.2 * loss_3(pre, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss

        print("===> Epoch[{}]: loss: {:.5f}  avg_loss: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration % 100 + 1)))

        if (iteration + 1) % 100 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch, "/home/checkpoint")
            print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = os.path.join(path, "pretrain_SwinT.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    train()

