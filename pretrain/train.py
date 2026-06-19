import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from pretrain.losses import HFL_loss, grad_loss, image_loss
from model import Frepa_ViT
from pretrain.utils import TrainSetLoader, do_datalist

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    parser = argparse.ArgumentParser(description="Frepa pre-training")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA training")
    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--nEpochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--gamma", type=float, default=0.9)
    opt = parser.parse_args()

    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"=> using gpu id: '{opt.gpus}'")
    opt.seed = random.randint(1, 10000)
    print(f"Random Seed: {opt.seed}")
    torch.manual_seed(opt.seed)
    if use_cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = Frepa_ViT(in_chans=3, mid_chans=128).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    loss_1 = image_loss().to(device)
    loss_2 = grad_loss().to(device)
    loss_3 = HFL_loss().to(device)

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print(f"Epoch {epoch}")
        data_set = TrainSetLoader(do_datalist("/media/chuy/rescaled_data"))
        data_loader = DataLoader(
            dataset=data_set,
            num_workers=opt.threads,
            batch_size=opt.batchSize,
            shuffle=True,
        )
        trainor(data_loader, optimizer, model, epoch, loss_1, loss_2, loss_3, device)
        scheduler.step()


def trainor(data_loader, optimizer, model, epoch, loss_1, loss_2, loss_3, device):
    print(f"Epoch={epoch}, lr={optimizer.param_groups[0]['lr']}")
    model.train()
    loss_epoch = 0

    for iteration, (damaged, gt) in enumerate(data_loader):
        damaged = damaged.to(device)
        gt = gt.to(device)

        pre = model(damaged)
        loss = loss_1(pre, gt) + loss_2(pre, gt) + 0.2 * loss_3(pre, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()

        avg_loss = loss_epoch / (iteration % 100 + 1)
        print(f"===> Epoch[{epoch}]({iteration}): loss={loss.item():.5f}  avg={avg_loss:.5f}")

        if (iteration + 1) % 100 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch, "/home/checkpoint")


def save_checkpoint(model, epoch, path):
    model_out_path = os.path.join(path, f"pretrain_ViT_epoch{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    train()
