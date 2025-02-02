import torch
import denoising_diffusion_pytorch
from denoising_diffusion_pytorch1.gaussian_diffusion import GaussianDiffusion
from denoising_diffusion_pytorch1.dataloader import get_loader
from tqdm.auto import tqdm
from torchvision import transforms
import os
import json
import argparse
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--time_steps', default=250, type=int)
    parser.add_argument('--words_emb_dim', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--max_sent', default=4, type=int)
    parser.add_argument('--max_word', default=32, type=int)
    parser.add_argument('--pred_method', default='pred_x0', type=str)
    parser.add_argument('--dataset', default='iuxray', type=str)
    parser.add_argument('--root_dir', default='iu_xray', type=str)
    parser.add_argument('--tsv_path', default='data1234.json', type=str)
    parser.add_argument('--image_path', default='images', type=str)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output', default='checkpoints', type=str)
    parser.add_argument('--resume_checkpoints', default='checkpoints/df_epoch199_timesteps250.pth', type=str)

    args = parser.parse_args()
    return args
args = parse_args()

with open(args.tsv_path, 'r') as f:
    data = json.load(f)

train_data = data.get("train", [])
print(f"Number of samples in 'train': {len(train_data)}")

train_loader, vocab = get_loader(
        args.root_dir, args.tsv_path, args.image_path, transform,
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        dataset=args.dataset, max_sent=args.max_sent, max_word=args.max_word,
        drop_last=True,
        mode='train'
)

vocab_size = len(vocab)
print('vocab_size:',vocab_size)
print("train_loader.shape",train_loader)

gd = GaussianDiffusion(args.time_steps, args.max_sent, args.max_word, vocab_size, args.words_emb_dim, args.hidden_dim, args.pred_method,
                           loss_type='l2').cuda()


if __name__ == "__main__":

    if args.resume:
        save_name=args.resume_checkpoints
        state_dict = torch.load(save_name)
        gd.load_state_dict(state_dict['model'])

    #
    params_dicts = [
        {'params': gd.emb.parameters(), 'lr': 1e-5},
        {'params': gd.model.parameters(), 'lr': 1e-5},
        {'params': gd.anti_emb.parameters(), 'lr': 1e-5},
    ]

    optim_model = torch.optim.Adam(params=params_dicts)
    print('%.2fM parameter nubmer of total model' % (sum([param.nelement() for param in gd.parameters()]) / 1e6))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_model, T_max=args.epochs, eta_min=5e-8)

#train

    for epoch in range(args.epochs):
        gd.train()
        t_loss1 = 0
        t_loss2 = 0
        t_loss3 = 0
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty, please check your data or data loader configuration.")
        for i, (img, captions, prob) in enumerate(tqdm(train_loader)):
            # print('captions.shape',captions.shape)
            optim_model.zero_grad()
            img = img.cuda()
            # print('img.shape',img.shape)
            captions = captions.cuda()
            loss, loss1, loss2, loss3 = gd(captions, img)

            t_loss1 += loss1  # loss_dm
            t_loss2 += loss2  # loss_emb_i
            t_loss3 += loss3  # loss_ae
            loss.backward()

            optim_model.step()

        scheduler.step()

        print('epoch:%d loss: noisy(%.3f)  ae(%.3f) ' %
              (epoch, t_loss1 / i * 1e4,  t_loss3 / i * 1e6))
        torch.save({'model': gd.state_dict()}, f"{args.output}/df_epoch{epoch}_timesteps{args.time_steps}.pth")
        print("saved model")



