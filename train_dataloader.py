import torch
from denoising_diffusion_pytorch1.dataloader import get_loader
from tqdm.auto import tqdm
from torchvision import transforms
import os
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

def loader(args):

    train_loader, vocab = get_loader(
            args.root_dir, args.tsv_path, args.image_path, transform,
            batch_size=args.batch_size, shuffle=True, num_workers=1,
            dataset=args.dataset, max_sent=args.max_sent, max_word=args.max_word,
            drop_last=True,
            mode='train'
    )
    return train_loader,vocab

