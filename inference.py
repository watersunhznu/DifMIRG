import torch
import denoising_diffusion_pytorch
from denoising_diffusion_pytorch1.gaussian_diffusion import GaussianDiffusion
from denoising_diffusion_pytorch1.dataloader import get_loader
from tqdm.auto import tqdm
from torchvision import transforms
import os
from train_dataloader import loader
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
    parser.add_argument('--checkpoints', default='checkpoints/df_epoch170_timesteps250.pth', type=str)

    args = parser.parse_args()
    return args

args = parse_args()
_,vocab=loader(args)

val_loader, vocab = get_loader(
            args.root_dir, args.tsv_path, args.image_path, transform,
            batch_size=args.batch_size, shuffle=True, num_workers=1,
            dataset=args.dataset, max_sent=args.max_sent, max_word=args.max_word,
            drop_last=True,
            mode='val',vocab=vocab
        )
vocab_size = len(vocab)
gd = GaussianDiffusion(args.time_steps, args.max_sent, args.max_word, vocab_size, args.words_emb_dim, args.hidden_dim, args.pred_method,
                            loss_type='l2').cuda()


if __name__ == '__main__':
    #inference
    state_dict = torch.load(args.checkpoints)
    gd.load_state_dict(state_dict['model'])
    gd.eval()
    inference_samples = next(iter(val_loader))
    img, captions, prob = inference_samples
    print(len(captions))
    img = img.cuda()
    topic = gd.encoder(img)  # image_feature
    pred = gd.sample(topic, topic.size(0))  # produce_xt
    # print(pred)

    hypotheses = list()


    print(f"pred.shape: {pred.shape}")  # torch.Size([16, 128])
    pred = pred.view(args.batch_size, 4, 32)
    for j in range(captions.shape[0]):
        pred_caption = []
        target_caption = []
        for k in range(captions.shape[1]):
            words_x = pred[j, k, :].tolist()

            pred_caption.append(" ".join([vocab.idx2word[w] for w in words_x if
                                              w not in {vocab.word2idx['<pad>']}]) + ".")

            hypotheses.append(pred_caption)


    print(hypotheses[0])


