import argparse
import os
from dataloaders.utils import *
from torchvision.utils import make_grid  # save_image
from dataloaders import make_data_loader
from utils.metrics import Evaluator
from tqdm import tqdm
from modeling.unet import Unet
from modeling.CENet import CE_Net
from modeling.DconnNet import DconnNet
from modeling.MEGANet import MEGANet
from modeling.CPFNet import CPFNet
from modeling.AC_MambaSeg import AC_MambaSeg
from modeling.H_vmunet import H_vmunet
from modeling.BWMamba import BWMamba
from modeling.HCMNet import HCMNet
from modeling.VMUNet import VMUNet
from modeling.VMUNetv2 import VMUNetV2
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):

    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, value_range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im = im.resize((256, 256))
    im.save(filename)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--out-path', type=str, default='./run/BUSI/',
                        help='mask image to save')
    parser.add_argument('--batch-size', type=int, default=4,
                        metavar='N', help='input batch size for test ')
    parser.add_argument('--ckpt', type=str, default='./run/BUSI/VMUNetV2/experiment_20240913_094256/checkpoint.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='BUSI',
                        choices=['BUSI', 'BUDB', 'STU'],
                        help='dataset name')
    parser.add_argument('--image-size', type=int, default=(512, 512),
                        help='BUSI:(512, 512), STU:(128, 128) ')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    
    device=torch.device("cuda:{}".format(args.gpu_ids[0]))

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader_list, val_loader_list, test_loader_list, nclass = make_data_loader(args, **kwargs)
    train_loader, val_loader, test_loader = train_loader_list[0], val_loader_list[0], test_loader_list[0]

    model = VMUNetV2().to(device)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'])

    out_path = os.path.join(args.out_path, 'Output', 'VMUNetV2/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    evaluator = Evaluator(2)
    model.eval()
    evaluator.reset()
    tbar = tqdm(val_loader, desc='\r')
    for i, sample in enumerate(tbar):
        # image, target = sample['image'], sample['label']
        image, target = sample[0]['image'], sample[0]['label']
        if args.cuda:
            image, target = image.to(device), target.to(device)
        with torch.no_grad():
            output = model(image)

        target_ = torch.unsqueeze(target, 1)
        target_n = target_.cpu().numpy()

        # 输出1通道
        output = output.data.cpu().numpy()
        output[output >= 0.5] = 1
        output[output < 0.5] = 0

        # 输出2通道
        # output = torch.argmax(output, dim=1).float().cpu().data.numpy()

        evaluator.add_batch(target_n, output)
        mask = output * 255

        # save imgs
        for i in range(args.batch_size):
            out_image = make_grid(image[i, :].clone().cpu().data, 3, normalize=True)
            out_GT = make_grid(decode_seg_map_sequence(target_n[i, :], dataset=args.dataset), 3, normalize=False,
                               value_range=(0, 255))
            out_pred_label_sum = make_grid(decode_seg_map_sequence(mask[i, :], dataset=args.dataset), 3,
                                           normalize=False, value_range=(0, 255))

            img_name = sample[1][i].split('.')[0]
            save_image(out_image, out_path + img_name + '.png')
            save_image(out_GT, out_path + img_name + '_segmentation' + '.png')
            save_image(out_pred_label_sum, out_path + img_name + '_pred' + '.png')

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    IoU = evaluator.Intersection_over_Union()
    Dice = evaluator.Dice()
    SE = evaluator.Sensitivity()
    SP = evaluator.Specificity()
    print('Test:')
    print("Acc:{}, IoU:{}, Dice:{}, SE:{}, SP:{}".format(Acc, IoU, Dice, SE, SP))


if __name__ == "__main__":
   main()
