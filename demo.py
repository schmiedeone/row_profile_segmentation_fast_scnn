import os
import argparse
import torch
import time

from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                    help='path to the input picture')
parser.add_argument('--input-pic-list', type=str,
                    default="",
                    help='path to the input picture list')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')
parser.add_argument('--weight-name', type=str,
                    default="",
                    help='custom weight name')
parser.add_argument('--data-root', type=str,
                    default="",
                    help='dataset root folder')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)

args = parser.parse_args()


def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_path_list = []
    if args.input_pic_list != "":
        file = open(args.input_pic_list, "r")
        image_path_list_dirty = file.readlines()
        image_root = os.path.join(args.data_root, "images")
        for img_path in image_path_list_dirty:
            img_path_no_trail = img_path.strip()
            image_path_list.append(os.path.join(image_root,img_path_no_trail))
    else :
        image_path_list.append(args.input_pic)
    
    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu, weight_name=args.weight_name).to(device)
    print('Finished loading model!')
    model.eval()

    for image_in in image_path_list:
        start = time.time()
        image = Image.open(image_in).convert('RGB')
        image = image.resize((960,540),0)
        image_transformed = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_transformed)
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, args.dataset).convert("RGBA")
        image = image.convert("RGBA")
        end = time.time()
        print("Time :",(end-start) * 10**3, "ms")

        image.putalpha(255)
        mask.putalpha(70)

        image_overlayed = Image.alpha_composite(image, mask)
        image_overlayed_name = os.path.splitext(os.path.split(image_in)[-1])[0] + '_result.png'
        image_overlayed.save(os.path.join(args.outdir, image_overlayed_name))




if __name__ == '__main__':
    demo()
