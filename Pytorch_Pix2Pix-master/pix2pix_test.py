import argparse
import os
import random
import torch
import torchvision
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from PIL import Image
from network import Generator

parser = argparse.ArgumentParser(description='Easy Implementation of Pix2Pix')

"""
Task                RawSize     Batch Size      Epochs      EpochsInPaper       RandomCrop&Mirroring    #ofSamples      #ofSamples(Validation)
Map-Aerial          600*600     1               100         200                 Yes                     1096            1098
Edges-Bag           256*256     4               15          15                  No                      138567          200
Semantic-Photo      256*256     1               100         200                 Yes                     2975            500
"""

# Task
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders train, val, etc)')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')

# Options
parser.add_argument('--no_resize_or_crop', action='store_true', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=1, help='test Batch size')

# misc
parser.add_argument('--model_path', type=str, default='./models')
parser.add_argument('--sample_path', type=str, default='./test_results')

##### Helper Functions for Data Loading & Pre-processing
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

##### Helper Functions for Data Loading & Pre-processing
class ImageFolder(data.Dataset):
    def __init__(self, opt):
        # os.listdir Function gives all lists of directory
        self.root = opt.dataroot
        self.no_resize_or_crop = opt.no_resize_or_crop
        self.no_flip = opt.no_flip
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])
        self.dir_AB = os.path.join(opt.dataroot, 'train')  # ./maps/train
        self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB)))

    def __getitem__(self, index):
        AB_path = self.image_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        if(not self.no_resize_or_crop):
            AB = AB.resize((286 * 2, 286), Image.BICUBIC)
            AB = self.transform(AB)

            w = int(AB.size(2) / 2)
            h = AB.size(1)
            w_offset = random.randint(0, max(0, w - 256 - 1))
            h_offset = random.randint(0, max(0, h - 256 - 1))

            A = AB[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
            B = AB[:, h_offset:h_offset + 256, w + w_offset:w + w_offset + 256]
        else:
            AB = self.transform(AB)
            w_total = AB.size(2)
            w = int(w_total / 2)

            A = AB[:, :256, :256]
            B = AB[:, :256, w:w + 256]

        if (not self.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.image_paths)

##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

######################### Main Function
def main():
    # Pre-settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    dataset = ImageFolder(args)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batchSize,
                                  shuffle=True,
                                  num_workers=2)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (args.num_epochs))

    # Load pre-trained model
    generator = Generator(args.batchSize)
    generator.load_state_dict(torch.load(g_path))
    generator.eval()

    if torch.cuda.is_available():
        generator = generator.cuda()

    total_step = len(data_loader) # For Print Log
    for i, sample in enumerate(data_loader):
        AtoB = args.which_direction == 'AtoB'
        input_A = sample['A' if AtoB else 'B']
        input_B = sample['B' if AtoB else 'A']

        real_A = to_variable(input_A)
        fake_B = generator(real_A)
        real_B = to_variable(input_B)

        # print the log info
        print('Validation[%d/%d]' % (i + 1, total_step))
        # save the sampled images
        res = torch.cat((torch.cat((real_A, fake_B), dim=3), real_B), dim=3)
        torchvision.utils.save_image(denorm(res.data), os.path.join(args.sample_path, 'Generated-%d.png' % (i + 1)))

if __name__ == "__main__":
    main()