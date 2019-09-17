import torch
from torch.autograd import Variable
import tools.utils as utils
import tools.dataset as dataset
from PIL import Image
from collections import OrderedDict
import numpy as np
import cv2
import glob
import os
import re
import click
from models.moran import MORAN

model_path = './demo.pth'
img_dir = '/Users/thanhtu/Projects/Python/CRAFT-pytorch/cropped/'
alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:8:t:u:v:w:x:y:z:$'

target_height = 32
target_width = 100

cuda_flag = False
if torch.cuda.is_available():
    cuda_flag = True
    MORAN = MORAN(1, len(alphabet.split(':')), 256, target_height,
                  target_width, BidirDecoder=True, CUDA=cuda_flag)
    MORAN = MORAN.cuda()
else:
    MORAN = MORAN(1, len(alphabet.split(':')), 256, target_height, target_width,
                  BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=cuda_flag)

print('loading pretrained model from %s' % model_path)
if cuda_flag:
    state_dict = torch.load(model_path)
else:
    state_dict = torch.load(model_path, map_location='cpu')
MORAN_state_dict_rename = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # remove `module.`
    MORAN_state_dict_rename[name] = v
MORAN.load_state_dict(MORAN_state_dict_rename)

for p in MORAN.parameters():
    p.requires_grad = False
MORAN.eval()

converter = utils.strLabelConverterForAttention(alphabet, ':')

image_paths = []

image_paths.extend(glob.glob(os.path.join(img_dir, "*.jpg")))
found = 0
not_found = 0
count = 0
ground_truth = ''


for img_path in image_paths:
    count += 1
    # if count > 23:
    #     break
    basename = os.path.basename(img_path)
    # match_obj = re.search('\d*_([\d-]+)_', basename)
    # ground_truth = match_obj.group(1)
    image = Image.open(img_path).convert('L')
    width, height = image.size
    new_height = round(target_width*(height/width))
    transformer = dataset.resizeNormalize((target_width, new_height))
    image = transformer(image)

    if cuda_flag:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    text = torch.LongTensor(1 * 5)
    length = torch.IntTensor(1)
    text = Variable(text)
    length = Variable(length)

    max_iter = 20
    t, l = converter.encode('0'*max_iter)
    utils.loadData(text, t)
    utils.loadData(length, l)
    output = MORAN(image, length, text, text, test=True, debug=True)

    preds, preds_reverse = output[0]
    demo = output[1]

    _, preds = preds.max(1)
    _, preds_reverse = preds_reverse.max(1)

    sim_preds = converter.decode(preds.data, length.data)
    sim_preds = sim_preds.strip().split('$')[0]
    # sim_preds_reverse = converter.decode(preds_reverse.data, length.data)
    # sim_preds_reverse = sim_preds_reverse.strip().split('$')[0]

    # print('\nResult:\n' + 'Left to Right: ' + sim_preds +
    #       '\nRight to Left: ' + sim_preds_reverse + '\n\n')
    len_sim = len(sim_preds)
    if len_sim == 7 or len_sim == 8:
        sim_preds = sim_preds[0:2] + '-' + \
            sim_preds[2:len_sim-4] + '-' + sim_preds[-4:]

    if ground_truth == sim_preds:
        found += 1
        click.secho("{}-{}-{}".format(
            found, sim_preds, basename), fg='green', bold=True)

        open_cv_image = cv2.imread(img_path)
        # Convert RGB to BGR
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        ch, cw, _ = open_cv_image.shape
        h, w, _ = demo.shape
        new_width = round(
            h * (cw/ch))

        open_cv_image = cv2.resize(open_cv_image, (new_width, h))
        demo = demo[0:h, round(w/3):w]
        demo = cv2.resize(demo, (w, h))
        demo = cv2.hconcat([open_cv_image, demo])
        cv2.imshow(ground_truth, demo)

    else:
        not_found += 1
        click.secho("{}-{}-{}".format(
            not_found, sim_preds, basename), fg='red', bold=True)
        cv2.imshow(sim_preds, demo)


cv2.waitKey(0)
