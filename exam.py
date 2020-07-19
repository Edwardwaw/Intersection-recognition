import argparse
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models.resnet import resnet50
import os
import PIL

from model import EfficientNet


#__INPUT__dataset path 
DATA_EXAM_PATH = 'test'

#mean and std of dataset
# TEST_MEAN = (0.485, 0.456, 0.4063)
# TEST_STD = (0.229, 0.224, 0.22)
TEST_MEAN = (0.422,0.421,0.409)
TEST_STD = (0.264,0.261,0.277)


#default weigth path
WEIGTH_PATH = "Efficient21-best.pth"

#output class 
out_classes = ('A1', 'A2', 'A3', 'A4',          \
                'B1', 'B2', 'B3', 'B4',         \
                'C1', 'C2', 'C3', 'C4',     \
                'D1', 'D2', 'D3', 'D4','D5',     \
                'E1', 'E2', 'E3', 'E4','E5',          \
                'F1', 'F2', 'F3', 'F4', 'F5',    \
                'G1', 'G2', 'G3', 'G4',       \
                'H1', 'H2', 'H3', 'H4',     \
                'I1', 'I2', 'I3', 'I4',     \
                )


def dat_label_writer(label, file):

    label_no = 0

    if label == 'A1' or label == 'A2' or label == 'A3' or label == 'A4':
        label_no = 1
    elif label == 'B1' or label == 'B2' or label == 'B3' or label == 'B4':
        label_no = 2
    elif label == 'C1' or label == 'C2' or label == 'C3' or label == 'C4':
        label_no = 3
    elif label == 'D1' or label == 'D2' or label == 'D3' or label == 'D4' or label == 'D5':
        label_no = 4
    elif label == 'E1' or label == 'E2' or label == 'E3' or label == 'E4' or label == 'E5':
        label_no = 5
    elif label == 'F1' or label == 'F2' or label == 'F3' or label == 'F4' or label == 'F5':
        label_no = 6
    elif label == 'G1' or label == 'G2' or label == 'G3' or label == 'G4':
        label_no = 7
    elif label == 'H1' or label == 'H2' or label == 'H3' or label == 'H4':
        label_no = 8
    elif label == 'I1' or label == 'I2' or label == 'I3' or label == 'I4':
        label_no = 9

    file.write(str(label_no) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=WEIGTH_PATH, help='the weights file you want to test')
    args = parser.parse_args()

    # net = resnet50()
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 38)

    net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=39)

    net = net.cuda()

    print("__Resnet load success")

    net.load_state_dict(torch.load(args.weights), True)
    print(net)

    net.eval()

    # __ Walk the exam folder
    folder_path = DATA_EXAM_PATH
    path_list = os.listdir(folder_path)

    # Output .dat file
    datfile = open('label.dat', 'w')


    image_num = 1

    with torch.no_grad():
        while image_num <= len(path_list):
                if os.path.exists(os.path.join(folder_path, str(image_num)+'.jpg')):
                    image_path = os.path.join(folder_path, str(image_num)+'.jpg')
                else:
                    image_path = os.path.join(folder_path, str(image_num)+'.png')

                image = PIL.Image.open(image_path).convert('RGB')

                transform_test = transforms.Compose([
                    transforms.Resize(480),
                    transforms.ToTensor(),
                    transforms.Normalize(TEST_MEAN, TEST_STD)
                ])

                image = transform_test(image).unsqueeze(0)

                image = Variable(image).cuda()
                output = net(image)
                pred_score, pred = output.topk(5, 1, largest=True, sorted=True)

                print('Image: ', " ".join('%s' % image_num))
                # pred_score=torch.sigmoid(pred_score)
                pred_score_list=pred_score.tolist()
                pred_list = pred.tolist()
                print('Predicted: ', " ".join('%5s' % out_classes[pred_list[0][j]] for j in range(5)))
                dat_label_writer(out_classes[pred_list[0][0]], datfile)


                image_num = image_num + 1

    print('__ All image prediction process done')

