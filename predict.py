import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import random

from model import fpn_classification

def get_image():
    folder_path = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    image_path =os.path.join(folder_path, "data_set", "flower_data","test")

    if not os.path.exists(image_path):
        print(f"Folder {image_path} does not exist.")
        return None
    all_files = os.listdir(image_path)
    image_files = [f for f in all_files if f.lower().endswith('.jpg')]
    choice = random.choice(image_files)
    return os.path.join(image_path, choice)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #加载照片
    img_path = get_image()
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    #加载类别数据
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    #加载模型
    model = fpn_classification(num_classes=5).to(device)

    #加载权重
    weights_path = "./fpn_classification.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    #进行预测
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
