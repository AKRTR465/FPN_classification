import os
import sys
import json
import torch
import torch.nn as nn
import random
import numpy as np

from torchvision import transforms, datasets
from tqdm import tqdm
from model import fpn_classification

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def set_seed(seed):
    """
    固定所有可能的随机种子以确保实验的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def main():
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    #创建TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/fpn_classification_{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    #初始化训练日志
    training_log = {
        'model_name': 'fpn_classification',
        'seed': 2025,
        'batch_size': 32,
        'epochs': 40,
        'optimizer': 'SGD',
        'lr': 0.004,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'device': str(device),
        'training_history': []
    }

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "data_set", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                        transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    #写入类别字典到json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    #批次数
    batch_size = 48

    #设置线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                           transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    #记录数据信息到日志
    training_log['train_samples'] = train_num
    training_log['val_samples'] = val_num
    training_log['num_classes'] = len(cla_dict)

    model = fpn_classification()
    model = model.to(device)

    #设置损失函数
    loss_function = nn.CrossEntropyLoss()

    #设置优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.004, momentum=0.9, weight_decay=0.0001)

    model_name = 'fpn_classification'
    print("using {} model.".format(model_name))

    epochs = 60
    best_acc = 0.0
    save_path = f'./{model_name}.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
        #训练
        model.train()
        running_loss = 0.0
        train_acc = 0.0
        train_samples = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            #计算训练准确率
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            train_samples += labels.size(0)

            #打印日志
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 计算平均训练损失和准确率
        train_loss = running_loss / train_steps
        train_accuracy = train_acc / train_samples

        # validate
        model.eval()
        val_acc = 0.0
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                val_steps += 1

                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = val_acc / val_num
        val_loss_avg = val_loss / val_steps

        #获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        #记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss_avg, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accurate, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        #记录到日志字典
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_accuracy),
            'val_loss': float(val_loss_avg),
            'val_acc': float(val_accurate),
            'learning_rate': float(current_lr)
        }
        training_log['training_history'].append(epoch_log)

        print('[epoch %d] train_loss: %.3f train_acc: %.3f val_loss: %.3f val_accuracy: %.3f lr: %.6f' %
              (epoch + 1, train_loss, train_accuracy, val_loss_avg, val_accurate, current_lr))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
            print(f'best acc:{best_acc:.3f}')

    #记录最终结果
    training_log['best_val_acc'] = float(best_acc)
    training_log['final_train_acc'] = float(train_accuracy)
    training_log['final_val_acc'] = float(val_accurate)

    #保存训练日志到 JSON 文件
    log_filename = f'training_log_{timestamp}.json'
    with open(log_filename, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f'Training log saved to: {log_filename}')

    #关闭TensorBoard
    writer.close()
    
    print('Finished Training')


if __name__ == '__main__':
    main()
