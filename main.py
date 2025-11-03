from torchvision import transforms, datasets
import json
import os
import time
import torchvision.models.resnet
from client import *
import copy
import random
from thop import profile
import mongo
import csv

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

import sys
import os

T = 3
PR = 0.5
suffix = ""

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# sys.stdout = Logger(f'res50-t{int(T)}-pr{int(PR*10)}{suffix}.txt')

def imagenet100_dataset():
    dir = '/home/lyh/python_projects/data/imagenet-100/'
    mean=[0.4576, 0.4529, 0.3916]
    std=[0.1990, 0.1942, 0.1928]
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = datasets.ImageFolder(dir+'train/', transform_train)
    test_set = datasets.ImageFolder(dir+'val/', transform_test)
    # train_set, test_set = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * 0.8), int(len(full_dataset) * 0.2)])
    class_size = 100
    return train_set, test_set, class_size

def imagenet1k_dataset():
    dir = '/home/lyh/python_projects/data/imagenet-1k/'
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop(32, padding=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = datasets.ImageFolder(dir+'train/', transform_train)
    test_set = datasets.ImageFolder(dir+'val/', transform_test)
    # train_set, test_set = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * 0.8), int(len(full_dataset) * 0.2)])
    class_size = 1000
    return train_set, test_set, class_size

def cifar100_dataset():
    CIFAR_PATH = "../data/"
    # mean = [125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0]
    # std = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0] #这是错的
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), #这里必须要padding=4,因为图已经很小了，如果边缘不填充，卷积损失的边缘信息足以大大降低准确率。
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.5,
        #                          scale=(0.02, 0.33),
        #                          ratio=(0.3, 3.3),
        #                          value=0,
        #                          inplace=False),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    cifar100_training = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=True, download=False,
                                                      transform=transform_train)

    cifar100_testing = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=False, download=False,
                                                     transform=transform_test)
    class_size = 100

    return cifar100_training, cifar100_testing, class_size


def flower_dataset():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # 来自官网参数
        "val": transforms.Compose([transforms.Resize(256),  # 将最小边长缩放到256
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.getcwd()
    image_path = data_root + "/flower_data/"  # flower data set path

    train_dataset = datasets.ImageFolder(root=image_path + "train",
                                         transform=data_transform["train"])

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                            transform=data_transform["val"])
    class_size = 5

    return train_dataset, validate_dataset, class_size


train_dataset, validate_dataset, class_size = imagenet100_dataset()
train_num = len(train_dataset)
# div = [int(0.1 * train_num), int(0.2 * train_num), int(0.2 * train_num), int(0.1 * train_num)]
# temp_sets = torch.utils.data.random_split(train_dataset, [train_num - sum(div)] + div)
# train_dataset = temp_sets[0]
# train_set = temp_sets[1:]

val_num = len(validate_dataset)

# net = resnet34()
# load pretrain weights

# model_weight_path = "./resnet34-pre.pth"
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)#载入模型参数

# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure

# inchannel = net.fc.in_features
# net.fc = nn.Linear(inchannel, 5)

load_path = 'imagenet_100_resNet50.pth'
save_path = 'imagenet_100_resNet50_1.pth'


def prune_test(server):
    prune_rate = 0.6
    print("server pruning:")
    inp = torch.randn(1, 3, 32, 32).to(device)
    macs_before, params_before = profile(server.model, inputs=(inp,))
    acc_before = server.test()
    server.get_loss()
    server.prune(prune_rate, 0.5)
    # server.test()
    macs_after, params_after = profile(server.model, inputs=(inp,))
    print("FLOPs_rate = {0:.4f} G\tParams_rate = {1:.4f} M".format(1 - macs_after / macs_before,
                                                                   1 - params_after / params_before))

    print("server fine_tuning:")
    for epoch in range(150):  # int(prune_rate * 50)
        t1 = time.time()
        server.train()
        server.test()
        t2 = time.time()
        print("epoch {0} time:{1:.2f}".format(epoch, t2 - t1))

    acc_after = server.test()
    print("acc_rate = {0}".format(1 - acc_after / acc_before))

    print("\n")


def distillation_test(server, student=None):
    t = T
    prune_rate = PR
    print("t = {0}, pr = {1:.1f}". format(t, prune_rate))
    theta = 0.5
    inp = torch.randn(1, 3, 244, 244).to(device)
    if student is None:
        teacher = copy.deepcopy(server.model)
        print("pruning:")
        macs_before, params_before = profile(server.model, inputs=(inp,))
        acc_before = server.test()
        server.get_loss()
        server.prune(prune_rate, 0.5)
        # server.test()
        macs_after, params_after = profile(server.model, inputs=(inp,))
        print("before FLOPs_rate = {0:.4f} G\tParams_rate = {1:.4f} M".format(macs_before, params_before))
        print("after FLOPs_rate = {0:.4f} G\tParams_rate = {1:.4f} M".format(macs_after, params_after))
    else:
        teacher = server
        server = student
        # macs_before, params_before = profile(teacher.model, inputs=(inp,))
        # macs_after, params_after = profile(server.model, inputs=(inp,))
        # print("FLOPs_rate = {0:.4f} G\tParams_rate = {1:.4f} M".format(1 - macs_after / macs_before,
        #                                                                1 - params_after / params_before))
        acc_before = teacher.test()
        teacher = teacher.model

    print("distilling:")
    max_acc = server.test()
    for epoch in range(150):  # int(prune_rate * 50)
        t1 = time.time()
        # server.train()
        server.distill(teacher, t, theta)
        acc = server.test()
        t2 = time.time()
        print("epoch {0} time:{1:.2f}".format(epoch, t2 - t1))
        if acc > max_acc:
            # torch.save(server.model.state_dict(), save_path)
            max_acc = acc
    print("max acc: {0:4f}".format(max_acc))

    acc_after = server.test()
    print("acc_rate = {0}".format(1 - acc_after / acc_before))

    print("\n")


def federal_test(server: PreparedModel):
    print("server pruning:")
    server.test()
    server.get_loss()

    inp = torch.randn(1, 3, 32, 32).to(device)
    macs_before, params_before = profile(server.model, inputs=(inp,))
    server.prune(0.3, 0.3)
    macs_after, params_after = profile(server.model, inputs=(inp,))
    print("before FLOPs_rate = {0:.4f} G\tbefore Params_rate = {1:.4f} M".format(macs_before, params_before))
    print("after FLOPs_rate = {0:.4f} G\tafter Params_rate = {1:.4f} M".format(macs_after, params_after))

    server.test()

    # print("server fine_tuning:")
    # for epoch in range(15):
    #     t1 = time.time()
    #     server.train()
    #     server.test()
    #     t2 = time.time()
    #     print("epoch {0} time:{1:.2f}".format(epoch, t2 - t1))
    # print("\n")

    clients = []
    for i in range(len(train_set)):
        clients.append(PreparedModel(train_dataset, validate_dataset, copy.deepcopy(server.model), device))

    for k in range(30):
        random_clients = random.sample(clients, int(0.8 * len(clients)))
        for i in range(len(random_clients)):
            print("client {0} training:".format(i))
            for j in range(3):
                random_clients[i].train()
                random_clients[i].test()

        server.parameters_aggregation(random_clients)

        print("server aggregated {0}:".format(k))
        server.test()
        print("\n")

        for client in random_clients:
            client.parameters_synchronization(server)


def write_in_csv(row):
    out = open(".\\data.csv", "a", newline="")
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow([str(row)])
    out.close()


trained = True
total_epochs = 150

def main():

    server = PreparedModel(train_dataset, validate_dataset, resnet50(num_classes=class_size), device)
    server.total_epochs = total_epochs
    # server.model.load_state_dict(torch.load(load_path))
    # num=0
    # for i, p in enumerate(server.model.parameters()):
    #     if i < 150:
    #         p.requires_grad = False
    #     num = i
    # print(num)
    if not trained:
        # mongo.start_draw_final()

        # max_acc = 0
        max_acc = server.test()
        for epoch in range(total_epochs):
            t1 = time.time()
            server.train()
            acc = server.test()
            # write_in_csv(acc)
            t2 = time.time()
            print("epoch {0} time:{1:.2f}".format(epoch, t2 - t1))
            if acc > max_acc:
                torch.save(server.model.state_dict(), save_path)
                max_acc = acc
        # torch.save(server.model.state_dict(), save_path)
    else:
        server.model.load_state_dict(torch.load(save_path))
    # federal_test(server)
    # prune_test(server)
    distillation_test(server)
    # student = PreparedModel(train_dataset, validate_dataset, resnet34(num_classes=class_size), device)
    # student.model.load_state_dict(torch.load("resNet34_0.pth"))
    #
    # distillation_test(student, server)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
