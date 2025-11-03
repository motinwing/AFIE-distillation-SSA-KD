import torch
from torch import nn, optim, tensor
from torch.utils.data import DataLoader
from resnet import *

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs, min_lr_ratio=0.0001, max_lr_ratio=0.1):
    """
    创建带Warmup的Cosine学习率调度器
    """
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Warmup阶段：线性增加到初始学习率
            return max_lr_ratio * max(float(current_epoch) / float(max(1, num_warmup_epochs)), min_lr_ratio)
        else:
            # Cosine下降阶段
            progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
            return max(min_lr_ratio, max_lr_ratio * 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class PreparedModel:
    def __init__(self, train_dataset, test_dataset, model: ResNet, device):
        self.device = device
        # 全局变量
        self.batch_size = 128  # 每次喂入的数据量
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)
        self.data_size = len(train_dataset)

        # self.epoch_num = 12  # 总迭代次数

        self.lr = 0.01
        self.step_size = 60  # 每n次epoch更新一次学习率
        self.gamma = 0.2

        self.model = model.to(device)

        # 在多分类情况下一般使用交叉熵
        self.softmax = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # self.schedule = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=-1)

        warmup_epochs = 5
        total_epochs = 150
        self.schedule = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_epochs=warmup_epochs,
            num_training_epochs=total_epochs,
            min_lr_ratio=0.0001,  # 最终学习率为初始学习率的1%
            max_lr_ratio=self.lr
        )

    def get_loss(self):
        self.model.eval()
        for i in range(2):
            data = next(iter(self.train_loader))
            images, labels = data
            logits = self.model(images.to(self.device))
            loss = self.loss_function(logits, labels.to(self.device))
            loss.backward()
        # for step, data in enumerate(self.train_loader, start=0):
        #     images, labels = data
        #     logits = self.model(images.to(self.device))
        #     loss = self.loss_function(logits, labels.to(self.device))
        #     loss.backward()

    def train(self):
        # train
        self.model.train()
        ep = 0.01
        running_loss = 0.0
        for step, data in enumerate(self.train_loader, start=0):
            images, labels = data
            self.optimizer.zero_grad()
            logits = self.model(images.to(self.device))
            # one_hot = torch.zeros(logits.size(0), 100).scatter_(1, labels.unsqueeze(1), 1-99.0/100.0*ep)
            # one_hot += 1.0/100.0*ep*torch.ones_like(one_hot)
            # loss = self.loss_function(logits, one_hot.to(self.device))
            loss = self.loss_function(logits, labels.to(self.device))
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print train process
            rate = (step + 1) / len(self.train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}    {:.4f}".format(int(rate * 100), a, b, running_loss / (step + 1), loss), end="")
        self.schedule.step()
        print()

    def distill(self, teacher, t, theta=1):
        # distill
        self.model.train()
        teacher.eval()
        running_loss = 0.0
        for step, data in enumerate(self.train_loader, start=0):
            images, labels = data
            self.optimizer.zero_grad()
            logits = self.model(images.to(self.device)) / t
            t_logits = self.softmax(teacher(images.to(self.device)) / t)
            loss1 = self.loss_function(logits, t_logits)
            loss2 = self.loss_function(logits, labels.to(self.device)) / (t*t)
            loss = (1 - theta) * loss1 + theta * loss2
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print train process
            # rate = (step + 1) / len(self.train_loader)
            # a = "*" * int(rate * 50)
            # b = "." * int((1 - rate) * 50)
            # print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}    {:.4f}".format(int(rate * 100), a, b, running_loss / (step + 1), loss), end="")
        self.schedule.step()
        print("\rtrain loss: {:.4f}".format(loss), end=" ")

    def test(self):
        # validate
        self.model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_num = 0.0
        with torch.no_grad():
            for val_data in self.test_loader:
                val_images, val_labels = val_data
                outputs = self.model(val_images.to(self.device))  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                val_num += val_images.size(0)
                acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()
            val_accurate = acc / val_num
            print('test_accuracy: %.3f' % val_accurate)
        return val_accurate

    def prune(self, global_sparsity, theta):
        self.model.resnet_prune(global_sparsity, theta)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=-1)

    def parameters_synchronization(self, server):
        model_dict = self.model.state_dict()
        for name in server.model.state_dict():
            model_dict[name] = server.model.state_dict()[name]
        self.model.load_state_dict(model_dict)

    def parameters_aggregation(self, clients):
        whole_size = 0
        for i, client in enumerate(clients):
            whole_size += client.data_size

        model_dict = self.model.state_dict()
        for name in self.model.state_dict():
            if model_dict[name].type() != 'torch.cuda.FloatTensor':
                continue
            model_dict[name] = torch.zeros_like(model_dict[name])
            for client in clients:
                model_dict[name] += (client.data_size / whole_size) * client.model.state_dict()[name]
        self.model.load_state_dict(model_dict)
