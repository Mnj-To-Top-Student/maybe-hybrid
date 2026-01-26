import numpy as np
import time
import torch
from flcore.clients.clientbase import Client


class clientBABU(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.fine_tuning_epochs = 10

        # Update optimizer to only include trainable parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.SGD(
                self.model.head.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4
            )

    def train(self):
        # Freeze backbone
        for p in self.model.base.cnn[:4].parameters():
            p.requires_grad = False

        # Train later CNN + transformer + head
        for p in self.model.base.cnn[4:].parameters():
            p.requires_grad = True
        for p in self.model.base.transformer.parameters():
            p.requires_grad = True
        for p in self.model.head.parameters():
            p.requires_grad = True
        
        # Unfreeze head
        # for param in self.model.head.parameters():
        #     param.requires_grad = True

        self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=1e-4
                )
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for _ in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.loss(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # for epoch in range(max_local_epochs):
        #     for i, (x, y) in enumerate(trainloader):

        #         # if epoch == 0 and i == 0:
        #         #     print("Backbone grad example:",
        #         #         next(self.model.base.parameters()).requires_grad)
        #         #     print("Head grad example:",
        #         #         next(self.model.head.parameters()).requires_grad)

        #         if type(x) == type([]):
        #             x[0] = x[0].to(self.device)
        #         else:
        #             x = x.to(self.device)
        #         y = y.to(self.device)
        #         if self.train_slow:
        #             time.sleep(0.1 * np.abs(np.random.rand()))
        #         output = self.model(x)
        #         loss = self.loss(output, y)
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # Explicitly expose metrics to avoid missing attribute errors
    def test_metrics(self):
        return super().test_metrics()

    def train_metrics(self):
        return super().train_metrics()

    def set_parameters(self, base):
        for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    # def fine_tune(self, which_module=['base', 'head']):
    #     trainloader = self.load_train_data()
        
    #     start_time = time.time()
        
    #     self.model.train()

    #     if 'head' in which_module:
    #         for param in self.model.head.parameters():
    #             param.requires_grad = True

    #     if 'base' not in which_module:
    #         for param in self.model.base.parameters():
    #             param.requires_grad = False
            

    #     for epoch in range(self.fine_tuning_epochs):
    #         for i, (x, y) in enumerate(trainloader):
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             loss = self.loss(output, y)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #     self.train_time_cost['total_cost'] += time.time() - start_time
    def fine_tune(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        # 🔒 Freeze backbone completely
        for p in self.model.base.parameters():
            p.requires_grad = False

        # 🔓 Train ONLY head
        for p in self.model.head.parameters():
            p.requires_grad = True

        # 🚀 Aggressive optimizer for head
        optimizer = torch.optim.AdamW(
            self.model.head.parameters(),
            lr=self.learning_rate * 5,   # THIS is the key change
            weight_decay=1e-4
        )

        for epoch in range(self.fine_tuning_epochs):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)

                out = self.model(x)
                loss = self.loss(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_time_finetune(self, epochs=5, lr=1e-3):
        self.model.train()

        # Freeze everything except head
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.head.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.model.head.parameters(), lr=lr)
        loss_fn = self.loss

        loader = self.load_test_data(batch_size=16)

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = loss_fn(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
