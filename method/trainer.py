import torch
from torch import nn
from torch.utils.data import DataLoader
from method.optimizer.lion import Lion
from method.optimizer.signal_momentum import Signum


class Trainer:
    """
    Train a deep neural network and see how it works on a validation set during training
    """

    def __init__(self, model, batch_size: int, num_epoch: int, training_set, validation_set=None, validate_step_size=10):
        """
        Initialization Trainer. Criterion defaults to nn.CrossEntropyLoss()
        :param model: Pytorch neural network
        :param batch_size: How many data are extracted for training/validation at a time
        :param num_epoch: How many times will the training_set be iteratively trained
        :param training_set: Dataset for training
        :param validation_set: Dataset for validation. (optional) Defaults to None mean does not compute the model's loss value on the validation set during training.
        :param validate_step_size: How many epochs to calculate the loss value of model on validation_set. (optional) Defaults to 10. (This parameter is invalid when validation_set is False)
        """
        # Acceleration with cuda
        if torch.cuda.is_available():
            print('cuda activated')
            model = model.to(torch.device('cuda'))
        else:
            print('cuda not activated')
            model = model.to(torch.device('cpu'))        
        self.model = model
        # Initialize the global variable
        self.num_epoch = num_epoch
        self.validate_step_size = validate_step_size
        # Use torch's DataLoader to load the data
        self.training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)
        if validation_set is None:
            self.validation_loader = None
        else:
            self.validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, drop_last=True)
        # Set criterion of loss function
        self.criterion = nn.CrossEntropyLoss()

    def __calculate_loss(self, data: torch.Tensor) -> torch.Tensor:
        """
        Private function to calculate the loss
        :param data: Data required to calculate loss
        :return: loss of data
        """
        # Extract label and input
        img, label = data
        # Acceleration with cuda
        if torch.cuda.is_available():
            img = img.to(torch.device('cuda'))
            label = label.to(torch.device('cuda'))
        else:
            img = img.to(torch.device('cpu'))   
            label = label.to(torch.device('cpu'))   
        # Calculate loss
        out = self.model(img)
        loss = self.criterion(out, label)
        # Return
        return loss

    def __train_model(self, optimizer) -> tuple[list, list, list, list]:
        """
        Private function to training model
        :param optimizer: Optimizer based on Pytorch
        :return: A tuple with 4 list. It records the loss value and index of training/validation set
        """
        # Initialize
        loss_list_index_training, loss_list_value_training = [], []
        loss_list_index_validation, loss_list_value_validation = [], []
        # Iterative for epoch
        for epoch in range(self.num_epoch):
            # Training in each epoch
            for index, data in enumerate(self.training_loader):
                # Initialize
                loss_list = []
                # Calculate loss
                loss = self.__calculate_loss(data)
                # Update parameter
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Keep track of loss value
                loss_list.append(loss.data.item())
                print(f"epoch: {epoch} steps: {index} loss: {loss.data.item():.4f}")
            # Keep track of loss value
            loss_list_index_training.append(epoch)
            loss_list_value_training.append(sum(loss_list) / len(loss_list))
            # Validate as required
            if self.validation_loader is not None and epoch % self.validate_step_size == 0:
                with torch.no_grad():
                    # validation in each epoch
                    for index, data in enumerate(self.validation_loader):
                        # Initialize
                        loss_list = []
                        # calculate loss
                        loss = self.__calculate_loss(data)
                        # Keep track of loss value
                        loss_list.append(loss.data.item())
                # Keep track of loss value
                loss_list_index_validation.append(epoch)
                loss_list_value_validation.append(sum(loss_list) / len(loss_list))
                print(f"validation_set epoch: {epoch} loss: {sum(loss_list)/len(loss_list):.4f}")
        # Return
        return loss_list_index_training, loss_list_value_training, loss_list_index_validation, loss_list_value_validation

    def sgd(self, learning_rate: float) -> tuple[list, list, list, list]:
        """
        Training a model by using SGD
        :param learning_rate: Learning rate of training
        :return: A tuple with 4 list. It records the loss value and index of training/validation set (Used to track the training process and does NOT affect the training effect)
        """
        # Initialize optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        # Training
        print("Training by SGD")
        return self.__train_model(optimizer)

    def momentum(self, learning_rate: float, beta_1: float) -> tuple[list, list, list, list]:
        """
        Training a model by using momentum
        :param learning_rate: Learning rate of training
        :param beta_1: Factor for first-order moment estimation
        :return: A tuple with 4 list. It records the loss value and index of training/validation set (Used to track the training process and does NOT affect the training effect)
        """
        # Initialize optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=beta_1, dampening=(1 - beta_1))
        # Training
        print("Training by momentum")
        return self.__train_model(optimizer)

    def rmsprop(self, learning_rate: float, beta_2: float) -> tuple[list, list, list, list]:
        """
        Training a model by using RMSprop
        :param learning_rate: Learning rate of training
        :param beta_2: Factor for Second-order moment estimation
        :return: A tuple with 4 list. It records the loss value and index of training/validation set (Used to track the training process and does NOT affect the training effect)
        """
        # Initialize optimizer
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, alpha=beta_2, centered=True)
        # Training
        print("Training by RMSprop")
        return self.__train_model(optimizer)

    def adam(self, learning_rate: float, beta_1: float, beta_2: float, weight_decay=0) -> tuple[list, list, list, list]:
        """
        Training a model by using adam
        :param learning_rate: Learning rate of training
        :param beta_1: Factor for first-order moment estimation
        :param beta_2: Factor for Second-order moment estimation
        :param weight_decay: L2 regularization coefficient. Defaults to 0.
        :return: A tuple with 4 list. It records the loss value and index of training/validation set (Used to track the training process and does NOT affect the training effect)
        """
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(beta_1, beta_2), weight_decay=weight_decay)
        # Training
        print("Training by adam")
        return self.__train_model(optimizer)
    
    def signum(self, learning_rate: float, beta_1: float) -> tuple[list, list, list, list]:
        """
        Training a model by using signum
        :param learning_rate: Learning rate of training
        :param beta_1: Factor for first-order moment estimation
        :return: A tuple with 4 list. It records the loss value and index of training/validation set (Used to track the training process and does NOT affect the training effect)
        """
        # Initialize optimizer
        optimizer = Signum(self.model.parameters(), lr=learning_rate, beta=beta_1)
        # Training
        print("Training by Signum")
        return self.__train_model(optimizer)
    

    def lion(self, learning_rate: float, beta_1: float, beta_3: float) -> tuple[list, list, list, list]:
        """
        Training a model by using lion
        :param learning_rate: Learning rate of training
        :param beta_1: Factor for first-order moment estimation
        :param beta_3: Factor for weighted update
        :return: A tuple with 4 list. It records the loss value and index of training/validation set (Used to track the training process and does NOT affect the training effect)
        """
        # Initialize optimizer
        optimizer = Lion(self.model.parameters(), lr=learning_rate, betas=(beta_1, beta_3))
        # Training
        print("Training by Lion")
        return self.__train_model(optimizer)
