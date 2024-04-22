import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    """
    Train a deep neural network and see how it works on a validation set during training
    """

    def __init__(
        self,
        model,
        batch_size: int,
        num_epoch: int,
        training_set,
        validation_set=False,
        verify_step_size=10,
    ):
        """
            Initialization Trainer. Criterion defaults to nn.CrossEntropyLoss()
        Args:
            model (_type_): Pytorch neural network
            batch_size (int): How many data are extracted for training/validation at a time
            num_epoch (int): How many times will the training_set be iteratively trained
            training_set (TensorDataset): Data set for training
            validation_set (TensorDataset, optional): Data set for validation. Defaults to False mean does not compute the model's loss value on the validation set during training.
            verify_step_size (int, optional): How many epochs to calculate the loss value of model on validation_set. Defaults to 10. This parameter is invalid when validation_set is False.
        """
        self.model = model
        self.num_epoch = num_epoch
        self.training_loader = DataLoader(
            training_set, batch_size=batch_size, shuffle=True
        )
        if validation_set is False:
            self.validation_loader = False
        else:
            self.validation_loader = DataLoader(
                validation_set, batch_size=batch_size, shuffle=True
            )
        self.verify_step_size = verify_step_size
        self.criterion = nn.CrossEntropyLoss()

    def __calculate_loss(self, data: torch.Tensor) -> torch.Tensor:
        """
        Private function to calculate the loss
        :param data: Data required to calculate loss value
        :param criterion: Criteria for calculation of losses
        :return: loss value of data
        """
        # Reshape the data
        img, label = data
        # Acceleration with cuda
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = img
            label = label
        # Calculate loss
        out = self.model(img)
        loss = self.criterion(out, label)
        # Return
        return loss

    def __train_model(self, optimizer) -> tuple[list, list, list, list]:
        """
        Private function to training model
        :param optimizer: Optimizer based on Pytorch
        :param criterion: Scoring criteria based on Pytorch
        :param train_loader: Training dataset
        :param train_loader_for_loss: The dataset used to calculate the loss
        :return: A list consisting of all the loss values from the training process
        """
        # Initialize
        loss_list_index_training = []
        loss_list_value_training = []
        loss_list_index_validation = []
        loss_list_value_validation = []
        # Training
        for epoch in range(self.num_epoch):
            for index, data in enumerate(self.training_loader):
                loss_list = []
                loss = self.__calculate_loss(data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.data.item())
                print(f"epoch: {epoch} steps: {index} loss: {loss.data.item():.4f}")
            loss_list_index_training.append(epoch)
            loss_list_value_training.append(sum(loss_list) / len(loss_list))
            if epoch % self.verify_step_size == 0:
                with torch.no_grad():
                    for index, data in enumerate(self.validation_loader):
                        loss_list = []
                        loss = self.__calculate_loss(data)
                        # Keep track of solutions
                        loss_list.append(loss.data.item())
                loss_list_index_validation.append(epoch)
                loss_list_value_validation.append(sum(loss_list) / len(loss_list))
                print(
                    f"validation_set epoch: {epoch} loss: {sum(loss_list)/len(loss_list):.4f}"
                )
        # Return
        return (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        )

    def sgd(self, learning_rate: float) -> list:
        """
        Training a model by using SGD
        :param learning_rate: Learning rate of training
        :return: A list consisting of all the loss values from the training process
        """
        # Define criterion and optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        # Training
        print("Training by SGD")
        (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        ) = self.__train_model(optimizer)
        # Return
        return (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        )

    def momentum(self, learning_rate: float, beta_1: float) -> list:
        """
        Training a model by using momentum
        :param dataset: Total training set
        :param learning_rate: Learning rate of training
        :param beta_1: Factor for first-order moment estimation
        :return: A list consisting of all the loss values from the training process
        """
        # Define criterion and optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=beta_1,
            dampening=(1 - beta_1),
        )
        # Training
        print("Training by momentum")
        (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        ) = self.__train_model(optimizer)
        # Return
        return (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        )

    def rmsprop(self, dataset, learning_rate: float, beta_2: float) -> list:
        """
        Training a model by using RMSprop
        :param dataset: Total training set
        :param learning_rate: Learning rate of training
        :param beta_2: Factor for Second-order moment estimation
        :return: A list consisting of all the loss values from the training process
        """
        optimizer = optim.RMSprop(
            self.model.parameters(), lr=learning_rate, alpha=beta_2, centered=True
        )
        # Training
        print("Training by RMSprop")
        (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        ) = self.__train_model(optimizer)
        # Return
        return (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        )

    def adam(self, dataset, learning_rate: float, beta_1: float, beta_2: float) -> list:
        """
        Training a model by using adam
        :param dataset: Total training set
        :param learning_rate: Learning rate of training
        :param beta_1: Factor for first-order moment estimation
        :param beta_2: Factor for Second-order moment estimation
        :return: A list consisting of all the loss values from the training process
        """
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(beta_1, beta_2)
        )
        # Training
        print("Training by adam")
        (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        ) = self.__train_model(optimizer)
        # Return
        return (
            loss_list_index_training,
            loss_list_value_training,
            loss_list_index_validation,
            loss_list_value_validation,
        )
