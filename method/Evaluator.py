import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class Evaluator:
    """
    Evaluate a deep neural network by provided dataset
    """

    def __init__(self, model, dataset, batch_size=False):
        """
        Initialize evaluator
        :param model: Pytorch neural network
        :param dataset: Dataset for evaluation
        :param batch_size: How many data are extracted for evaluation at a time
        """
        # Acceleration with cuda
        if torch.cuda.is_available():
            print('cuda activated')
            model = model.to(torch.device('gpu'))
        else:
            print('cuda not activated')
            model = model.to(torch.device('cpu'))
        with torch.no_grad():
            # Switch the model to evaluation mode
            model.eval()
            # Use torch's DataLoader to load the data
            if not batch_size:
                batch_size = len(dataset)
            testing_loader = DataLoader(dataset, batch_size, shuffle=False)
            # Create the list for required value
            self.label_pred_list = []
            self.label_true_list = []
            loss_list = []
            # Use models to make predictions
            for data in testing_loader:
                img, label_true = data
                # Acceleration with cuda
                if torch.cuda.is_available():
                    img = img.to(torch.device('gpu'))
                    label_true = label_true.to(torch.device('gpu'))
                else:
                    img = img.to(torch.device('cpu')) 
                    label_true = label_true.to(torch.device('cpu')) 
                # Calculate output
                out = model(img)
                # Calculate loss
                criterion = nn.CrossEntropyLoss()
                loss = criterion(out, label_true)
                # Calculate predict value
                label_pred = torch.argmax(out, dim=1)
                # Add the predict results to list
                self.label_pred_list = self.label_pred_list + label_pred.tolist()
                self.label_true_list = self.label_true_list + label_true.tolist()
                loss_list.append(loss)
            # Calculate average loss
            self.loss_value = sum(loss_list) / len(loss_list)

    def get_loss(self) -> float:
        """
        :return: The loss value of the model on the current dataset
        """
        return self.loss_value

    def get_result(self) -> tuple[list, list]:
        """
        :return: A tuple with 2 list. One list records the predicted results of the model, and the other list records the true results
        """
        return self.label_pred_list, self.label_true_list

    def get_accuracy(self) -> float:
        """
        :return: The accuracy of the model on the current dataset
        """
        correct = sum(1 for x, y in zip(self.label_pred_list, self.label_true_list) if x == y)
        error = sum(1 for x, y in zip(self.label_pred_list, self.label_true_list) if x != y)
        return correct / (correct + error)

    def plot_confusion_matrix(self, classes: list, figsize=(10, 10)):
        """
        Plot the confusion matrix for this model on the current dataset
        :param classes: A list of multiple strings. Each string represents a class
        :param figsize: Figure size of matplotlib.pyplot. Defaults to (10, 10)
        """
        # Calculate
        confusion = confusion_matrix(self.label_true_list, self.label_pred_list)
        # Set canvas
        plt.figure(figsize=figsize)
        # Thermodynamic diagram
        plt.imshow(confusion, cmap=plt.cm.Blues)
        # Set legend
        plt.colorbar()
        # Set axis scale
        indices = range(len(confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        # Set title
        plt.xlabel("Predict")
        plt.ylabel("True")
        plt.title("Confusion Matrix", fontsize=12)
        # Displays specific values on confusion matrix
        for i in range(len(confusion)):
            for j in range(len(confusion[i])):
                plt.text(j, i, format(confusion[i][j], "d"), fontsize=16, horizontalalignment="center", verticalalignment="center", color="white" if confusion[i, j] > confusion.max() / 2.0 else "black")
        plt.show()
