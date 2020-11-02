import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
import numpy as np
import copy
import matplotlib.pyplot as plt
import argparse
import pickle

class Dataset:
    def __init__(self, positives, negatives):
        self.pos = np.loadtxt(positives)
        self.neg = np.loadtxt(negatives)
        self.xs = np.concatenate((self.pos, self.neg))
        self.targets = np.concatenate((np.ones((len(self.pos), )), np.zeros((len(self.neg), ))))

train_dataset = Dataset("dicewars/ai/xfrejl00/positives.trn", "dicewars/ai/xfrejl00/negatives.trn") 
val_dataset = Dataset("dicewars/ai/xfrejl00/positives.val", "dicewars/ai/xfrejl00/negatives.val")


class LogisticRegressionMultiFeature(torch.nn.Module):
    def __init__(self, nb_features):
        super().__init__()
        self.max_accuracy = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.output_size = 1
        self.nb_features = nb_features
        self.linear = torch.nn.Linear(self.nb_features, 16)
        self.linear2 = torch.nn.Linear(16, 16)
        self.linear3 = torch.nn.Linear(16, 1)
        self.leakyrelu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.leakyrelu(x)
        x = self.linear2(x)
        x = self.leakyrelu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return torch.squeeze(x)
    
    def prob_class_1(self, x):
        prob = self(torch.from_numpy(x))
        return prob.detach().numpy()

def evaluate(classifier, inputs, targets):
    probabilities = classifier.prob_class_1(inputs)
    probabilities = np.where(probabilities >= 0.5, 1, 0)
    return np.sum(probabilities == targets) / len(targets)


def batch_provider(xs, targets, batch_size=10):
    for _ in range(int(np.ceil(len(xs) / batch_size))):
        indices = np.random.randint(0, len(xs), batch_size)
        batch_samples = torch.tensor(xs[indices], dtype=float)
        batch_targets = torch.tensor(targets[indices], dtype=float)
        yield batch_samples, batch_targets


def train_multi_fea_llr(nb_features, nb_epochs, lr, batch_size, load_model=False):
    model = LogisticRegressionMultiFeature(nb_features)
    if load_model:
        model.load_state_dict(torch.load("dicewars/ai/xfrejl00/classifier_model.pt"))
        model.eval()

    best_model = copy.deepcopy(model)
    losses = []
    accuracies = []
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Best model accuracy so far: {model.max_accuracy.double()}")
    
    for epoch in range(1, nb_epochs+1):
        loss_epoch = 0
        for x, t in batch_provider(train_dataset.xs, train_dataset.targets, batch_size):
            optimizer.zero_grad()
            model_output = model(x)
            loss = criterion(model_output, t) # Compare model output with targets
            loss_epoch += loss
            loss.backward()
            optimizer.step()

        avg_loss = loss_epoch / (len(train_dataset.xs) / batch_size)
        accuracy = evaluate(model, val_dataset.xs, val_dataset.targets)
        if accuracy >= model.max_accuracy.double():
            best_model = copy.deepcopy(model)
            model.max_accuracy = torch.nn.parameter.Parameter(torch.tensor(accuracy), requires_grad=False)
        print(f"Epoch {epoch}/{nb_epochs}: loss - {avg_loss}, validation accuracy: {accuracy}")
        accuracies.append(accuracy)
        losses.append(avg_loss)
    return best_model, losses, accuracies


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", dest="load_model", action="store_true", default=False, help="Select whether model snapshot will be loaded.")
    args = parser.parse_args()

    max_epochs = 500
    nb_features = 11
    learning_rate = 0.001
    batch_size = 128

    model_multi_fea, losses, accuracies = train_multi_fea_llr(nb_features, max_epochs, learning_rate, batch_size, args.load_model)
    print(f"Best model accuracy: {model_multi_fea.max_accuracy.double()}")

    torch.save(model_multi_fea.state_dict(), "dicewars/ai/xfrejl00/classifier_model.pt")

    plt.ylabel("Loss value")
    plt.xlabel("Epochs")
    x = np.linspace(1, max_epochs, max_epochs)
    plt.plot(x, losses, color="blue")
    plt.legend(["Training loss"])
    plt.show()
    plt.close()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(x, accuracies, color="blue")
    plt.legend(["Validation accuracy"])
    plt.show()
    plt.close()


def main():
    train()

if __name__ == "__main__":
    main()