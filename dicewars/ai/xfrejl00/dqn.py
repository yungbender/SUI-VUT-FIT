from torch import from_numpy, squeeze, float32, set_default_dtype, tensor, load, save
from torch.nn import Linear, parameter, LeakyReLU, Sigmoid, Module, SmoothL1Loss, Dropout, BatchNorm1d, Tanh, Softmax, MSELoss
from torch.optim import Adam
import numpy as np
import copy
import argparse
import pickle

class Dataset:
    def __init__(self, data, labels, cols):
        self.xs = np.loadtxt(data, usecols=cols, dtype=float)
        self.targets = np.loadtxt(labels, dtype=float)

class LogisticRegressionMultiFeature(Module):
    def __init__(self, nb_features, lr=0):
        super().__init__()
        self.max_accuracy = parameter.Parameter(tensor(0.0))
        self.output_size = 1
        self.nb_features = nb_features
        self.linear = Linear(self.nb_features, 1)
        self.linear2 = Linear(16, 16)
        self.linear3 = Linear(16, 1)
        self.leakyrelu = LeakyReLU()
        self.tanh = Tanh()
        self.criterion = MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=0.001)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return squeeze(x)
    
    def prob_class_1(self, x):
        prob = self(from_numpy(x).float())
        return prob.detach().numpy()

    def batch_provider(self, xs, targets, batch_size=10):
        for _ in range(int(np.ceil(len(xs) / batch_size))):
            indices = np.random.randint(0, len(xs), batch_size)
            batch_samples = tensor(xs[indices], dtype=float32)
            batch_targets = tensor(targets[indices], dtype=float32)
            yield batch_samples, batch_targets

    def training_step(self, x, t):
        self.optimizer.zero_grad()
        model_output = self(x)
        loss = self.criterion(model_output, squeeze(t)) # Compare model output with targets
        current_loss = loss.detach().item()
        loss.backward()
        self.optimizer.step()

        return current_loss

    def evaluate(self, inputs, targets):
        probabilities = self.prob_class_1(inputs)
        probabilities = np.where(probabilities >= 0.5, 1, 0)
        return np.sum(probabilities == targets) / len(targets)

def train_model(nb_features, nb_epochs, lr, batch_size, train_dataset, val_dataset, load_model=False):
    model = LogisticRegressionMultiFeature(nb_features, lr)
    if load_model:
        model.load_state_dict(load(f"dicewars/ai/xfrejl00/dqn_model_{nb_features}.pt"))

    best_model = copy.deepcopy(model)
    losses = []
    accuracies = []
    print(f"Best model accuracy so far: {model.max_accuracy.double():0.4f}")
    
    for epoch in range(1, nb_epochs+1):
        loss_epoch = 0
        for x, t in model.batch_provider(train_dataset.xs, train_dataset.targets, batch_size):
            loss_epoch += model.training_step(x, t)

        avg_loss = loss_epoch / (len(train_dataset.xs) / batch_size)
        accuracy = model.evaluate(val_dataset.xs, val_dataset.targets)
        if accuracy >= model.max_accuracy.double():
            model.max_accuracy = parameter.Parameter(tensor(accuracy), requires_grad=False)
            best_model = copy.deepcopy(model)
            epoch_max = epoch

        print(f"Epoch {epoch}/{nb_epochs}: loss - {avg_loss:0.4f}, validation accuracy: {accuracy:0.4f}")
        accuracies.append(accuracy)
        losses.append(avg_loss)

    return best_model, losses, accuracies

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", dest="load_model", action="store_true", default=False, help="Select whether model snapshot will be loaded.")
    args = parser.parse_args()

    max_epochs = 500
    nb_features = 7 # 5 states, reward, winrate prediction, board states, custom stats
    learning_rate = 0.0001
    batch_size = 1

    train_dataset = Dataset("dicewars/ai/xfrejl00/dqn_data.trn", "dicewars/ai/xfrejl00/dqn_labels.trn", cols=np.arange(0, nb_features)) 
    val_dataset = Dataset("dicewars/ai/xfrejl00/dqn_data.val", "dicewars/ai/xfrejl00/dqn_labels.val", cols=np.arange(0, nb_features))

    model_multi_fea, losses, accuracies = train_model(nb_features, max_epochs, learning_rate, batch_size, train_dataset, val_dataset, args.load_model)
    print(f"Best model accuracy: {model_multi_fea.max_accuracy.double():0.4f}")

    save(model_multi_fea.state_dict(), f"dicewars/ai/xfrejl00/dqn_model_{nb_features}.pt")

    import matplotlib.pyplot as plt
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