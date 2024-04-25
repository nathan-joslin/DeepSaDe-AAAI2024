import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Optimizer import *
import time
import numpy as np
import torch.utils.data as data
import random


class FFNeuralNetTorch(nn.Module):
    """
    This is a FeedForward Neural Network Implementation with skip-connections.
    """
    def __init__(self, sizes, mapped_features=None, activation='relu'):
        super().__init__()
        self.sizes = sizes
        self.Layers = nn.ModuleList()
        self.mapped_features = mapped_features
        self.mapped_feature_weights = None
        self.runtime = None
        self.activation = activation
        if self.mapped_features is None:
            self.mapped_features = []
        else:
            self.initialize_mapped_weights()
        for i in range(len(self.sizes) - 1):
            self.Layers.append(torch.nn.Linear(self.sizes[1 + i - 1], self.sizes[1 + i]))

    def initialize_mapped_weights(self):
        self.mapped_feature_weights = torch.nn.Parameter(torch.empty(self.sizes[-1], len(self.mapped_features)))
        nn.init.normal_(self.mapped_feature_weights)

    def forward(self, X):
        out = X
        for i in range(len(self.Layers)):
            out = self.Layers[i](out)
            if i != len(self.Layers) - 1:
                out = F.relu(out)
            else:
                if len(self.mapped_features) > 0:
                    out = out + torch.matmul(self.mapped_feature_weights, X.T[self.mapped_features]).T
        return out

    def input_last_layer(self, X):
        out = X
        for i in range(len(self.Layers) - 1):
            out = self.Layers[i](out)
            out = F.relu(out)
        return out

    def inputs_all_layer(self, X):
        out = X
        output = []
        for i in range(len(self.Layers) - 1):
            out = self.Layers[i](out)
            out = F.relu(out)
            output.append(out)
        return output

    def bound_propagation_last_layer(self):
        upper_bounds = [[1] * self.Layers[0].in_features]
        lower_bounds = [[0] * self.Layers[0].in_features]
        for i in range(len(self.Layers) - 1):
            ub = []
            lb = []
            for j in range(len(self.Layers[i].weight)):
                w = self.Layers[i].weight[j].detach().numpy().copy()
                b = self.Layers[i].bias[j].item()
                u = max(0, b + np.sum(np.maximum(w * upper_bounds[-1], w * lower_bounds[-1])))
                l = max(0, b + np.sum(np.minimum(w * upper_bounds[-1], w * lower_bounds[-1])))
                ub.append(u)
                lb.append(l)

            upper_bounds.append(ub)
            lower_bounds.append(lb)
        return upper_bounds[-1], lower_bounds[-1]

    @staticmethod
    def get_dataloader(X, y, batch_size=5):
        X_pytorch = torch.from_numpy(X)
        train_data = [[X_pytorch[i], y[i]] for i in range(len(X_pytorch))]
        train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_data_loader

    def learn(self, X, y, learning_rate=0.001, batch_size=10, epochs=5,
              loss=nn.BCEWithLogitsLoss(), momentum=0, weight_decay=0.001, indices_val=None,
              transfer_learning_early_stopping=True):
        assert (indices_val is not None)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X_val, y_val = X[indices_val], y[indices_val]
        X_train, y_train = np.delete(X, indices_val, axis=0), np.delete(y, indices_val, axis=0)
        train_data_loader = FFNeuralNetTorch.get_dataloader(X_train, y_train, batch_size=batch_size)

        loss_fn = loss
        opt = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        stop_training = False
        best_val_loss = float('inf')
        count_not_improved = 0
        patience = 50

        for e in range(epochs):
            if stop_training:
                break
            for i, instance in enumerate(train_data_loader):
                inputs, labels = instance
                opt.zero_grad()
                outputs = self.forward(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                opt.step()

            loss_val = loss_fn(self.forward(torch.from_numpy(X_val)), torch.tensor(y_val))
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                count_not_improved = 0
            else:
                count_not_improved += 1

            print('epoch {}, validation loss: {}; training loss: {}; best loss: {}'.format(e + 1, loss_val.item(),
                                                                                           loss_fn(self.forward(
                                                                                               torch.from_numpy(X)),
                                                                                               torch.tensor(
                                                                                                   y)).item(),
                                                                                           best_val_loss))
            if transfer_learning_early_stopping:
                if count_not_improved == patience:
                    stop_training = True
                    break

        print('Finished Training')


class NeuralDeepSaDeBaseLearner(Optimizer):
    """
    This is the based class for DeepSaDe Learner.
    """
    def __init__(self, verbose=False, sizes=None, base_nn=None, mapped_features=None):
        super().__init__(verbose)
        self.runtime = 0
        self.W = None
        self.validation_losses = []
        # self.all_weights = []
        self.current_weight_last_layer = None
        self.K = False
        self.mapped_features = mapped_features
        if self.mapped_features is None:
            self.mapped_features = []
        if base_nn is None:
            self.mirror_nn = FFNeuralNetTorch(sizes=sizes, mapped_features=self.mapped_features)
        else:
            self.mirror_nn = base_nn
        self.loss = None
        self.best_model = None
        self.previous_model = None

    @staticmethod
    def sumproduct(X, W):
        assert (len(W) == len(X) + 1)
        out = W[0]
        for i in range(len(X)):
            out = out + X[i] * W[i + 1]
        return out

    def add_weight_constraints(self, low=-1000, high=1000):
        Fh = []
        for W in self.W:
            for w in W:
                Fh.append(And(w >= low, w <= high))
        self.Hard_Constraints.extend(Fh)

    def add_knowledge_constraints(self, X, y, K=None):
        raise NotImplementedError

    def add_decision_constraints(self, X, y, margins=None, mapped_inputs=None):
        raise NotImplementedError

    def add_gradient_constraints(self, gradients=None, current_weights=None, learning_range=None):
        assert (gradients is not None)
        assert (learning_range is not None)
        assert (current_weights is not None)
        for i in range(len(self.W)):
            for k in range(len(self.W[i])):
                if gradients[i][k] > 0:
                    self.Hard_Constraints.append(
                        And(self.W[i][k] < current_weights[i][k], self.W[i][k] >= current_weights[i][k] -
                            learning_range * gradients[i][k]))
                elif gradients[i][k] < 0:
                    self.Hard_Constraints.append(
                        And(self.W[i][k] > current_weights[i][k], self.W[i][k] <= current_weights[i][k] +
                            learning_range * abs(gradients[i][k])))
                else:
                    self.Hard_Constraints.append(self.W[i][k] == current_weights[i][k])

    @staticmethod
    def make_custom_dataloader(X, y, batch_size=5, shuffle=True):
        raise NotImplementedError

    @staticmethod
    def get_dataloader(X, y, batch_size=5, custom_dataloader=False):
        if not custom_dataloader:
            X_pytorch = torch.from_numpy(X)
            train_data = [[X_pytorch[i], y[i]] for i in range(len(X_pytorch))]
            train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            train_data_loader = \
                NeuralDeepSaDeBaseLearner.make_custom_dataloader(X, y, batch_size=batch_size,
                                                                 shuffle=True)
        return train_data_loader

    def get_margins(self, y, M):
        raise NotImplementedError

    def update_last_layer(self, W):
        self.mirror_nn.Layers[-1].weight.data = \
            torch.tensor(np.array([np.transpose(np.transpose(W)[1:(self.mirror_nn.Layers[-1].in_features + 1)])]),
                         dtype=torch.float32)[0]
        self.mirror_nn.Layers[-1].bias.data = torch.tensor(np.array([np.transpose(np.transpose(W)[0])]), dtype=torch.float32)[0]
        if len(self.mapped_features) > 0:
            self.mirror_nn.mapped_feature_weights.data = torch.tensor(
                np.transpose((np.transpose(W)[(self.mirror_nn.Layers[-1].in_features + 1):])),
                dtype=torch.float32)

    def get_gradients_last_layer(self):
        weight_gradients = list(self.mirror_nn.Layers[-1].weight.grad.detach().numpy().copy())
        bias_gradients = self.mirror_nn.Layers[-1].bias.grad.detach().numpy().copy()
        mapped_feature_gradients = []
        if len(self.mapped_features) > 0:
            mapped_feature_gradients = list(self.mirror_nn.mapped_feature_weights.grad.detach().numpy().copy())
        for j in range(len(weight_gradients)):
            weight_gradients[j] = np.insert(weight_gradients[j], 0, bias_gradients[j].item())
            if len(self.mapped_features) > 0:
                weight_gradients[j] = np.insert(weight_gradients[j], len(weight_gradients[j]),
                                                mapped_feature_gradients[j])
        return np.array(weight_gradients)

    def get_weights_last_layer(self):
        weights = list(self.mirror_nn.Layers[-1].weight.detach().numpy().copy())
        bias = self.mirror_nn.Layers[-1].bias.detach().numpy().copy()
        mapped_feature_weight = []
        if len(self.mapped_features) > 0:
            mapped_feature_weight = list(self.mirror_nn.mapped_feature_weights.detach().numpy().copy())
        for j in range(len(weights)):
            weights[j] = np.insert(weights[j], 0, bias[j].item())
            if len(self.mapped_features) > 0:
                weights[j] = np.insert(weights[j], len(weights[j]), mapped_feature_weight[j])
        return np.array(weights)

    @staticmethod
    def manipulate_gradients_0(gradients):
        return -1 * gradients

    @staticmethod
    def manipulate_gradients_1(gradients):
        new_gradients = []
        for G in gradients:
            new_g = []
            for g in G:
                if random.choice([True, False]):
                    new_g.append(-1 * g)
                else:
                    new_g.append(g)
            new_gradients.append(new_g)
        return np.array(new_gradients)

    @staticmethod
    def get_sign(gradients):
        return np.sign(gradients)

    def find_initial_solution(self, K=None):
        raise NotImplementedError

    def LocalSearchOptimizer(self, K, W, G, learning_rate=None, steps=None):
        raise NotImplementedError

    def LearnBaseModel(self, X, y, learning_rate=0.01, batch_size=10, epochs=10, weight_decay=0.001, indices_val=None,
                       transfer_learning_early_stopping=True):
        self.mirror_nn.learn(X, y, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, loss=self.loss,
                             weight_decay=weight_decay, indices_val=indices_val,
                             transfer_learning_early_stopping=transfer_learning_early_stopping)

    def _learn_dc(self,
                  X, y,
                  M=None,
                  K=None,
                  weight_limit=100,
                  batch_size=10,
                  epochs=5,
                  maximal_step_size=0.001,
                  learning_rate=0.001,
                  waiting_period=10,
                  momentum=0,
                  validation_set_ratio=0.1,
                  custom_dataloader=False,
                  dc_initialization=False,
                  gradient_randomization_strategy=0,
                  transfer_learning=False,
                  transfer_learning_epochs=1000,
                  weight_decay=0,
                  transfer_learning_early_stopping=True
                  ):
        raise NotImplementedError

    def _learn_dc_ls(self,
                     X, y,
                     M=None,
                     K=None,
                     weight_limit=100,
                     batch_size=10,
                     epochs=5,
                     maximal_step_size=0.001,
                     learning_rate=0.001,
                     waiting_period=10,
                     momentum=0,
                     validation_set_ratio=0.1,
                     custom_dataloader=False,
                     dc_initialization=False,
                     gradient_randomization_strategy=0,
                     transfer_learning=False,
                     transfer_learning_epochs=1000,
                     weight_decay=0,
                     maxsmt_optimization=True
                     ):

        start = time.time()
        with open('log.txt', 'a+') as f:
            f.write('network = {}; learning rate = {}; maximal step size = {}; epochs = {}; batch size = {}\n'.format(
                self.mirror_nn.sizes, learning_rate,
                maximal_step_size, epochs, batch_size))

        indices_val = random.sample(range(len(X)), int(len(X) * validation_set_ratio))

        if transfer_learning:
            print('Initiating Self Transfer Learning')
            self.LearnBaseModel(X, y, learning_rate=learning_rate, batch_size=batch_size,
                                epochs=transfer_learning_epochs, weight_decay=0, indices_val=indices_val)
            print('Finished Self Transfer Learning')

        self.solver.set('timeout', waiting_period * 1000)
        M.sort()

        # initialize the z3 weights for the last layer
        self.W = [RealVector('w_{}'.format(t), self.mirror_nn.sizes[-2] + 1 + len(self.mapped_features))
                  for t in range(self.mirror_nn.sizes[-1])]

        # initialize the loss and define the optimized for the neural network
        loss_fn = self.loss
        opt = torch.optim.SGD(self.mirror_nn.parameters(), lr=learning_rate,
                              momentum=momentum, weight_decay=weight_decay)

        # making a validation set
        indices_val = random.sample(range(len(X)), int(len(X) * validation_set_ratio))
        X_val, y_val = X[indices_val], y[indices_val]
        X, y = np.delete(X, indices_val, axis=0), np.delete(y, indices_val, axis=0)

        # initialize gradient and the current value of the validation loss
        current_loss = float('inf')
        train_data_loader = NeuralDeepSaDeBaseLearner.get_dataloader(X, y, batch_size=batch_size,
                                                                     custom_dataloader=custom_dataloader)

        if not dc_initialization:
            if K is not None:
                self.current_weight_last_layer = self.find_initial_solution(K=K)
                if self.current_weight_last_layer is None:
                    return None
                else:
                    self.update_last_layer(self.current_weight_last_layer)

        rev = False
        loss_val = 0
        for e in range(epochs):
            for i, instance in enumerate(train_data_loader):
                # getting a batch of instances
                inputs, labels = instance
                opt.zero_grad()
                outputs = self.mirror_nn(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()

                gradients_last_layer = self.get_gradients_last_layer()

                if rev and len(self.validation_losses) > 0:
                    if gradient_randomization_strategy == 0:
                        gradients_last_layer = NeuralDeepSaDeBaseLearner.manipulate_gradients_0(gradients_last_layer)
                    elif gradient_randomization_strategy == 1:
                        gradients_last_layer = NeuralDeepSaDeBaseLearner.manipulate_gradients_1(gradients_last_layer)
                    elif gradient_randomization_strategy == 2:
                        gradients_last_layer = gradients_last_layer
                    else:
                        raise ValueError('Invalid gradient randomization input')

                # first check if an admissible solution can be found in the negative direction of the gradients
                if self.current_weight_last_layer is not None:
                    W_learnt = self.LocalSearchOptimizer(K, self.current_weight_last_layer, gradients_last_layer,
                                                         learning_rate=learning_rate, steps=5)
                else:
                    W_learnt = None

                if W_learnt is None and (maxsmt_optimization or dc_initialization):
                    print('Initializing SaDe as LocalSearch did not yield a solution.')

                    # extract input to the last layer
                    X_last_layer = self.mirror_nn.input_last_layer(inputs).detach().numpy().copy()

                    # adding domain constraints
                    if K is not None:
                        self.add_knowledge_constraints(X, y, K)

                    # adding decision constraints on the batch
                    margins = self.get_margins(labels.numpy().copy(), M)
                    self.add_decision_constraints(X_last_layer, labels.numpy().copy(), margins=margins,
                                                  mapped_inputs=inputs.T[self.mapped_features].T.numpy().copy())

                    # adding gradient constraints or the weight constraint in the first iteration
                    if self.current_weight_last_layer is None:
                        self.add_weight_constraints(low=-weight_limit, high=weight_limit)
                    else:
                        if maximal_step_size < 0:
                            maximal_step_size_calculated = learning_rate * np.max(np.absolute(gradients_last_layer))
                            print('maximal step size is {} * {} = {}'.format(learning_rate,
                                                                             np.max(np.absolute(gradients_last_layer)),
                                                                             maximal_step_size_calculated))
                        else:
                            maximal_step_size_calculated = maximal_step_size

                        # getting the signs of the gradients instead of the exact value
                        gradients_last_layer = NeuralDeepSaDeBaseLearner.get_sign(gradients_last_layer)

                        self.add_gradient_constraints(gradients_last_layer, self.current_weight_last_layer,
                                                      maximal_step_size_calculated)

                    # MaxSMT solving
                    self.OptimizationFuMalik(length=len(M))

                    if self.out == sat:
                        W_learnt = [
                            [self.solver.model()[u].numerator_as_long() / self.solver.model()[u].denominator_as_long()
                             for u in w] for w in self.W]
                        dc_initialization = False

                if W_learnt is not None:
                    # updating the last layer weights of the mirror neural network
                    self.update_last_layer(W_learnt)

                    # save the best network so far using a validation set
                    loss_val = loss_fn(self.mirror_nn(torch.from_numpy(X_val)), torch.tensor(y_val))
                    print('validation set loss:', loss_val.item())
                    self.validation_losses.append(loss_val.item())
                    if loss_val.item() < current_loss:
                        current_loss = loss_val.item()
                        self.best_model = copy.deepcopy(self.mirror_nn)

                    # making the gradients for the last layer 0, so that they don't get updated with the step method
                    self.mirror_nn.Layers[-1].weight.grad = torch.zeros(
                        self.mirror_nn.Layers[-1].weight.grad.size())
                    self.mirror_nn.Layers[-1].bias.grad = torch.zeros(self.mirror_nn.Layers[-1].bias.grad.size())

                    if len(self.mapped_features) > 0:
                        self.mirror_nn.mapped_feature_weights.grad = \
                            torch.zeros(self.mirror_nn.mapped_feature_weights.grad.size())

                    # update the parameters of the first n-1 layers
                    opt.step()
                    self.current_weight_last_layer = W_learnt
                    rev = False
                else:
                    rev = True

                self.Soft_Constraints = []
                self.Hard_Constraints = []
                self.T = None
                self.out = None
                self.solver.reset()

            with open('log.txt', 'a+') as f:
                f.write('epoch: {}, validation loss: {}, training loss: {}\n'.format(e + 1, loss_val,
                                                                                     loss_fn(self.mirror_nn(
                                                                                         torch.from_numpy(X)),
                                                                                         torch.tensor(y)).item()))
        self.runtime = time.time() - start

    def learn(self,
              X, y,
              M=None,
              K=None,
              weight_limit=100,
              batch_size=10,
              epochs=5,
              maximal_step_size=0.001,
              learning_rate=0.001,
              waiting_period=10,
              momentum=0,
              validation_set_ratio=0.1,
              custom_dataloader=False,
              learn_baseline=False,
              optimization='dc',
              dc_initialization=False,
              gradient_randomization_strategy=0,
              transfer_learning=False,
              transfer_learning_epochs=1000,
              weight_decay=0,
              maxsmt_optimization=True
              ):

        if K is None and learn_baseline:
            start = time.time()
            self.mirror_nn.learn(X, y, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
                                 loss=nn.BCEWithLogitsLoss(), momentum=momentum)
            self.best_model = self.mirror_nn
            print('learned model using stochastic gradient descent!')
            self.runtime = time.time() - start
        else:
            if optimization != 'dc' and optimization != 'ls':
                raise ValueError("optimization = 'dc' for sade optimization, "
                                 "optimization = 'ls' for local search optimization")

            self._learn_dc_ls(X, y,
                              M=M,
                              K=K,
                              weight_limit=weight_limit,
                              batch_size=batch_size,
                              epochs=epochs,
                              maximal_step_size=maximal_step_size,
                              learning_rate=learning_rate,
                              waiting_period=waiting_period, 
                              momentum=momentum,
                              validation_set_ratio=validation_set_ratio,
                              custom_dataloader=custom_dataloader,
                              dc_initialization=dc_initialization,
                              gradient_randomization_strategy=gradient_randomization_strategy,
                              transfer_learning=transfer_learning,
                              transfer_learning_epochs=transfer_learning_epochs,
                              weight_decay=weight_decay,
                              maxsmt_optimization=maxsmt_optimization)
