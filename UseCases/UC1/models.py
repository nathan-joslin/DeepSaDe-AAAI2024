import sys
sys.path.insert(0, 'path to src')
from Learners import *


class SaDeExpenseRegressor(NeuralSaDeRegressor):
    def __init__(self, verbose=False, sizes=None, base_nn=None, mapped_features=None):
        super().__init__(verbose, sizes, base_nn, mapped_features)
        
    def find_initial_solution(self, K=None, bounds = None, low=-0.1, high=-0.01):
        assert (K is not None)

        solver = Solver()
        W = [RealVector('w_{}'.format(t), self.mirror_nn.sizes[-2] + 1 + len(self.mapped_features))
                  for t in range(self.mirror_nn.sizes[-1])]
        
        for Weight in W:
            for w in Weight:
                solver.add(And(w >= low, w <= high))
        
        _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))
        
        upper_bounds, lower_bounds = self.mirror_nn.bound_propagation_last_layer()
        for f in self.mapped_features:
            upper_bounds.append(1)
            lower_bounds.append(0)
                
        solver.add(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i], _X[i] >= lower_bounds[i]) 
                                                   for i in range(len(_X))]),
                                              And(NeuralDeepSaDeBaseLearner.sumproduct(_X, W[1]) <= 0.05 * (
                                                                            _X[-1] - K.min_[-1]) / K.scale_[-1],
                                                                Sum([NeuralDeepSaDeBaseLearner.sumproduct(_X, W[j]) for
                                                                     j in range(len(W))]) <= (
                                                                            _X[-1] - K.min_[-1]) / K.scale_[-1]
                                                                )
                                     )
                         )
                  )  

        out = solver.check()
        if out == sat:
            print('Found Initial Solution!')
            W_learnt = [[0 if solver.model()[u] is None else 
                         solver.model()[u].numerator_as_long() / solver.model()[u].denominator_as_long()
                         for u in w] for w in W]
            return W_learnt
        else:
            print(out)
            print("could not find the initial solution")
            
    def LocalSearchOptimizer(self, K=None, W_prev=None, G=None, steps=5, learning_rate=0.1):
        solver = Solver()
        upper_bounds, lower_bounds = self.mirror_nn.bound_propagation_last_layer()
        for f in self.mapped_features:
            upper_bounds.append(1)
            lower_bounds.append(0)
         
        for s in range(steps):
            W_candidate = [[W_prev[i][j] - G[i][j] * learning_rate * (steps - s) / (steps)
                            for j in range(len(G[i]))]
                           for i in range(len(G))]
            if K is not None:
                _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))   
                solver.add(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i],
                                                                     _X[i] >= lower_bounds[i])
                                                                 for i in range(len(_X))]),
                                                            And(NeuralDeepSaDeBaseLearner.sumproduct(_X,
                                                                                                 W_candidate[1]) <= 0.05 * (
                                                                            _X[-1] - K.min_[-1]) / K.scale_[-1],
                                                                Sum([NeuralDeepSaDeBaseLearner.sumproduct(_X, W_candidate[j]) for
                                                                     j in range(len(W_candidate))]) <= (
                                                                            _X[-1] - K.min_[-1]) / K.scale_[-1]
                                                                )
                                                            )))
                
            out = solver.check()
            if out == sat:
                print('Found Solution!')
                print(s)
                return W_candidate
            else:
                solver.reset()
        print("No Solution Found!")
        return None
            
    def add_knowledge_constraints(self, X, y, K=None):
        if K is not None:
            print('adding knowledge constraints')
            upper_bounds, lower_bounds = self.mirror_nn.bound_propagation_last_layer()
            _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))
            for f in self.mapped_features:
                upper_bounds.append(1)
                lower_bounds.append(0)
            print(lower_bounds)
            print(upper_bounds)
            self.Hard_Constraints.append(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i],
                                                                     _X[i] >= lower_bounds[i])
                                                                 for i in range(len(_X))]),
                                                            And(NeuralDeepSaDeBaseLearner.sumproduct(_X,
                                                                                                 self.W[1]) <= 0.05 * (
                                                                            _X[-1] - K.min_[-1]) / K.scale_[-1],
                                                                Sum([NeuralDeepSaDeBaseLearner.sumproduct(_X, self.W[j]) for
                                                                     j in range(len(self.W))]) <= (
                                                                            _X[-1] - K.min_[-1]) / K.scale_[-1]
                                                                )
                                                            )))

class ExpenseRegularizedModel(FFNeuralNetTorch):
    def __init__(self, sizes=None, mapped_features=None):
        super().__init__(sizes, mapped_features)
        self.best_model = {}
        
    @staticmethod
    def regularized_loss(pred, target, inputs, alpha=100, beta=100, K=None):
        regularization_term_1 = 0
        regularization_term_2 = 0
        if K is not None:
            regularization_term_1 = torch.sum(torch.max(torch.tensor(0, dtype=torch.float32, requires_grad=True),
                                                        torch.sum(pred, dim=1) - (inputs.T[12] - K.min_[-1]) /
                                                        K.scale_[-1]))
            regularization_term_2 = torch.sum(torch.max(torch.tensor(0, dtype=torch.float32, requires_grad=True),
                                                        pred.T[1] - 0.05 * (inputs.T[12] - K.min_[-1]) / K.scale_[-1]))
        return (1 - alpha - beta) * torch.sum((pred - target) ** 2) / (len(pred) * len(pred.T)) + alpha * (
                1 / len(inputs)) * regularization_term_1 + beta * (1 / len(inputs)) * regularization_term_2

    def learn(self, X, y, learning_rate=0.001, batch_size=10, epochs=5, K=None,
              loss=None, momentum=0, alpha=10, beta=10, validation_set_ratio=0.1):
        assert (loss is None)
        start = time.time()

        indices_val = random.sample(range(len(X)), int(len(X) * validation_set_ratio))

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X_val, y_val = X[indices_val], y[indices_val]
        X_train, y_train = np.delete(X, indices_val, axis=0), np.delete(y, indices_val, axis=0)

        train_data_loader = FFNeuralNetTorch.get_dataloader(X_train, y_train, batch_size=batch_size)

        loss_fn = ExpenseRegularizedModel.regularized_loss
        opt = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        stop_training = False
        count_not_improve = 0
        patience = 50
        current_loss = float('inf')
        
        for e in range(epochs):
            for i, instance in enumerate(train_data_loader):
                inputs, labels = instance
                opt.zero_grad()
                outputs = self.forward(inputs)
                loss = loss_fn(outputs, labels, inputs, alpha, beta, K)
                loss.backward()
                opt.step()
                # print statistics
            
            loss_val = loss_fn(self.forward(torch.from_numpy(X_val)), torch.tensor(y_val), torch.from_numpy(X_val), alpha, beta, K)
            
            if loss_val.item() < current_loss:
                current_loss = loss_val.item()
                for name, param in self.named_parameters():
                    self.best_model[name] = copy.deepcopy(param.data)
                count_not_improve = 0
            else:
                count_not_improve += 1
                                    
            print('epoch {}, validation loss: {}; training loss: {}; best loss: {}'.format(e + 1, loss_val.item(),
                                                                                           loss_fn(self.forward(
                                                                                               torch.from_numpy(X_train)),
                                                                                               torch.tensor(
                                                                                                   y_train), 
                                                                                                  torch.from_numpy(X_train), alpha, beta, K).item(),
                                                                                           current_loss))
            if count_not_improve == patience:
                stop_training = True
                break
           
        self.runtime = time.time() - start
        for name, param in self.named_parameters():
            param.data = self.best_model[name]
        print('Finished Training')
