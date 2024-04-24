import sys
sys.path.insert(0, 'path to src')

from src.Learners import *


class NeuralCombinedMnistClassifier(NeuralSaDeClassifier):
    def __init__(self, verbose=False, sizes=None, mapped_features=None, base_nn=None):
        super().__init__(verbose, sizes, mapped_features=mapped_features, problem_type='ml', base_nn=base_nn)
    
    def find_initial_solution(self, K=None, bounds = None, low=-1, high=1):
        assert (K is not None)
        solver = Solver()
        W = [RealVector('w_{}'.format(t), self.mirror_nn.sizes[-2] + 1 + len(self.mapped_features))
                  for t in range(self.mirror_nn.sizes[-1])]
        
        for Weight in W:
            for w in Weight:
                solver.add(And(w >= low, w <= high))
        
        _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))
        
        upper_bounds, lower_bounds = self.mirror_nn.bound_propagation_last_layer()
        print(upper_bounds, lower_bounds)
        solver.add(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i], _X[i] >= lower_bounds[i]) for i in range(len(_X))]), 
                       Sum([If(NeuralDeepSaDeBaseLearner.sumproduct(_X, W[j]) > 0, j, 0) for j in range(len(W))]) > 10))
            )
        
        
        out = solver.check()
        print(out)
        if out == sat:
            print('Found Initial Solution!')
            W_learnt = [[0 if solver.model()[u] is None else 
                         solver.model()[u].numerator_as_long() / solver.model()[u].denominator_as_long()
                         for u in w] for w in W]
            return W_learnt
        else:
            print("could not find the initial solution")
    
    def LocalSearchOptimizer(self, K=None, W_prev=None, G=None, steps=5, learning_rate=0.1):
        solver = Solver()
        upper_bounds, lower_bounds = self.mirror_nn.bound_propagation_last_layer()
        
        for s in range(steps):
            W_candidate = [[W_prev[i][j] - G[i][j] * learning_rate * (steps - s) / (steps)
                            for j in range(len(G[i]))]
                           for i in range(len(G))]
            if K is not None:
                _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))
                solver.add(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i], _X[i] >= lower_bounds[i]) 
                                                   for i in range(len(_X))]),
                                                                       Sum([If(NeuralDeepSaDeBaseLearner.sumproduct(_X, W_candidate[j]) > 0, j, 0) 
                                            for j in range(len(W_candidate))]) > 10)
                                                          )
                                        )
                
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
            print(upper_bounds)
            _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))
            
            self.Hard_Constraints.append(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i], _X[i] >= lower_bounds[i]) 
                                                   for i in range(len(_X))]),
                                                                       Sum([If(NeuralDeepSaDeBaseLearner.sumproduct(_X, self.W[j]) > 0, j, 0) 
                                            for j in range(len(self.W))]) > 10)
                                                          )
                                        )

            
class MNISTBaselineModel(FFNeuralNetTorch):
    def __init__(self, sizes=None, mapped_features=None):
        super().__init__(sizes, mapped_features)
        self.best_model = {}
        
    def learn(self, X, y, learning_rate=0.001, batch_size=10, epochs=5, K=None,
              loss=nn.BCEWithLogitsLoss(), momentum=0, validation_set_ratio=0.1):
        start = time.time()
        indices_val = random.sample(range(len(X)), int(len(X) * validation_set_ratio))
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X_val, y_val = X[indices_val], y[indices_val]
        X, y = np.delete(X, indices_val, axis=0), np.delete(y, indices_val, axis=0)

        train_data_loader = FFNeuralNetTorch.get_dataloader(X, y, batch_size=batch_size)
        loss_fn = loss
                
        opt = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        
        stop_training = False
        count_not_improve = 0
        patience = 50
        current_loss = float('inf')
        
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
                
            loss_val = loss_fn(self.forward(torch.tensor(X_val)), torch.tensor(y_val))    
            
            if loss_val.item() < current_loss:
                current_loss = loss_val.item()
                for name, param in self.named_parameters():
                    self.best_model[name] = copy.deepcopy(param.data)
                count_not_improve = 0
            else:
                count_not_improve += 1
                    
            print('epoch {}, validation loss: {}; training loss: {}; best loss: {}'.format(e+1, loss_val.item(), 
                                                                            loss_fn(self.forward(torch.from_numpy(X)), torch.tensor(y)).item(),
                                                                                          current_loss))

            if count_not_improve == patience:
                stop_training = True
                break
                
        self.runtime = time.time() - start               
        for name, param in self.named_parameters():
            param.data = self.best_model[name]
            
        print('Finished Training')
        
    def predict(self, X):
        X_torch = np.array(X)
        X_torch = X_torch.astype(np.float32)
        X_torch = torch.tensor(X_torch)
        score = self.forward(X_torch).detach().numpy()
        pred = []
        for p in score:
            pred.append([1 if j > 0 else 0 for j in p])
        return np.array(pred)

    @staticmethod
    def accuracy_score(pred, y):
        s = 0
        for i in range(len(y)):
            s += np.dot(pred[i], y[i]) / len(set.union(set(np.where(y[i] == 1)[0]), set(np.where(pred[i] == np.float32(1))[0])))
        return s / len(y)