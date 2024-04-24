import sys
sys.path.insert(0, 'path to src')
from src.Learners import *
import copy


class NeuralMusicClassifier(NeuralSaDeClassifier):
    def __init__(self, verbose=False, sizes=None, mapped_features=None, constraint_num=1, base_nn=None):
        super().__init__(verbose, sizes, mapped_features=mapped_features, problem_type='mc', base_nn=base_nn)
        self.property = constraint_num
    
    def find_solution_SMT(self, K=None, W_prev=None, learning_rate=0.1):
        assert (K is not None)
        _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))
        W = [RealVector('w_{}'.format(t), self.mirror_nn.sizes[-2] + 1 + len(self.mapped_features))
                      for t in range(self.mirror_nn.sizes[-1])]
        upper_bounds, lower_bounds = self.mirror_nn.bound_propagation_last_layer()
        upper_bounds.extend([1])
        lower_bounds.extend([1])
        solver = Solver()
        if W_prev is None:
            W_prev = [[0]*len(W[0])]*len(W)
        
        while True:
            solver.add(And([And([And(W[i][j] >= W_prev[i][j] - learning_rate,
                             W[i][j] <= W_prev[i][j] + learning_rate)
                         for j in range(len(W[i]))]) for i in range(len(W))]))
            
            solver.add(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i],
                                                   _X[i] >= lower_bounds[i]) for i in range(len(_X))]),
                                          And([NeuralSaDeClassifier.sumproduct(_X, W[0]) < 0,
                                               NeuralSaDeClassifier.sumproduct(_X, W[1]) < 0,
                                               NeuralSaDeClassifier.sumproduct(_X, W[2]) < 0,
                                               Or(NeuralSaDeClassifier.sumproduct(_X, W[3]) > 0,
                                                  NeuralSaDeClassifier.sumproduct(_X, W[4]) > 0)
                                              ]))))

            out = solver.check()
            if out == sat:
                print('Found a solution!')
                W_learnt = [[0 if solver.model()[u] is None else 
                         solver.model()[u].numerator_as_long() / solver.model()[u].denominator_as_long()
                         for u in w] for w in W]
                return W_learnt
            else:
                solver.reset()
                learning_rate = 2*learning_rate
    
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
        upper_bounds.extend([1])
        lower_bounds.extend([1])
        
        solver.add(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i],
                                               _X[i] >= lower_bounds[i]) for i in range(len(_X))]),
                                      And([NeuralSaDeClassifier.sumproduct(_X, W[0]) < 0,
                                           NeuralSaDeClassifier.sumproduct(_X, W[1]) < 0,
                                           NeuralSaDeClassifier.sumproduct(_X, W[2]) < 0,
                                           Or(NeuralSaDeClassifier.sumproduct(_X, W[3]) > 0,
                                              NeuralSaDeClassifier.sumproduct(_X, W[4]) > 0)
                                          ]))))
        
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
        upper_bounds.extend([1])
        lower_bounds.extend([1])
        
        for s in range(steps):
            W_candidate = [[W_prev[i][j] - G[i][j] * learning_rate * (steps - s) / (steps)
                            for j in range(len(G[i]))]
                           for i in range(len(G))]
            if K is not None:
                _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))                
                solver.add(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i],
                                               _X[i] >= lower_bounds[i]) for i in range(len(_X))]),
                                      And([NeuralSaDeClassifier.sumproduct(_X, W_candidate[0]) < 0,
                                           NeuralSaDeClassifier.sumproduct(_X, W_candidate[1]) < 0,
                                           NeuralSaDeClassifier.sumproduct(_X, W_candidate[2]) < 0,
                                           Or(NeuralSaDeClassifier.sumproduct(_X, W_candidate[3]) > 0,
                                              NeuralSaDeClassifier.sumproduct(_X, W_candidate[4]) > 0)
                                          ]))))

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
            upper_bounds, lower_bounds = self.mirror_nn.bound_propagation_last_layer()
            upper_bounds.extend([1])
            lower_bounds.extend([1])
            
            _X = RealVector('x', self.mirror_nn.Layers[-1].in_features + len(self.mapped_features))
            
            print('Adding Knowledge Constraints')
            print('upper bounds:', upper_bounds)
            
            self.Hard_Constraints.append(ForAll(_X, Implies(And([And(_X[i] <= upper_bounds[i],
                                                                     _X[i] >= lower_bounds[i])
                                                                 for i in range(len(_X))]),
                                                            And([
                                                                NeuralSaDeClassifier.sumproduct(
                                                                    _X, self.W[0]) < 0,
                                                                NeuralSaDeClassifier.sumproduct(
                                                                    _X, self.W[1]) < 0,
                                                                NeuralSaDeClassifier.sumproduct(
                                                                    _X, self.W[2]) < 0,
                                                                Or(NeuralSaDeClassifier.sumproduct(
                                                                    _X, self.W[3]) > 0,
                                                                   NeuralSaDeClassifier.sumproduct(
                                                                       _X, self.W[4]) > 0
                                                                   )
                                                            ]))))
                
    
class MusicRegularizedModel(FFNeuralNetTorch):
    def __init__(self, sizes=None, mapped_features=None, constraint_num=1):
        super().__init__(sizes, mapped_features)
        self.property = constraint_num
        self.best_model = {}
    
    @staticmethod
    def regularized_loss(pred, target, inputs, alpha=100, K=None, loss_type='sl'):
        regularization_term = torch.tensor(0)
        if K is not None:
            I = torch.where(inputs.T[12] == 1)[0]
            if len(I) > 0:
                if loss_type == 'sl':
                    regularization_term = - torch.sum(torch.log((torch.sigmoid(pred).T[3] + torch.sigmoid(pred).T[4])*(1 - torch.sigmoid(pred).T[0])*(1 - torch.sigmoid(pred).T[1])*(1 - torch.sigmoid(pred).T[2]))[I])/len(inputs)
                elif loss_type == 'sbr':
                    regularization_term = 1 - torch.sum(torch.max(torch.sigmoid(pred.T[3:].T), dim=1)[0][I])/len(inputs)
        l = nn.BCEWithLogitsLoss()
        return l(pred, target) * (1 - alpha) + regularization_term * alpha
    
    def learn(self, X, y, learning_rate=0.001, batch_size=10, epochs=5, K=None,
              loss=None, momentum=0, alpha=10, loss_type='sl', validation_set_ratio=0.1):
        assert (loss is None)
        sys.setrecursionlimit(10000)
        start = time.time()
        indices_val = random.sample(range(len(X)), int(len(X) * validation_set_ratio))

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        X_val, y_val = X[indices_val], y[indices_val]
        X_train, y_train = np.delete(X, indices_val, axis=0), np.delete(y, indices_val, axis=0)

        train_data_loader = FFNeuralNetTorch.get_dataloader(X_train, y_train, batch_size=batch_size)
        loss_fn = MusicRegularizedModel.regularized_loss
                
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
                loss = loss_fn(outputs, labels, inputs, alpha, K, loss_type)
                loss.backward()
                opt.step()
            
            loss_val = loss_fn(self.forward(torch.from_numpy(X_val)), torch.tensor(y_val), torch.from_numpy(X_val), alpha, K, loss_type)
            
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
                                                                                                  torch.from_numpy(X_train), alpha, K, 
                                                                                                   loss_type).item(),
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
            pred.append(np.argmax(p))
        pred_d = np.zeros((len(pred), len(score[0])))
        pred_d[np.arange(len(pred)), pred] = 1
        return pred_d
        
    @staticmethod
    def accuracy_score(pred, y):
        s = 0
        for i in range(len(y)):
            s += np.dot(pred[i], y[i]) / sum(y[i])
        return s / len(y)        