from src.BaseLearners import *


class NeuralSaDeClassifier(NeuralDeepSaDeBaseLearner):
    def __init__(self, verbose=False, sizes=None, problem_type='ml', base_nn=None, mapped_features=None):
        super().__init__(verbose, sizes, base_nn, mapped_features)
        self.loss = nn.BCEWithLogitsLoss()
        self.problem_type = problem_type

    def add_knowledge_constraints(self, X, y, K=None):
        pass  # to be implemented for each use case

    def add_decision_constraints(self, X, y, margins=None, mapped_inputs=None):
        if mapped_inputs is None:
            mapped_inputs = []
        Fs = []
        for i in range(len(X)):
            x = np.append(X[i], mapped_inputs[i])
            for j in range(len(y[i])):
                margins_j = margins[j]
                if y[i][j] == 1:
                    for m in margins_j:
                        Fs.append(NeuralDeepSaDeBaseLearner.sumproduct(x, self.W[j]) > m)
                else:
                    for m in margins_j:
                        Fs.append(NeuralDeepSaDeBaseLearner.sumproduct(x, self.W[j]) < -m)

        self.Soft_Constraints.extend(Fs)

    def get_margins(self, y, M):
        return [[m for m in M] for l in np.transpose(y)]

    def predict(self, X):
        if self.problem_type == 'mc':
            return self.predict_multiclass(X)
        else:
            return self.predict_multilabel(X)

    def predict_multiclass(self, X):
        X_torch = np.array(X)
        X_torch = X_torch.astype(np.float32)
        X_torch = torch.tensor(X_torch)
        score = self.best_model.forward(X_torch).detach().numpy()
        pred = []
        for p in score:
            pred.append(np.argmax(p))
        pred_d = np.zeros((len(pred), len(score[0])))
        pred_d[np.arange(len(pred)), pred] = 1
        return np.array(pred_d)

    def predict_multilabel(self, X):
        X_torch = np.array(X)
        X_torch = X_torch.astype(np.float32)
        X_torch = torch.tensor(X_torch)
        score = self.best_model.forward(X_torch).detach().numpy()
        pred = []
        for p in score:
            pred.append([1 if j > 0 else 0 for j in p])
        return np.array(pred)

    def accuracy_score(self, pred, y):
        if self.problem_type == 'mc':
            return NeuralSaDeClassifier.accuracy_score_mc(pred, y)
        else:
            return NeuralSaDeClassifier.accuracy_score_ml(pred, y)
        
    @staticmethod
    def accuracy_score_mc(pred, y):
        s = 0
        for i in range(len(y)):
            s += np.dot(pred[i], y[i]) / (sum(y[i]))
        return s / len(y)
    
    @staticmethod
    def accuracy_score_ml(pred, y):
        s = 0
        for i in range(len(y)):
            s += np.dot(pred[i], y[i]) / len(set.union(set(np.where(y[i] == 1)[0]), set(np.where(pred[i] == np.float32(1))[0])))
        return s / len(y)


class NeuralSaDeRegressor(NeuralDeepSaDeBaseLearner):
    def __init__(self, verbose=False, sizes=None, base_nn=None, mapped_features=None):
        super().__init__(verbose, sizes, base_nn, mapped_features)
        self.loss = nn.MSELoss()

    def add_knowledge_constraints(self, X, y, K=None):
        pass  # to be implemented for each use case

    def add_decision_constraints(self, X, y, margins=None, mapped_inputs=None):
        if mapped_inputs is None:
            mapped_inputs = []
        Fs = []
        for i in range(len(X)):
            x = np.append(X[i], mapped_inputs[i])

            for j in range(len(y[i])):
                margins_j = margins[j]
                for m in margins_j:
                    Fs.append(And(NeuralDeepSaDeBaseLearner.sumproduct(x, self.W[j]) >= y[i][j] - m,
                                  NeuralDeepSaDeBaseLearner.sumproduct(x, self.W[j]) <= y[i][j] + m))

        self.Soft_Constraints.extend(Fs)

    def get_margins(self, y, M):
        return [[max(labels) * m for m in M] for labels in np.transpose(y)]

    def predict(self, X):
        score = self.best_model.forward(torch.tensor(np.array(X, dtype=np.float32))).detach().numpy()
        return score

