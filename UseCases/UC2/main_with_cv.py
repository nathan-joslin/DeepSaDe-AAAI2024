from models import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scoop import futures
import argparse
import os
from scipy.io import arff
import pandas as pd
import pickle

MAXSMT_OPTIMIZATION = True
if MAXSMT_OPTIMIZATION:
    SADE_EPOCHS = 50
else:
    SADE_EPOCHS = 400
    
BASELINE_EPOCHS = 400
LEARNING_RATE = 0.1
MAXIMAL_STEP_SIZE = 0.1
SADE_BATCH_SIZE = 5
BASELINE_BATCH_SIZE = 5
SADE_MOMENTUM = 0
BASELINE_MOMENTUM = 0
SADE_WAITING_PERIOD = 5
SADE_MARGINS = [0, 1, 2]
SADE_INITIAL_WEIGHT = 1
OUTER_FOLDS = 5
INNER_FOLDS = 5
NETWORK = [13, 50, 30, 10, 2]
FOLDER_NAME = None
DEEPSADE_APPROACH = 'ls'
DC_INITIALIZATION = False
GRADIENT_RANDOMIZATION_STRATEGY = 1
MAPPED_FEATURES = [0, 4]
TRANSFER_LEARNING = False
VALIDATION_SET_RATIO = 0.1
WEIGHT_DECAY=0
BASELINE_APPROACH='sbr'

# python -m scoop -vv -n 5 main_with_cv.py --baseline 0 --deepsade 1 --seed 1

def get_violations(pred, X, K):
    scaled_income = K.scale_[0] * (5000 - K.data_min_[0])
    indices = np.where((X.T[0] >= 0) & (X.T[0] <= scaled_income) & 
                       (X.T[4] == 0))[0]
    return sum(pred[indices].T[1])

def accuracy_score(pred, y):
    s = 0
    for i in range(len(y)):
        s += np.dot(pred[i], y[i]) / sum(y[i])
    return s / len(y)

def calculate_corrected_accuracy(X, pred, y, K, corrected=True):
    if corrected:
        scaled_income = K.scale_[0] * (5000 - K.data_min_[0])
        indices_to_remove = np.where((X.T[0] >= 0) & (X.T[0] <= scaled_income) & 
                                     (X.T[4] == 0) & (y.T[1] == 1))[0]
        indices = list(set(range(len(X))) - set(indices_to_remove))
        return accuracy_score(pred[indices], y[indices])
    else:
        return accuracy_score(pred, y)
    
def single_baseline_cv(X, y, train_idx=None, test_idx=None, K=None, alpha=0, fold_index=0):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    net = LoanRegularizedModel(sizes=NETWORK)
    net.learn(X_train, y_train, learning_rate=LEARNING_RATE, batch_size=BASELINE_BATCH_SIZE, epochs=BASELINE_EPOCHS,
              momentum=BASELINE_MOMENTUM, K=K, alpha=alpha, validation_set_ratio=VALIDATION_SET_RATIO, loss_type=BASELINE_APPROACH)

    pred_train = net.predict(X_train)
    pred_test = net.predict(X_test)

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        accuracy_train_corrected = calculate_corrected_accuracy(X_train, pred_train, y_train, K)
        accuracy_test_corrected = calculate_corrected_accuracy(X_test, pred_test, y_test, K)

        accuracy_train = calculate_corrected_accuracy(X_train, pred_train, y_train, K, corrected=False)
        accuracy_test = calculate_corrected_accuracy(X_test, pred_test, y_test, K, corrected=False)

        total_violations_train = get_violations(pred_train, X_train, K)
        total_violations_test = get_violations(pred_test, X_test, K)

        return [[fold_index, alpha, accuracy_train_corrected, accuracy_test_corrected, accuracy_train, accuracy_test,
                 total_violations_train, total_violations_test]]


def get_alpha(X, y, K=None):
    kf_CV = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=random.randint(0, 1000))
    split_data_CV = list(kf_CV.split(X))
    list_X = []
    list_y = []
    list_train_idx = []
    list_test_idx = []
    list_K = []
    list_index = []
    list_alphas = []
    list_betas = []
    fold_index = 1
    for train_index, test_index in split_data_CV:
        for alpha in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            list_X.append(X)
            list_y.append(y)
            list_train_idx.append(train_index)
            list_test_idx.append(test_index)
            list_K.append(K)
            list_index.append(fold_index)
            list_alphas.append(alpha)
        fold_index += 1

    all_baseline_output = []
    baseline_results = list(futures.map(single_baseline_cv, list_X, list_y, list_train_idx, list_test_idx, list_K,
                                        list_alphas, list_index))

    for out in baseline_results:
        if out is not None:
            all_baseline_output.extend(out)

    all_baseline_output = pd.DataFrame(all_baseline_output)
    all_baseline_output.columns = ['fold', 'alpha', 'accuracy train corrected', 'accuracy test corrected',
                                   'accuracy train', 'accuracy test', 'violations train', 'violations test']

    processed_output_baseline = []
    for a in np.unique(all_baseline_output['alpha']):
        sub = all_baseline_output[(all_baseline_output['alpha'] == a)]
        accuracy_mean = np.mean(sub['accuracy test corrected'])
        accuracy_std = np.std(sub['accuracy test corrected'])

        violations_mean = np.mean(sub['violations test'])
        violations_std = np.std(sub['violations test'])

        processed_output_baseline.append([a, accuracy_mean, accuracy_std, violations_mean, violations_std])
    
    processed_output_baseline = pd.DataFrame(processed_output_baseline)
    processed_output_baseline.columns = ['alpha', 'accuracy mean', 'accuracy std', 'violations mean', 'violations std']
    return processed_output_baseline[processed_output_baseline['violations mean'] ==
                                     min(processed_output_baseline['violations mean'])].sort_values(by='accuracy mean')[
        ['alpha']].values[0]


def single_baseline_run(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, folder_name='0'):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    a = get_alpha(X_train, y_train, K=K)[0]
    net = LoanRegularizedModel(sizes=NETWORK)
    net.learn(X_train, y_train, learning_rate=LEARNING_RATE, batch_size=BASELINE_BATCH_SIZE, epochs=BASELINE_EPOCHS,
              momentum=BASELINE_MOMENTUM, K=K, alpha=a, validation_set_ratio=VALIDATION_SET_RATIO, loss_type=BASELINE_APPROACH)

    pred_train = net.predict(X_train)
    pred_test = net.predict(X_test)

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        accuracy_train_corrected = calculate_corrected_accuracy(X_train, pred_train, y_train, K)
        accuracy_test_corrected = calculate_corrected_accuracy(X_test, pred_test, y_test, K)

        accuracy_train = calculate_corrected_accuracy(X_train, pred_train, y_train, K, corrected=False)
        accuracy_test = calculate_corrected_accuracy(X_test, pred_test, y_test, K, corrected=False)

        total_violations_train = get_violations(pred_train, X_train, K)
        total_violations_test = get_violations(pred_test, X_test, K)
        
        print('this:', total_violations_train, total_violations_test)
        torch.save(net, folder_name + '/baseline_model_fold_{}_{}.pt'.format(fold_index, BASELINE_APPROACH))

        return [[fold_index, a, accuracy_train_corrected, accuracy_test_corrected, accuracy_train, accuracy_test,
                 total_violations_train, total_violations_test, net.runtime]]


def single_sade_run(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, folder_name='0'):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    while True:
        model = NeuraLoanClassifier(verbose=False, sizes=NETWORK, mapped_features=MAPPED_FEATURES)
        model.learn(X_train, y_train,
                    M=SADE_MARGINS,
                    K=K,
                    weight_limit=SADE_INITIAL_WEIGHT,
                    batch_size=SADE_BATCH_SIZE,
                    epochs=SADE_EPOCHS,
                    maximal_step_size=MAXIMAL_STEP_SIZE,
                    learning_rate=LEARNING_RATE,
                    waiting_period=SADE_WAITING_PERIOD,
                    momentum=SADE_MOMENTUM,
                    optimization=DEEPSADE_APPROACH,
                    dc_initialization=DC_INITIALIZATION,
                    gradient_randomization_strategy=GRADIENT_RANDOMIZATION_STRATEGY,
                    transfer_learning=TRANSFER_LEARNING,
                    weight_decay=WEIGHT_DECAY,
                    transfer_learning_epochs=BASELINE_EPOCHS,
                    validation_set_ratio=VALIDATION_SET_RATIO, maxsmt_optimization=MAXSMT_OPTIMIZATION
                    )

        if len(model.validation_losses) > 30:
            break

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        accuracy_train_corrected = calculate_corrected_accuracy(X_train, pred_train, y_train, K)
        accuracy_test_corrected = calculate_corrected_accuracy(X_test, pred_test, y_test, K)

        accuracy_train = calculate_corrected_accuracy(X_train, pred_train, y_train, K, corrected=False)
        accuracy_test = calculate_corrected_accuracy(X_test, pred_test, y_test, K, corrected=False)

        total_violations_train = get_violations(pred_train, X_train, K)
        total_violations_test = get_violations(pred_test, X_test, K)

        if MAXSMT_OPTIMIZATION:
            torch.save(model.best_model, folder_name + '/deepsade_model_fold_{}.pt'.format(fold_index))
        else:
            torch.save(model.best_model, folder_name + '/deepsade_l_model_fold_{}.pt'.format(fold_index))
            
        return [[fold_index, get_violations(y_train, X_train, K), 
                 accuracy_train_corrected, accuracy_test_corrected, 
                 accuracy_train, accuracy_test,
                 total_violations_train, total_violations_test, model.runtime]], model.validation_losses

def data_processing(data):
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    categorical_features = list(data.select_dtypes(include='object').columns)
    categorical_features = list(set(categorical_features))
    numerical_features = [c for c in data.columns if c not in categorical_features]
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    for _c in categorical_features:
        data[_c] = pd.Categorical(data[_c])
    df_transformed = pd.get_dummies(data, drop_first=True)
    return df_transformed, scaler

def add_violations(y, X, K, count=20):
    scaled_income = K.scale_[0] * (5000 - K.data_min_[0])
    indices = list(np.where((X.T[0] >= 0) & (X.T[0] <= scaled_income) & 
                       (X.T[4] == 0))[0])
    instances_to_change = random.sample(indices, count)
    for i in instances_to_change:
        new_y = [0, 1]
        y[i] = np.array(new_y)
        
def save_config(folder):
    with open('{}/config.txt'.format(folder), 'a+') as f:
        f.write(' SADE_EPOCHS = {}\n BASELINE_EPOCHS = {}\n LEARNING_RATE = {}\n MAXIMAL_STEP_SIZE = {}\n SADE_BATCH_SIZE = {}\n BASELINE_BATCH_SIZE = {}\n SADE_MOMENTUM = {}\n BASELINE_MOMENTUM = {}\n SADE_WAITING_PERIOD = {}\n SADE_MARGINS = {}\n SADE_INITIAL_WEIGHT = {}\n OUTER_FOLDS = {}\n INNER_FOLDS = {}\n NETWORK = {}\n DEEPSADE_APPROACH = {}\n VALIDATION_SET_RATIO = {}\n DC_INITIALIZATION = {}\n GRADIENT_RANDOMIZATION_STRATEGY = {}\n MAPPED_FEATURES = {}\n TRANSFER_LEARNING = {}\n WEIGHT_DECAY = {}\n BASELINE_APPROACH = {}\n MAXSMT_OPTIMIZATION = {}\n'.format(
            SADE_EPOCHS, 
            BASELINE_EPOCHS, 
            LEARNING_RATE, 
            MAXIMAL_STEP_SIZE, 
            SADE_BATCH_SIZE, 
            BASELINE_BATCH_SIZE, 
            SADE_MOMENTUM,
            BASELINE_MOMENTUM,
            SADE_WAITING_PERIOD, 
            SADE_MARGINS,
            SADE_INITIAL_WEIGHT,
            OUTER_FOLDS,
            INNER_FOLDS,
            NETWORK,
            DEEPSADE_APPROACH,
            VALIDATION_SET_RATIO,
            DC_INITIALIZATION,
            GRADIENT_RANDOMIZATION_STRATEGY,
            MAPPED_FEATURES,
            TRANSFER_LEARNING,
            WEIGHT_DECAY,
            BASELINE_APPROACH, MAXSMT_OPTIMIZATION))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, help="1: baseline model is trained; 0: baseline model is not trained",
                        required=True)
    parser.add_argument("--deepsade", type=int, help="1: deepsade model is trained; 0: deepsade model is not trained",
                        required=True)
    parser.add_argument("--seed", type=int, help="random seed of the experiment",
                        required=True)

    args = parser.parse_args()
    do_baseline = [True if args.baseline == 1 else False][0]
    do_deepsade = [True if args.deepsade == 1 else False][0]
    seed = args.seed

    FOLDER_NAME = str(seed)
    if not os.path.exists('{}'.format(FOLDER_NAME)):
        os.makedirs('{}'.format(FOLDER_NAME))

    start = time.time()
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    save_config(FOLDER_NAME)

    dataset = 'loan_data_set.csv'
    dataset = pd.read_csv(dataset).iloc[:, 1:]
    dataset, scaler = data_processing(dataset)
    y = np.array(dataset.iloc[:, -1], dtype=np.float32)
    X = np.array(dataset.iloc[:, :-1], dtype=np.float32)
    y = np.array(pd.get_dummies(y), dtype=np.float32)
    
    add_violations(y, X, scaler, 40)
    y = np.array(y, dtype=np.float32)
    
    # number of folds for cross-validation
    kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=random.randint(0, 1000))
    split_data = list(kf.split(X))

    list_X = []
    list_y = []
    list_train_idx = []
    list_test_idx = []
    list_K = []
    list_index = []
    list_folders = []
    
    f = 1

    for train_index, test_index in split_data:
        list_X.append(X)
        list_y.append(y)
        list_train_idx.append(train_index)
        list_test_idx.append(test_index)
        list_K.append(scaler)
        list_index.append(f)
        list_folders.append(FOLDER_NAME)
        f = f + 1

    if do_baseline:
        all_baseline_output = []
        baseline_results = list(futures.map(single_baseline_run, list_X, list_y, list_train_idx, list_test_idx,
                                            list_K, list_index, list_folders))

        for out in baseline_results:
            if out is not None:
                all_baseline_output.extend(out)

        all_baseline_output = pd.DataFrame(all_baseline_output)
        all_baseline_output.columns = ['fold', 'alpha', 'accuracy train corrected', 'accuracy test corrected',
                                       'accuracy train', 'accuracy test', 'violations train', 'violations test', 'runtime']

        all_baseline_output.to_csv(FOLDER_NAME + '/baseline_output_{}.csv'.format(BASELINE_APPROACH))

    if do_deepsade:
        all_deepsade_output = []
        all_val_losses = []

        deepsade_results = list(futures.map(single_sade_run, list_X, list_y, list_train_idx, list_test_idx,
                                            list_K, list_index, list_folders))

        for out in deepsade_results:
            if out is not None:
                all_deepsade_output.extend(out[0])
                all_val_losses.append(out[1])

        all_deepsade_output = pd.DataFrame(all_deepsade_output)
        all_deepsade_output.columns = ['fold', 'num violations', 'accuracy train corrected', 'accuracy test corrected', 'accuracy train', 'accuracy test',
                                       'violations train', 'violations test', 'runtime']
        
        if MAXSMT_OPTIMIZATION:
            all_deepsade_output.to_csv(FOLDER_NAME + '/deepsade_output_{}.csv'.format(DEEPSADE_APPROACH))
            with open(FOLDER_NAME + '/deepsade_all_val_losses.pkl', 'wb') as f:
                pickle.dump(all_val_losses, f)
        else:
            all_deepsade_output.to_csv(FOLDER_NAME + '/deepsade_l_output_{}.csv'.format(DEEPSADE_APPROACH))
            with open(FOLDER_NAME + '/deepsade_l_all_val_losses.pkl', 'wb') as f:
                pickle.dump(all_val_losses, f)

    print('Finished Experiments')
    print("Total runtime", time.time() - start)
