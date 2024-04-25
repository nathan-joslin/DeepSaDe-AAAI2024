from models import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scoop import futures
import argparse
import os
import pickle

SADE_EPOCHS = 5
BASELINE_EPOCHS = 10
LEARNING_RATE = 0.001
MAXIMAL_STEP_SIZE = 0.1
SADE_BATCH_SIZE = 5
BASELINE_BATCH_SIZE = 5
SADE_MOMENTUM = 0
BASELINE_MOMENTUM = 0
SADE_WAITING_PERIOD = 10
SADE_MARGINS = [0.1]
SADE_INITIAL_WEIGHT = 1
OUTER_FOLDS = 5
INNER_FOLDS = 5
NETWORK = [13, 50, 50, 14, 5]
FOLDER_NAME = None
DEEPSADE_APPROACH = 'ls'
DC_INITIALIZATION = True
GRADIENT_RANDOMIZATION_STRATEGY = 1
TRANSFER_LEARNING = False
WEIGHT_DECAY = 0
VALIDATION_SET_RATIO = 0.1
MAPPED_FEATURES = [12]
MAXSMT_OPTIMIZATION=True

# python -m scoop -vv -n 5 main_with_cv.py --baseline 0 --deepsade 1 --seed 1


def get_violations(pred, X, K):
    A = np.sum(pred, axis=1) > (X.T[12] - K.min_[-1]) / K.scale_[-1]
    B = pred.T[1] > 0.05 * (X.T[12] - K.min_[-1]) / K.scale_[-1]
    return sum(A + B)


def calculate_corrected_mse(X, pred, y, K=None, corrected=True):
    if corrected:
        indices = np.where((np.sum(y, axis=1) <= (X.T[12] - K.min_[-1]) / K.scale_[-1]) * (
                y.T[1] <= 0.05 * (X.T[12] - K.min_[-1]) / K.scale_[-1]))[0]
        return mean_squared_error(pred[indices], y[indices])
    else:
        return mean_squared_error(pred, y)


def single_baseline_cv(X, y, train_idx=None, test_idx=None, K=None, alpha=0, beta=0, fold_index=0):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    net = ExpenseRegularizedModel(sizes=NETWORK)
    net.learn(X_train, y_train, learning_rate=LEARNING_RATE, batch_size=BASELINE_BATCH_SIZE, epochs=BASELINE_EPOCHS,
              momentum=BASELINE_MOMENTUM, K=K, alpha=alpha, beta=beta, validation_set_ratio=VALIDATION_SET_RATIO)

    pred_train = net.forward(torch.tensor(X_train)).detach().numpy()
    pred_test = net.forward(torch.tensor(X_test)).detach().numpy()

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        mse_train_corrected = calculate_corrected_mse(X_train, pred_train, y_train, K)
        mse_test_corrected = calculate_corrected_mse(X_test, pred_test, y_test, K)

        mse_train = calculate_corrected_mse(X_train, pred_train, y_train, K, corrected=False)
        mse_test = calculate_corrected_mse(X_test, pred_test, y_test, K, corrected=False)

        total_violations_train = get_violations(pred_train, X_train, K)
        total_violations_test = get_violations(pred_test, X_test, K)

        return [[fold_index, alpha, beta, mse_train_corrected, mse_test_corrected, mse_train, mse_test,
                 total_violations_train, total_violations_test]]


def get_alpha_beta(X, y, K=None):
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
        for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49]:
            list_X.append(X)
            list_y.append(y)
            list_train_idx.append(train_index)
            list_test_idx.append(test_index)
            list_K.append(K)
            list_index.append(fold_index)
            list_alphas.append(alpha)
            list_betas.append(alpha)
        fold_index += 1

    all_baseline_output = []
    baseline_results = list(futures.map(single_baseline_cv, list_X, list_y, list_train_idx, list_test_idx, list_K,
                                        list_alphas, list_betas, list_index))

    for out in baseline_results:
        if out is not None:
            all_baseline_output.extend(out)

    all_baseline_output = pd.DataFrame(all_baseline_output)
    all_baseline_output.columns = ['fold', 'alpha', 'beta', 'mse train corrected', 'mse test corrected',
                                   'mse train', 'mse test', 'violations train', 'violations test']

    processed_output_baseline = []
    for a in np.unique(all_baseline_output['alpha']):
        for b in np.unique(all_baseline_output['beta']):
            sub = all_baseline_output[(all_baseline_output['beta'] == b) &
                                      (all_baseline_output['alpha'] == a)]
            mse_mean = np.mean(sub['mse test corrected'])
            mse_std = np.std(sub['mse test corrected'])

            violations_mean = np.mean(sub['violations test'])
            violations_std = np.std(sub['violations test'])

            processed_output_baseline.append([a, b, mse_mean, mse_std, violations_mean, violations_std])
    processed_output_baseline = pd.DataFrame(processed_output_baseline)
    processed_output_baseline.columns = ['alpha', 'beta', 'mse mean', 'mse std', 'violations mean', 'violations std']
    return processed_output_baseline[processed_output_baseline['violations mean'] ==
                                     min(processed_output_baseline['violations mean'])].sort_values(by='mse mean')[
        ['alpha', 'beta']].values[0]


def single_baseline_run(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, folder_name='0'):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    a, b = get_alpha_beta(X_train, y_train, K=K)
    net = ExpenseRegularizedModel(sizes=NETWORK)
    net.learn(X_train, y_train, learning_rate=LEARNING_RATE, batch_size=BASELINE_BATCH_SIZE, epochs=BASELINE_EPOCHS,
              momentum=BASELINE_MOMENTUM, K=K, alpha=a, beta=b, validation_set_ratio=VALIDATION_SET_RATIO)

    pred_train = net.forward(torch.tensor(X_train)).detach().numpy()
    pred_test = net.forward(torch.tensor(X_test)).detach().numpy()

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        mse_train_corrected = calculate_corrected_mse(X_train, pred_train, y_train, K)
        mse_test_corrected = calculate_corrected_mse(X_test, pred_test, y_test, K)
        mse_train = calculate_corrected_mse(X_train, pred_train, y_train, K, corrected=False)
        mse_test = calculate_corrected_mse(X_test, pred_test, y_test, K, corrected=False)

        total_violations_train = get_violations(pred_train, X_train, K)
        total_violations_test = get_violations(pred_test, X_test, K)

        torch.save(net, folder_name + '/baseline_model_fold_{}.pt'.format(fold_index))

        return [[fold_index, a, b, mse_train_corrected, mse_test_corrected, mse_train, mse_test,
                 total_violations_train, total_violations_test, net.runtime]]


def single_sade_run(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, folder_name='0'):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    while True:
        model = SaDeExpenseRegressor(verbose=False, sizes=NETWORK, mapped_features=MAPPED_FEATURES)
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
        mse_train_corrected = calculate_corrected_mse(X_train, pred_train, y_train, K)
        mse_test_corrected = calculate_corrected_mse(X_test, pred_test, y_test, K)

        mse_train = calculate_corrected_mse(X_train, pred_train, y_train, K, corrected=False)
        mse_test = calculate_corrected_mse(X_test, pred_test, y_test, K, corrected=False)

        total_violations_train = get_violations(pred_train, X_train, K)
        total_violations_test = get_violations(pred_test, X_test, K)

        torch.save(model.best_model, folder_name + '/deepsade_model_fold_{}.pt'.format(fold_index))

        return [[fold_index, get_violations(y_train, X_train, K), mse_train_corrected, mse_test_corrected, mse_train, mse_test,
                 total_violations_train, total_violations_test, model.runtime]], model.validation_losses

def save_config(folder):
    with open('{}/config.txt'.format(folder), 'w') as f:
        f.write(' SADE_EPOCHS = {}\n BASELINE_EPOCHS = {}\n LEARNING_RATE = {}\n MAXIMAL_STEP_SIZE = {}\n SADE_BATCH_SIZE = {}\n BASELINE_BATCH_SIZE = {}\n SADE_MOMENTUM = {}\n BASELINE_MOMENTUM = {}\n SADE_WAITING_PERIOD = {}\n SADE_MARGINS = {}\n SADE_INITIAL_WEIGHT = {}\n OUTER_FOLDS = {}\n INNER_FOLDS = {}\n NETWORK = {}\n DEEPSADE_APPROACH = {}\n VALIDATION_SET_RATIO = {}\n DC_INITIALIZATION = {}\n GRADIENT_RANDOMIZATION_STRATEGY = {}\n MAPPED_FEATURES = {}\n TRANSFER_LEARNING = {}\n WEIGHT_DECAY = {}\n MAXSMT_OPTIMIZATION = {}\n'.format(
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
            WEIGHT_DECAY, MAXSMT_OPTIMIZATION))

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

    dataset = pd.read_csv('expense_prediction.csv').iloc[:, 1:]
    y = np.array(dataset.iloc[:, -5:], dtype=np.float32)
    X = dataset.iloc[:, 0:dataset.shape[1] - 5]
    X['Household Head Sex'] = [1 if x == 'Male' else 0 for x in X['Household Head Sex']]
    scalar = MinMaxScaler()
    X[X.columns] = scalar.fit_transform(X[X.columns])
    X = np.array(X, dtype=np.float32)
    X = np.nan_to_num(X)

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
        list_K.append(scalar)
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
        all_baseline_output.columns = ['fold', 'alpha', 'beta', 'mse train corrected', 'mse test corrected',
                                       'mse train', 'mse test', 'violations train', 'violations test', 'runtime']

        all_baseline_output.to_csv(FOLDER_NAME + '/baseline_output.csv')

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
        all_deepsade_output.columns = ['fold', 'num violations', 'mse train corrected', 'mse test corrected', 'mse train', 'mse test',
                                       'violations train', 'violations test', 'runtime']
        all_deepsade_output.to_csv(FOLDER_NAME + '/deepsade_output_{}.csv'.format(DEEPSADE_APPROACH))
        
        with open(FOLDER_NAME + '/all_val_losses.pkl', 'wb') as f:
            pickle.dump(all_val_losses, f)
            
    print('Finished Experiments')
    print("Total runtime", time.time() - start)
