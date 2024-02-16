from models import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from scoop import futures
import argparse
import os
import idx2numpy
from itertools import chain
import pickle


SADE_EPOCHS = 40
BASELINE_EPOCHS = 40
LEARNING_RATE = 0.1
MAXIMAL_STEP_SIZE = 0.1
SADE_BATCH_SIZE = 10
BASELINE_BATCH_SIZE = 10
SADE_MOMENTUM = 0
BASELINE_MOMENTUM = 0
SADE_WAITING_PERIOD = 5
SADE_MARGINS = [0, 1, 2]
SADE_INITIAL_WEIGHT = 1
OUTER_FOLDS = 5
INNER_FOLDS = 5
NETWORK = [36, 25, 25, 10, 16]
FOLDER_NAME = None
VALIDATION_SET_RATIO=0.1
DC_INITIALIZATION = False
DEEPSADE_APPROACH = 'ls'
GRADIENT_RANDOMIZATION_STRATEGY = 1
WEIGHT_DECAY=0
MAPPED_FEATURES=None
TRANSFER_LEARNING=False
MAXSMT_OPTIMIZATION = False
BASELINE_MODEL='sl'

# python -m scoop -vv -n 5 main_with_cv.py --baseline 1 --deepsade 1 --seed 3

def constraint_satsified(p):
    if (sum(p[:4]) == 1) and (sum(p[4:8]) == 1) and (sum(p[8:12]) == 1) and (sum(p[12:]) == 1) and (sum([p[0 + j*4] for j in range(4)]) == 1) and (sum([p[1 + j*4] for j in range(4)]) == 1) and (sum([p[2 + j*4] for j in range(4)]) == 1) and (sum([p[3 + j*4] for j in range(4)]) == 1):
        return True
    else:
        return False

def coherent_accuracy(pred, y):
    coherent_accuracy = 0
    for i in range(len(pred)):
        if all(pred[i] == y[i]):
            coherent_accuracy += 1
    return  coherent_accuracy/len(pred)

def corrected_accuracy(pred, y, accuracy_type='standard'):
    if accuracy_type == 'coherent':
        return coherent_accuracy(pred, y)
    elif accuracy_type == 'incoherent':
        return accuracy_score(np.array(pred).flatten(), y.flatten())
    else:
        return NeuralPreferenceClassifier.accuracy_score_ml(pred, y)

def get_violations(pred):
    count = 0
    for p in pred:
        if (sum(p[:4]) == 1) and (sum(p[4:8]) == 1) and (sum(p[8:12]) == 1) and (sum(p[12:]) == 1) and (sum([p[0 + j*4] for j in range(4)]) == 1) and (sum([p[1 + j*4] for j in range(4)]) == 1) and (sum([p[2 + j*4] for j in range(4)]) == 1) and (sum([p[3 + j*4] for j in range(4)]) == 1):
            count += 1
    return len(pred) - count


def single_baseline_cv(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, alpha=0):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    net = PreferenceRegularizedModel(sizes=NETWORK)
    net.learn(X_train, y_train, learning_rate=LEARNING_RATE, batch_size=BASELINE_BATCH_SIZE, epochs=BASELINE_EPOCHS,
              momentum=BASELINE_MOMENTUM, alpha=alpha)

    pred_train = net.predict(X_train)
    pred_test = net.predict(X_test)

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        accuracy_train_coherent = corrected_accuracy(pred_train, y_train, accuracy_type='coherent')
        accuracy_test_coherent = corrected_accuracy(pred_test, y_test, accuracy_type='coherent')
        
        accuracy_train_incoherent = corrected_accuracy(pred_train, y_train, accuracy_type='incoherent')
        accuracy_test_incoherent = corrected_accuracy(pred_test, y_test, accuracy_type='incoherent')
        
        accuracy_train = corrected_accuracy(pred_train, y_train)
        accuracy_test = corrected_accuracy(pred_test, y_test)

        total_violations_train = get_violations(pred_train)
        total_violations_test = get_violations(pred_test)

        return [[fold_index, alpha, accuracy_train_coherent, accuracy_test_coherent, accuracy_train_incoherent, 
                 accuracy_test_incoherent, accuracy_train, accuracy_test,
                 total_violations_train, total_violations_test, 1 - total_violations_train/len(X_train), 1 - total_violations_test/len(X_test), 
                 net.runtime]]

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
    fold_index = 1
    for train_index, test_index in split_data_CV:
        for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
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
                                       list_index, list_alphas))

    for out in baseline_results:
        if out is not None:
            all_baseline_output.extend(out)

    all_baseline_output = pd.DataFrame(all_baseline_output)
    all_baseline_output.columns = ['fold', 'alpha', 'accuracy train coherent', 'accuracy test coherent',
                                       'accuracy train incoherent', 'accuracy test incoherent',
                                       'accuracy train', 'accuracy test', 'violations train',
                                       'violations test', 'constraint accuracy train', 'constraint accuracy test', 'runtime']

    processed_output_baseline = []
    for a in np.unique(all_baseline_output['alpha']):
        sub = all_baseline_output[(all_baseline_output['alpha'] == a)]
        accuracy_mean = np.mean(sub['accuracy test'])
        accuracy_std = np.std(sub['accuracy test'])

        violations_mean = np.mean(sub['violations test'])
        violations_std = np.std(sub['violations test'])

        processed_output_baseline.append([a, accuracy_mean, accuracy_std, violations_mean, violations_std])
    
    processed_output_baseline = pd.DataFrame(processed_output_baseline)
    processed_output_baseline.columns = ['alpha', 'accuracy mean', 'accuracy std', 'violations mean', 'violations std']
#     return processed_output_baseline.sort_values(by='accuracy mean')[
#         ['alpha']].values[0]
    
    return processed_output_baseline[processed_output_baseline['violations mean'] ==
                                     min(processed_output_baseline['violations mean'])].sort_values(by='accuracy mean')[
        ['alpha']].values[0]
    
def single_baseline_run(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, folder_name='0'):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    net = PreferenceRegularizedModel(sizes=NETWORK)
    if BASELINE_MODEL == 'sl':
        a = get_alpha(X_train, y_train)[0]
    else:
        a = 0

    net.learn(X_train, y_train, learning_rate=LEARNING_RATE, batch_size=BASELINE_BATCH_SIZE, epochs=BASELINE_EPOCHS,
              momentum=BASELINE_MOMENTUM, alpha=a)

    pred_train = net.predict(X_train)
    pred_test = net.predict(X_test)

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        accuracy_train_coherent = corrected_accuracy(pred_train, y_train, accuracy_type='coherent')
        accuracy_test_coherent = corrected_accuracy(pred_test, y_test, accuracy_type='coherent')
        
        accuracy_train_incoherent = corrected_accuracy(pred_train, y_train, accuracy_type='incoherent')
        accuracy_test_incoherent = corrected_accuracy(pred_test, y_test, accuracy_type='incoherent')
        
        accuracy_train = corrected_accuracy(pred_train, y_train)
        accuracy_test = corrected_accuracy(pred_test, y_test)

        total_violations_train = get_violations(pred_train)
        total_violations_test = get_violations(pred_test)

        torch.save(net, folder_name + '/baseline_model_fold_{}_{}.pt'.format(fold_index, BASELINE_MODEL))

        return [[fold_index, a, accuracy_train_coherent, accuracy_test_coherent, accuracy_train_incoherent, 
                 accuracy_test_incoherent, accuracy_train, accuracy_test,
                 total_violations_train, total_violations_test, 1 - total_violations_train/len(X_train), 1 - total_violations_test/len(X_test), 
                 net.runtime]]


def single_sade_run(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, folder_name='0'):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    while True:
        model = NeuralPreferenceClassifier(verbose=False, sizes=NETWORK, mapped_features=None)
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
                    gradient_randomization_strategy=GRADIENT_RANDOMIZATION_STRATEGY,
                    dc_initialization=DC_INITIALIZATION,
                    validation_set_ratio=VALIDATION_SET_RATIO,
                    transfer_learning=TRANSFER_LEARNING,
                    maxsmt_optimization=MAXSMT_OPTIMIZATION
                    )

        if len(model.validation_losses) > 30:
            break

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        accuracy_train_coherent = corrected_accuracy(pred_train, y_train, accuracy_type='coherent')
        accuracy_test_coherent = corrected_accuracy(pred_test, y_test, accuracy_type='coherent')
        
        accuracy_train_incoherent = corrected_accuracy(pred_train, y_train, accuracy_type='incoherent')
        accuracy_test_incoherent = corrected_accuracy(pred_test, y_test, accuracy_type='incoherent')
        
        accuracy_train = corrected_accuracy(pred_train, y_train)
        accuracy_test = corrected_accuracy(pred_test, y_test)

        total_violations_train = get_violations(pred_train)
        total_violations_test = get_violations(pred_test)

        torch.save(model.best_model, folder_name + '/deepsade_model_fold_{}.pt'.format(fold_index))

        return [[fold_index, accuracy_train_coherent, accuracy_test_coherent, accuracy_train_incoherent, 
                 accuracy_test_incoherent, accuracy_train, accuracy_test,
                 total_violations_train, total_violations_test, 1 - total_violations_train/len(X_train), 1 - total_violations_test/len(X_test), 
                 model.runtime]], model.validation_losses

def save_config(folder):
    with open('{}/config.txt'.format(folder), 'a+') as f:
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
         
def to_one_hot(dense, n):
    one_hot = np.zeros(n)
    one_hot[dense] = 1
    return one_hot

def to_perm_matrix(ranking, items):
    # We're going to flatten along the rows, i.e. entries 0-4 are a row (the one hot ranking of the first item), 5-9, etc.
    ret = []
    n = len(items)
    for item in items:
        ret.extend(to_one_hot(ranking.index(item), n))

    return ret
    
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
    
    DATA_IND = [1, 2, 3, 5, 7, 8]
    LABEL_IND = [4, 6, 9, 10]
    
    data_path = 'sushi.soc'
    instances = []
    labels = []
    with open(data_path) as file:
        for line in file:
            tokens = line.strip().split(',')
            # Doesn't have enough entries, isn't data
            if len(tokens) < 10: continue
            # First digit is useless
            ranking = [int(x) for x in tokens[1:]]
            cur_data = []
            cur_label = []
            for item in ranking:
                if item in DATA_IND:
                    cur_data.append(item)
                else:
                    cur_label.append(item)
            instances.append(to_perm_matrix(cur_data, DATA_IND))
            labels.append(to_perm_matrix(cur_label, LABEL_IND))
    
    X = np.array(instances, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

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
        list_K.append(True)
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
        all_baseline_output.columns = ['fold', 'alpha', 'accuracy train coherent', 'accuracy test coherent',
                                       'accuracy train incoherent', 'accuracy test incoherent',
                                       'accuracy train', 'accuracy test', 'violations train',
                                       'violations test', 'constraint accuracy train', 'constraint accuracy test', 'runtime']

        all_baseline_output.to_csv(FOLDER_NAME + '/baseline_output_{}.csv'.format(BASELINE_MODEL))

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
        all_deepsade_output.columns = ['fold', 'accuracy train coherent', 'accuracy test coherent',
                                       'accuracy train incoherent', 'accuracy test incoherent',
                                       'accuracy train', 'accuracy test', 'violations train',
                                       'violations test', 'constraint accuracy train', 'constraint accuracy test', 'runtime']
        all_deepsade_output.to_csv(FOLDER_NAME + '/deepsade_output_{}.csv'.format(DEEPSADE_APPROACH))
        with open(FOLDER_NAME + '/all_val_losses.pkl', 'wb') as f:
            pickle.dump(all_val_losses, f)
            
    print('Finished Experiments')
    print("Total runtime", time.time() - start)
