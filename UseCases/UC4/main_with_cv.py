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


SADE_EPOCHS = 60
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
NETWORK = [3136, 50, 50, 10, 10]
FOLDER_NAME = None
VALIDATION_SET_RATIO=0.1
DC_INITIALIZATION = False
DEEPSADE_APPROACH = 'ls'
GRADIENT_RANDOMIZATION_STRATEGY = 1
WEIGHT_DECAY=0
MAPPED_FEATURES=None
TRANSFER_LEARNING=False
NUM_SAMPLES = 2000
MAXSMT_OPTIMIZATION = False

# python -m scoop -vv -n 5 main_with_cv.py --baseline 1 --deepsade 1 --seed 3

def sample_coordinate(high, size):
    if high > 0:
        return np.random.randint(high, size=size)
    else:
        return np.zeros(size).astype(np.int64)

def make_combined_image(label, indices_per_label):
    h, w = 28, 28
    image_size = 128
    imgs = [D[random.sample(indices_per_label[l], 1)] for l in label]
    n = len(imgs)
    image = np.zeros(([image_size, image_size])).astype(np.uint8)
    ys = sample_coordinate(image_size - h, n)
    xs = sample_coordinate(image_size // n - w, size=n)
    xs = [l * image_size // n + xs[l] for l in range(n)]
    for i in range(n):
        image[ys[i]:ys[i] + h, xs[i]:xs[i] + w] = imgs[i]
    return image


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
    elif accuracy_type == 'corrected':
        indices = [i for i in range(len(y)) if sum(np.where(y[i] == 1)[0]) <= 10]
        if len(indices) > 0:
            return NeuralCombinedMnistClassifier.accuracy_score_ml(pred[indices], y[indices])
        else:
            return np.nan
    else:
        return NeuralCombinedMnistClassifier.accuracy_score_ml(pred, y)

def get_violations(pred):
    return sum([1 for p in pred if sum(np.where(p == 1)[0]) <= 10])


def single_baseline_run(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, folder_name='0'):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    net = MNISTBaselineModel(sizes=NETWORK)
    net.learn(X_train, y_train, learning_rate=LEARNING_RATE, batch_size=BASELINE_BATCH_SIZE, epochs=BASELINE_EPOCHS,
              momentum=BASELINE_MOMENTUM)

    pred_train = net.predict(X_train)
    pred_test = net.predict(X_test)

    if np.sum(np.isnan(pred_train)) == 0 and np.sum(np.isnan(pred_test)) == 0:
        accuracy_train_coherent = corrected_accuracy(pred_train, y_train, accuracy_type='coherent')
        accuracy_test_coherent = corrected_accuracy(pred_test, y_test, accuracy_type='coherent')
        accuracy_train_incoherent = corrected_accuracy(pred_train, y_train, accuracy_type='incoherent')
        accuracy_test_incoherent = corrected_accuracy(pred_test, y_test, accuracy_type='incoherent')
        
        accuracy_train_corrected = corrected_accuracy(pred_train, y_train, accuracy_type='corrected')
        accuracy_test_corrected = corrected_accuracy(pred_test, y_test, accuracy_type='corrected')
        accuracy_train = corrected_accuracy(pred_train, y_train)
        accuracy_test = corrected_accuracy(pred_test, y_test)

        total_violations_train = get_violations(pred_train)
        total_violations_test = get_violations(pred_test)

        torch.save(net, folder_name + '/baseline_model_fold_{}.pt'.format(fold_index))

        return [[fold_index, accuracy_train_coherent, accuracy_test_coherent, accuracy_train_incoherent, accuracy_test_incoherent, 
                 accuracy_train_corrected, accuracy_test_corrected, accuracy_train, accuracy_test,
                 total_violations_train, total_violations_test, 1 - total_violations_train/len(X_train), 1 - total_violations_test/len(X_test), net.runtime]]


def single_sade_run(X, y, train_idx=None, test_idx=None, K=None, fold_index=0, folder_name='0'):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    while True:
        model = NeuralCombinedMnistClassifier(verbose=False, sizes=NETWORK, mapped_features=None)
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
                    transfer_learning=TRANSFER_LEARNING, maxsmt_optimization=MAXSMT_OPTIMIZATION
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
        
        accuracy_train_corrected = corrected_accuracy(pred_train, y_train, accuracy_type='corrected')
        accuracy_test_corrected = corrected_accuracy(pred_test, y_test, accuracy_type='corrected')
        accuracy_train = corrected_accuracy(pred_train, y_train)
        accuracy_test = corrected_accuracy(pred_test, y_test)

        total_violations_train = get_violations(pred_train)
        total_violations_test = get_violations(pred_test)

        torch.save(model.best_model, folder_name + '/deepsade_model_fold_{}.pt'.format(fold_index))

        return [[fold_index, accuracy_train_coherent, accuracy_test_coherent, accuracy_train_incoherent, accuracy_test_incoherent, 
                 accuracy_train_corrected, accuracy_test_corrected, accuracy_train, accuracy_test,
                 total_violations_train, total_violations_test, 1 - total_violations_train/len(X_train), 1 - total_violations_test/len(X_test),
                 model.runtime]], model.validation_losses

def save_config(folder):
    with open('{}/config.txt'.format(folder), 'a+') as f:
        f.write(' SADE_EPOCHS = {}\n BASELINE_EPOCHS = {}\n LEARNING_RATE = {}\n MAXIMAL_STEP_SIZE = {}\n SADE_BATCH_SIZE = {}\n BASELINE_BATCH_SIZE = {}\n SADE_MOMENTUM = {}\n BASELINE_MOMENTUM = {}\n SADE_WAITING_PERIOD = {}\n SADE_MARGINS = {}\n SADE_INITIAL_WEIGHT = {}\n OUTER_FOLDS = {}\n INNER_FOLDS = {}\n NETWORK = {}\n DEEPSADE_APPROACH = {}\n VALIDATION_SET_RATIO = {}\n DC_INITIALIZATION = {}\n GRADIENT_RANDOMIZATION_STRATEGY = {}\n MAPPED_FEATURES = {}\n TRANSFER_LEARNING = {}\n WEIGHT_DECAY = {}\n MAXSMT_OPTIMIZATION = {}\n NUMBER OF SAMPLES = {}\n'.format(
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
            MAXSMT_OPTIMIZATION, 
            NUM_SAMPLES))
        
def make_labels(n):
    while True:
        y = random.sample(range(10), n)
        if sum(y) > 10:
            return y
        
def make_data(n=20000):
    np.random.seed(1)
    random.seed(1)
    
    D = idx2numpy.convert_from_file('train-images-idx3-ubyte')
    labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
    indices_per_label = {l: list(np.where(labels == l)[0]) for l in np.unique(labels)}
    min_labels_per_image = 4
    max_labels_per_image = 4
    X = []
    y = [np.zeros(10) for i in range(len(labels[:n]))]
    for i in range(len(labels[:n])):
        y_i = make_labels(random.randint(min_labels_per_image, max_labels_per_image))
        y[i][y_i] = 1
        X_i = [D[random.sample(indices_per_label[l], 1)][0].flatten() for l in y_i]
        X.append(list(chain(*X_i)))
        if i % 1000 == 0:
            print('generated {} instances'.format(i))
    X = np.array(X, dtype=np.float32)
    X = X/255
    y = np.array(y)
    return X, y
        
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
    save_config(FOLDER_NAME)
    
    X, y = make_data(n=NUM_SAMPLES)
    
    # setting seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
        all_baseline_output.columns = ['fold', 'accuracy train coherent', 'accuracy test coherent',
                                       'accuracy train incoherent', 'accuracy test incoherent',
                                       'accuracy train corrected', 'accuracy test corrected',
                                       'accuracy train', 'accuracy test', 'violations train',
                                       'violations test', 'constraint accuracy train', 'constraint accuracy test', 'runtime']

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
        all_deepsade_output.columns = ['fold', 'accuracy train coherent', 'accuracy test coherent',
                                       'accuracy train incoherent', 'accuracy test incoherent',
                                       'accuracy train corrected', 'accuracy test corrected',
                                       'accuracy train', 'accuracy test', 'violations train',
                                       'violations test', 'constraint accuracy train', 'constraint accuracy test', 'runtime']
        all_deepsade_output.to_csv(FOLDER_NAME + '/deepsade_output_{}.csv'.format(DEEPSADE_APPROACH))
        with open(FOLDER_NAME + '/all_val_losses.pkl', 'wb') as f:
            pickle.dump(all_val_losses, f)
            
    print('Finished Experiments')
    print("Total runtime", time.time() - start)
