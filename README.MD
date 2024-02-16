This document provides instructions to run the experiments.

### Parameters and how to run the codes
5 folders corresponding to 5 use cases can be found in the UseCases Folder of the submitted zip. 
For each use case, a 'main_with_cv.py' file is provided which is the main experiment file.
To run this script, following command should be used:

    python -m scoop -vv -n 'num-cores' main_with_cv.py --baseline '1 for running the baseline models, 0 otherwise' --deepsade '1 for running the deepsade models, 0 otherwise' --seed 'experiment number'

Each main_with_cv.py file contains a list of global parameters:

1. SADE_EPOCHS = number of epochs for deepsade model
2. BASELINE_EPOCHS = number of epochs for baseline models
3. LEARNING_RATE = learning rate for both deepsade and baseline
4. MAXIMAL_STEP_SIZE = maximal step size for deepsade
5. SADE_BATCH_SIZE = batch size for deepsade model
6. BASELINE_BATCH_SIZE = batch size for baseline models
7. SADE_MOMENTUM = momentum for deepsade (unused)
8. BASELINE_MOMENTUM = momentum for baselines (unused)
9. SADE_WAITING_PERIOD = waiting period for deepsade (how mch time to wait at each iteration before initiating restart)
10. SADE_MARGINS = list of margin/error thresholds 
11. SADE_INITIAL_WEIGHT = initial bounds of the deepsade model
12. OUTER_FOLDS = outer fold for cross validation
13. INNER_FOLDS = inner fold for hyper parameter search
14. NETWORK = network configuration
15. VALIDATION_SET_RATIO = validation set size (in fractional form)
16. DC_INITIALIZATION = initialization (True if we want to involve decision constraints in the initialization)
17. DEEPSADE_APPROACH = 'ls' (always 'ls' which means the approach involves line search)
18. GRADIENT_RANDOMIZATION_STRATEGY = 1 (1 for 'random restart' explained in the paper)
19. WEIGHT_DECAY = weight decay (unused)
20. MAPPED_FEATURES = feature indices in a list for the features to be mapped using skip-connections
21. TRANSFER_LEARNING = to do transfer learning at the start (unused)
22. BASELINE_APPROACH = which baseline approach to use (depends on the use cases, some have it some do not)


### Dependencies

1. Z3py (Z3 solver: version 4.8.10)
2. Pytorch
3. Following library for semantic loss in UC 5: https://github.com/lucadiliello/semantic-loss-pytorch
4. pandas
5. numpy
6. sklearn
7. scoop (for parallelization)