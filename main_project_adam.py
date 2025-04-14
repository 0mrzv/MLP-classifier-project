from classifier import *
from util import *
import numpy as np
import pandas as pd

np.random.seed(42)

RUN_STAGE_4 = True # additional Adam optimization testing

def load_dataset(path):
    raw_data = np.genfromtxt(path, dtype=str, skip_header=1)
    inputs = raw_data[:, :-1].astype(float).T
    label_str = raw_data[:, -1]
    label_map = {'A': 0, 'B': 1, 'C': 2}
    labels = np.array([label_map[label] for label in label_str])
    return inputs, labels

train_inputs, train_labels = load_dataset("2d.trn.dat")
test_inputs, test_labels = load_dataset("2d.tst.dat")

batch_size = 128
n_classes = np.max(train_labels)+1
eps=500
patience=30
best_params = {"alpha": 0.5,
                   "hidden": 8,
                   "activation": "relu",
                   "best_epoch": 89
    }

# Plot data before training
# plot_dots(train_inputs, train_labels, None, test_inputs, test_labels, None)

# Normalization of inputs
mean = np.mean(train_inputs, axis=1, keepdims=True)
std = np.std(train_inputs, axis=1, keepdims=True)
train_inputs = (train_inputs - mean) / std
test_inputs = (test_inputs - mean) / std

# Split estimation + validation sets
count = train_inputs.shape[1]
perm = np.random.permutation(count)
split = int(0.8 * count)
est_inputs = train_inputs[:, perm[:split]]
est_labels = train_labels[perm[:split]]
val_inputs = train_inputs[:, perm[split:]]
val_labels = train_labels[perm[split:]]

# ----- STAGE 4: Adam optimization -----
if RUN_STAGE_4:
    l2_values = [0.0, 0.01, 0.1]
    lr_schedules = ["constant", "linear", "geometric", "step"]
    lr_gamma = 0.9
    lr_step_size = 10

    best_score_stage3 = float('inf')
    best_full_config_adam = {}
    results_3 = []

    for l2 in l2_values:
        for schedule in lr_schedules:
            print(f"Testing optimizer Adam, l2_lambda={l2}, lr_schedules={schedule}")
            model = MLPClassifier_mini_batch(dim_in=est_inputs.shape[0], dim_hid=best_params['hidden'], n_classes=n_classes, activation=best_params['activation'])

            est_CE, est_RE, _, _, best_epoch = model.train_adam(est_inputs, est_labels,
                        alpha=best_params['alpha'],
                        use_momentum=False,
                        use_adam=True,
                        momentum_gamma=0.0,
                        l2_penalty=True,
                        l2_lambda=l2,
                        eps=eps,
                        batch_size=batch_size,
                        early_stopping=True,
                        patience=patience,
                        val_inputs=val_inputs, val_labels=val_labels,
                        lr_schedule=schedule,
                        lr_gamma=lr_gamma,
                        lr_step_size=lr_step_size
                        )
            CE, RE = model.test(val_inputs, val_labels, l2_penalty=True, l2_lambda=l2)

            print(f"  -> Validation CE = {CE:.4f}")
            print(f"  -> Estimation CE = {est_CE[-1]:.4f}")

            results_3.append({
                "alpha": best_params['alpha'],
                "hidden": best_params['hidden'],
                "activation": best_params['activation'],
                "momentum": 0.0,
                "l2_lambda": l2,
                "lr_schedule": schedule,
                "est_CE": est_CE[-1],
                "val_CE": CE,
                "best_epoch": best_epoch
            })

            if CE <= best_score_stage3:
                best_score_stage3 = CE
                best_full_config_adam = {
                    "alpha": best_params['alpha'],
                    "hidden": best_params['hidden'],
                    "activation": best_params['activation'],
                    "momentum": 0.0,
                    "l2_lambda": l2,
                    "lr_schedule": schedule,
                    "best_epoch": best_epoch
                }

    print("\nBest full Adam config:", best_full_config_adam, "with CE =", best_score_stage3)
    df_results_3 = pd.DataFrame(results_3)
    # df_results_3.sort_values("val_CE").to_csv("results_stage3_batch_128_epoch_10_final_adam.csv")
    print("\nAll results:")
    print(df_results_3.sort_values("val_CE"))

    # final training & testing on best adam config
    final_model_adam = MLPClassifier_mini_batch(dim_in=train_inputs.shape[0], dim_hid=best_full_config_adam['hidden'],
                                                n_classes=n_classes, activation=best_full_config_adam['activation'])

    train_CE, train_RE, _, _, _ = final_model_adam.train_adam(train_inputs, train_labels,
                                                              alpha=best_full_config_adam['alpha'],
                                                              eps=int(best_full_config_adam['best_epoch'] * 1.25),
                                                              batch_size=batch_size,
                                                              l2_penalty=False,
                                                              l2_lambda=best_full_config_adam['l2_lambda'],
                                                              use_adam=True,
                                                              use_momentum=False,
                                                              momentum_gamma=best_full_config_adam['momentum'],
                                                              early_stopping=False,
                                                              patience=patience,
                                                              lr_schedule=best_full_config_adam['lr_schedule'],
                                                              lr_gamma=0.9,
                                                              lr_step_size=10
                                                              )

    # Testing
    test_CE, test_RE = final_model_adam.test(test_inputs, test_labels)
    print(f"\n[TEST] CE = {test_CE:.2%}, RE = {test_RE:.4f}")
    print("\nBest full adam config:", best_full_config_adam)

    # Plot results
    # _, train_predicted = final_model_adam.predict(train_inputs)
    _, test_predicted = final_model_adam.predict(test_inputs)
    # Choose which graphs you want to see
    # plot_dots(train_inputs, train_labels, train_predicted, test_inputs, test_labels, test_predicted, block=False)
    plot_dots(None, None, None, test_inputs, test_labels, test_predicted, title='Test data only', block=False)
    plot_both_errors(train_CE, train_RE, test_CE, test_RE, block=False)
    plot_accuracy(train_CE, test_CE, block=False)

    conf_matrix = compute_confusion_matrix(test_labels, test_predicted)
    class_names = ["A", "B", "C"]
    conf_matrix_percent = normalize_confusion_matrix(conf_matrix)
    conf_matrix_rounded = np.round(conf_matrix_percent, decimals=2)
    df_conf_matrix = pd.DataFrame(conf_matrix_rounded, index=class_names, columns=class_names)
    # df_conf_matrix.to_csv("conf_matrix_percent_final_adam.csv")
    print(df_conf_matrix)
else:
    best_full_config_adam = {
        "alpha": 0.5,
        "hidden": 8,
        "activation": "relu",
        "momentum": 0.0,
        "l2_lambda": 0.0,
        "lr_schedule": "geometric",
        "best_epoch": 33
    }