from classifier import *
from util import *
import numpy as np
import pandas as pd

np.random.seed(42)

RUN_STAGE_1 = False # first step of selecting hyper-parameters, if false - will be loaded from previous test
RUN_STAGE_2 = True # second step of selecting other features, if false - will be loaded from previous test
RUN_STAGE_3 = True # final testing and evaluation, can also work without two previous stages with loaded parameters
# it is adviced to keep RUN_STAGE_1 = False, since many hyper-parameters are being tested and RUN_STAGE_2 = True to see additional features testing with final evaluation

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
eps=200
patience=10

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

# # Plot different activation functions on fixed parameters
# alpha_plot = 0.5
# hidden_plot = 10
# activations = ["tanh", "relu", "leaky_relu", "sigmoid"]
# colors = ['b', 'g', 'r', 'm']
#
# plt.figure(figsize=(10, 6))
#
# for i, act in enumerate(activations):
#     print(f"Training with activation: {act}")
#     model = MLPClassifier_mini_batch(
#         dim_in=est_inputs.shape[0],
#         dim_hid=hidden_plot,
#         n_classes=3,
#         activation=act
#     )
#
#     ce_list, _, _, _, _ = model.train_adam(
#         est_inputs, est_labels,
#         alpha=alpha_plot,
#         eps=eps,
#         batch_size=batch_size,
#         early_stopping=False,
#         val_inputs=val_inputs,
#         val_labels=val_labels,
#         live_plot=False
#     )
#
#     plt.plot(range(1, len(ce_list)+1), ce_list, label=act, color=colors[i])
#
# plt.xlabel("Epoch")
# plt.ylabel("Classification Error (CE)")
# plt.title(f"Convergence comparison (α={alpha_plot}, hidden={hidden_plot}, batch={batch_size})")
# plt.legend()
# plt.grid(False)
# plt.tight_layout()
# plt.show()

# ----- STAGE 1: search alpha, hidden units and activation function -----
if RUN_STAGE_1:
    alphas = [0.05, 0.1, 0.2, 0.5]
    hidden_units = [5, 7, 8]
    activations = ["tanh", "relu", "leaky_relu", "sigmoid"]

    best_score = float('inf')
    best_params = {}
    results_1 = []

    for alpha in alphas:
        for h in hidden_units:
            for act in activations:
                print(f"Testing alpha={alpha}, hidden={h}, activation={act}")
                model = MLPClassifier_mini_batch(dim_in=est_inputs.shape[0], dim_hid=h, n_classes=n_classes, activation=act)

                est_CE, est_RE, _, _, best_epoch = model.train_adam(est_inputs,
                            est_labels,
                            alpha=alpha,
                            eps=eps,
                            batch_size=batch_size,
                            early_stopping=True,
                            patience=patience,
                            val_inputs=val_inputs,
                            val_labels=val_labels
                            )
                CE, RE = model.test(val_inputs, val_labels)

                print(f"  -> Validation CE = {CE:.4f}")
                print(f"  -> Estimation CE = {est_CE[-1]:.4f}")

                results_1.append({
                                "alpha": alpha,
                                "hidden": h,
                                "est_CE": est_CE[-1],
                                "val_CE": CE,
                                "activation": act,
                                "best_epoch": best_epoch
                            })

                if CE < best_score:
                    best_score = CE
                    best_params = {"alpha": alpha, "hidden": h, "activation": act, "best_epoch": best_epoch}

    print("\nBest basic config:", best_params, "with CE =", best_score)
    df_results_1 = pd.DataFrame(results_1)
    # df_results_1.sort_values("val_CE").to_csv("results_stage1_batch_128_10_final_sorted.csv")
    print("\nAll results:")
    print(df_results_1.sort_values("val_CE"))
else:
    best_params = {"alpha": 0.5,
                   "hidden": 8,
                   "activation": "relu",
                   "best_epoch": 89
    }


# ----- STAGE 2: test momentum, learning rate schedule and L2 regularization on best config -----
if RUN_STAGE_2:
    momentum_options = [0.0, 0.5, 0.9]
    l2_values = [0.0, 0.01, 0.1]
    lr_schedules = ["constant", "linear", "geometric", "step"]
    lr_gamma = 0.9
    lr_step_size = 10

    best_score_stage2 = float('inf')
    best_full_config = {}
    results_2 = []

    for momentum in momentum_options:
        for l2 in l2_values:
            for schedule in lr_schedules:
                print(f"Testing momentum={momentum}, l2_lambda={l2}, lr_schedules={schedule}")
                model = MLPClassifier_mini_batch(dim_in=est_inputs.shape[0], dim_hid=best_params['hidden'], n_classes=n_classes, activation=best_params['activation'])

                est_CE, est_RE, _, _, best_epoch = model.train_adam(est_inputs, est_labels,
                            alpha=best_params['alpha'],
                            use_momentum=True,
                            momentum_gamma=momentum,
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

                results_2.append({
                    "alpha": best_params['alpha'],
                    "hidden": best_params['hidden'],
                    "activation": best_params['activation'],
                    "momentum": momentum,
                    "l2_lambda": l2,
                    "lr_schedule": schedule,
                    "est_CE": est_CE[-1],
                    "val_CE": CE,
                    "best_epoch": best_epoch
                })

                if CE <= best_score_stage2:
                    best_score_stage2 = CE
                    best_full_config = {
                        "alpha": best_params['alpha'],
                        "hidden": best_params['hidden'],
                        "activation": best_params['activation'],
                        "momentum": momentum,
                        "l2_lambda": l2,
                        "lr_schedule": schedule,
                        "best_epoch": best_epoch
                    }

    print("\nBest full config:", best_full_config, "with CE =", best_score_stage2)
    df_results_2 = pd.DataFrame(results_2)
    # df_results_2.sort_values("val_CE").to_csv("results_stage2_batch_128_epoch_10_final_sorted.csv")
    print("\nAll results:")
    print(df_results_2.sort_values("val_CE"))
else:
    best_full_config = {
        "alpha": 0.5,
        "hidden": 8,
        "activation": "relu",
        "momentum": 0.9,
        "l2_lambda": 0.0,
        "lr_schedule": "linear",
        "best_epoch": 16
    }


# ----- STAGE 3: final training & testing on best config -----

if RUN_STAGE_3:
    final_model = MLPClassifier_mini_batch(dim_in=train_inputs.shape[0], dim_hid=best_full_config['hidden'], n_classes=n_classes, activation=best_full_config['activation'])

    train_CE, train_RE, _, _, _ = final_model.train_adam(train_inputs, train_labels,
                alpha=best_full_config['alpha'],
                eps=int(best_full_config['best_epoch']*1.25),
                batch_size=batch_size,
                l2_penalty=False,
                l2_lambda=best_full_config['l2_lambda'],
                use_momentum=True,
                momentum_gamma=best_full_config['momentum'],
                early_stopping=False,
                patience=patience,
                lr_schedule=best_full_config['lr_schedule'],
                lr_gamma=0.9,
                lr_step_size = 10
                )

    # Testing
    test_CE, test_RE = final_model.test(test_inputs, test_labels)
    print(f"\n[TEST] CE = {test_CE:.2%}, RE = {test_RE:.4f}")
    print("\nBest full config:", best_full_config)

    # Plot results
    # _, train_predicted = final_model.predict(train_inputs)
    _, test_predicted = final_model.predict(test_inputs)
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
    # df_conf_matrix.to_csv("conf_matrix_percent_final.csv")
    print(df_conf_matrix)