from mlp import *
from util import *

class MLPClassifier_mini_batch(MLP):
    def __init__(self, dim_in, dim_hid, n_classes, activation="tanh"):
        self.n_classes = n_classes
        self.activation = activation.lower()
        super().__init__(dim_in, dim_hid, dim_out=n_classes)

    def error(self, targets, outputs, l2_lambda=0.0):
        """
        Cost / loss / error function
        """
        mse_loss = np.sum((targets - outputs) ** 2, axis=0)
        l2_term = l2_lambda * (
                np.sum(self.W_hid[:, :-1] ** 2) +
                np.sum(self.W_out[:, :-1] ** 2)
        )
        return mse_loss + l2_term

    # @override
    def f_hid(self, x):
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    # @override
    def df_hid(self, x):
        if self.activation == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation == "relu":
            return (x > 0).astype(float)
        elif self.activation == "leaky_relu":
            return np.where(x > 0, 1.0, 0.01)
        elif self.activation == "sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    # @override
    def f_out(self, x):
        return 1 / (1 + np.exp(-x))  # sigmoid

    # @override
    def df_out(self, x):
        return self.f_out(x) * (1 - self.f_out(x))

    def predict(self, inputs):
        """
        Prediction = forward pass
        """
        outputs = np.stack([self.forward(x)[-1] for x in inputs.T]).T
        return outputs, onehot_decode(outputs)

    def test(self, inputs, labels, l2_penalty=False, l2_lambda=0.0):
        """
        Test model: forward pass on given inputs, and compute errors
        """
        targets = onehot_encode(labels, self.n_classes)  # convert labels to target vectors (each column is one target vector)
        outputs, predicted = self.predict(inputs)
        CE = np.mean(predicted != labels)       # mean classification error (a number from range <0, 1>)
        RE = np.mean(self.error(targets, outputs, l2_lambda=l2_lambda if l2_penalty else 0.0))      # regression error
        return CE, RE

    def train_adam(self, inputs, labels, alpha=0.1, eps=100,
              batch_size=16,
              live_plot=False, live_plot_interval=10,
              l2_penalty=False,
              l2_lambda=0.0,  # L2 regularization strength
              use_momentum=False,  # whether to use momentum
              momentum_gamma=0.9,  # strength of momentum
              early_stopping=False,  # whether to use early stopping
              patience=10,  # patience for early stopping
              val_inputs=None, val_labels=None,
              lr_schedule="constant",  # type of learning rate schedule
              lr_gamma=0.9,  # for geometric/step decay
              lr_step_size=10,  # for step decay
              use_adam=False,  # whether to use Adam optimizer
              beta1=0.9, beta2=0.999, epsilon=1e-8  # Adam hyperparameters
              ):

        (_, count) = inputs.shape
        targets = onehot_encode(labels, self.n_classes)

        indices = np.arange(count)
        prev_dW_hid = np.zeros_like(self.W_hid)
        prev_dW_out = np.zeros_like(self.W_out)

        # Adam: initialize first and second moment estimates
        m_hid = np.zeros_like(self.W_hid)
        v_hid = np.zeros_like(self.W_hid)
        m_out = np.zeros_like(self.W_out)
        v_out = np.zeros_like(self.W_out)

        CEs, REs = [], []
        val_CEs, val_REs = [], []
        best_val_CE = float("inf")
        best_weights = None
        epochs_no_improve = 0
        best_epoch = -1
        alpha_0 = alpha

        for ep in range(eps):
            np.random.shuffle(indices) # Shuffle for stochasticity

            for start in range(0, count, batch_size):
                end = min(start + batch_size, count)
                batch_idx = indices[start:end]

                x_batch = inputs[:, batch_idx]
                d_batch = targets[:, batch_idx]

                # Forward and backward for mini-batch
                a, h, b, y = self.forward(x_batch)
                dW_hid, dW_out = self.backward(x_batch, a, h, b, y, d_batch)

                if l2_penalty:
                    dW_hid -= l2_lambda * self.W_hid
                    dW_out -= l2_lambda * self.W_out

                # Learning rate schedule
                if lr_schedule == "constant":
                    alpha_t = alpha_0
                elif lr_schedule == "linear":
                    alpha_t = alpha_0 * (1 - ep / eps)
                elif lr_schedule == "geometric":
                    alpha_t = alpha_0 * (lr_gamma ** ep)
                elif lr_schedule == "step":
                    alpha_t = alpha_0 * (lr_gamma ** (ep // lr_step_size))
                else:
                    raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")

                if use_adam:
                    t = ep + 1

                    # Update moments
                    m_hid = beta1 * m_hid + (1 - beta1) * dW_hid
                    v_hid = beta2 * v_hid + (1 - beta2) * (dW_hid ** 2)
                    m_out = beta1 * m_out + (1 - beta1) * dW_out
                    v_out = beta2 * v_out + (1 - beta2) * (dW_out ** 2)

                    # Bias-corrected moments
                    m_hid_corr = m_hid / (1 - beta1 ** t)
                    v_hid_corr = v_hid / (1 - beta2 ** t)
                    m_out_corr = m_out / (1 - beta1 ** t)
                    v_out_corr = v_out / (1 - beta2 ** t)

                    # Update weights
                    self.W_hid += alpha_t * m_hid_corr / (np.sqrt(v_hid_corr) + epsilon)
                    self.W_out += alpha_t * m_out_corr / (np.sqrt(v_out_corr) + epsilon)

                if use_momentum:
                    dW_hid = momentum_gamma * prev_dW_hid + dW_hid
                    dW_out = momentum_gamma * prev_dW_out + dW_out
                    prev_dW_hid = dW_hid
                    prev_dW_out = dW_out

                self.W_hid += alpha_t * dW_hid
                self.W_out += alpha_t * dW_out

            # Evaluate metrics
            _, _, _, y_full = self.forward(inputs)
            CE = np.mean(onehot_decode(y_full) != labels)
            RE = np.mean(self.error(targets, y_full, l2_lambda=l2_lambda if l2_penalty else 0.0))
            CEs.append(CE)
            REs.append(RE)

            # Validation
            if val_inputs is not None and val_labels is not None:
                val_CE, val_RE = self.test(val_inputs, val_labels)
                val_CEs.append(val_CE)
                val_REs.append(val_RE)

                if early_stopping:
                    if val_CE < best_val_CE:
                        best_val_CE = val_CE
                        best_weights = (self.W_hid.copy(), self.W_out.copy())
                        best_epoch = ep + 1
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            print(f"Early stopping at epoch {ep + 1}")
                            break

            if (ep + 1) % 5 == 0:
                print(f"Epoch {ep + 1:3d}/{eps}, CE = {CE:6.2%}, RE = {RE:.5f}")

            if live_plot and ((ep + 1) % live_plot_interval == 0):
                _, predicted = self.predict(inputs)
                plot_dots(inputs, labels, predicted, block=False)
                plot_both_errors(CEs, REs, block=False)
                plot_areas(self, inputs, block=False)
                redraw()

        # Restore best weights (if early stopping)
        if early_stopping and best_weights is not None:
            self.W_hid, self.W_out = best_weights

        if live_plot:
            interactive_off()

        return CEs, REs, val_CEs, val_REs, best_epoch