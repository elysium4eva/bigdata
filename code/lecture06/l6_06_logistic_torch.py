"""l6_06_logistic_torch.py -- the same logistic regression through PyTorch autograd.

Every line carries an English comment. The script fits logistic regression twice on
the same data: once with a hand-written analytic gradient in NumPy and once with
PyTorch's automatic differentiation, then checks that the gradients and the fitted
coefficients agree.
"""
import numpy as np                                                 # NumPy for the reference implementation
import pandas as pd                                                # tabular data handling
import torch                                                       # automatic differentiation
import torch.nn as nn                                              # loss modules
from sklearn.linear_model import LogisticRegression                # independent reference fit
from _common_l6 import DAT, dump, sigmoid, add_intercept           # shared helpers

torch.manual_seed(20260603)                                        # torch seed for reproducibility

# ---------------------------------------------------------------- 1) data
data = pd.read_csv(DAT / "l6_two_class.csv")                       # the two-class sample from lesson 6.5
X_np = add_intercept(data[["x"]].values)                           # augmented design matrix with the bias column
y_np = data["y"].values.astype(float)                              # labels as floats for the loss
Xt = torch.tensor(X_np, dtype=torch.float64)                       # torch version of the design matrix
yt = torch.tensor(y_np, dtype=torch.float64)                       # torch version of the labels
print("rows:", X_np.shape[0], "| parameters:", X_np.shape[1])      # report the setup

# ---------------------------------------------------------------- 2) analytic gradient in NumPy
def loss_grad_np(w):                                               # cross-entropy and its gradient
    """Return (loss, gradient) for the logistic model at w."""      # short docstring
    p = sigmoid(X_np @ w)                                          # predicted probabilities
    eps = 1e-12                                                    # avoid log(0)
    loss = -np.mean(y_np * np.log(p + eps) + (1 - y_np) * np.log(1 - p + eps))   # mean cross-entropy
    grad = X_np.T @ (p - y_np) / len(y_np)                         # analytic gradient X^T (p - y) / n
    return float(loss), grad                                       # return both

w_np = np.zeros(X_np.shape[1])                                     # start from zeros
hist_np = []                                                       # loss history
for _ in range(800):                                               # full-batch gradient descent
    loss, grad = loss_grad_np(w_np)                                # evaluate loss and gradient
    hist_np.append(loss)                                           # record the loss
    w_np = w_np - 2.0 * grad                                       # gradient step with lr = 2.0
print("numpy  w:", np.round(w_np, 4), "| final loss %.4f" % hist_np[-1])   # report the numpy fit

# ---------------------------------------------------------------- 3) the same thing with autograd
w_t = torch.zeros(X_np.shape[1], dtype=torch.float64, requires_grad=True)   # parameter with gradients enabled
opt = torch.optim.SGD([w_t], lr=2.0)                               # plain stochastic gradient descent
bce = nn.BCEWithLogitsLoss()                                       # numerically stable logistic loss
hist_t = []                                                        # loss history
for _ in range(800):                                               # same number of epochs
    opt.zero_grad()                                                # clear the previous gradients
    logits = Xt @ w_t                                              # linear score
    loss = bce(logits, yt)                                         # loss computed in a stable way
    loss.backward()                                                # autograd fills w_t.grad
    opt.step()                                                     # apply the update
    hist_t.append(float(loss.item()))                              # record the loss
print("torch  w:", np.round(w_t.detach().numpy(), 4), "| final loss %.4f" % hist_t[-1])   # report the torch fit

# ---------------------------------------------------------------- 4) gradient agreement check
Xb = Xt[:20]                                                       # a small batch for the gradient check
yb = yt[:20]                                                       # matching labels
w_check = torch.tensor([0.1, 0.4], dtype=torch.float64, requires_grad=True)   # an arbitrary parameter point
loss_check = nn.functional.binary_cross_entropy_with_logits(Xb @ w_check, yb)   # loss at that point
loss_check.backward()                                              # autograd gradient
grad_torch = w_check.grad.numpy()                                  # copy the autograd gradient
_, grad_np = loss_grad_np(w_np)                                    # analytic gradient at the numpy optimum
p_b = sigmoid(Xb.detach().numpy() @ w_check.detach().numpy())      # probabilities for the same batch
grad_np_b = Xb.detach().numpy().T @ (p_b - yb.numpy()) / len(yb)    # analytic gradient for that batch
print("gradient gap (batch): %.2e" % float(np.max(np.abs(grad_torch - grad_np_b))))   # must be near zero
assert np.allclose(grad_torch, grad_np_b, atol=1e-10), "autograd and analytic gradients disagree"   # hard check

# ---------------------------------------------------------------- 5) independent sklearn fit
sk = LogisticRegression(max_iter=1000).fit(data[["x"]].values, y_np)   # sklearn reference on the raw feature
w_sk = np.array([float(sk.intercept_[0]), float(sk.coef_[0][0])])   # sklearn coefficients (bias, slope)
print("sklearn w:", np.round(w_sk, 4))                             # report the reference fit

# ---------------------------------------------------------------- 6) artefacts and figure
dump(pd.DataFrame({"parameter": ["bias", "slope"],                 # parameter names
                   "numpy_gd": np.round(w_np, 6),                  # hand-written gradient descent
                   "torch_autograd": np.round(w_t.detach().numpy(), 6),   # PyTorch autograd
                   "sklearn": np.round(w_sk, 6)}),                 # independent reference
     "l6_grad_compare.csv")                                        # persist the comparison
dump(pd.DataFrame({"epoch": np.arange(1, len(hist_np) + 1),        # epoch index
                   "numpy_loss": np.round(hist_np, 6),             # numpy loss curve
                   "torch_loss": np.round(hist_t, 6)}),            # torch loss curve
     "l6_torch_history.csv")                                       # persist the loss histories
print("done: autograd matches the analytic gradient")              # closing log
