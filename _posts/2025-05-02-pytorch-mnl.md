---
layout: post
title:  "Significance test of multinomial logit model coefficients in PyTorch"
comments: true
tags: Note
use_math: true
toc: true
views: true
# modified:
---

## Why not use an existing packages?
In my recent work on using graph neural networks (GNN) in DCM, the idea of implementing the multinomial logit (MNL) model in PyTorch with the functionality of significance testing has arisen from time to time. Although there have been many mature packages for MNL, including the excellent [Biogeme](https://biogeme.epfl.ch/), [Apollo](https://www.apollochoicemodelling.com/), [PyLogit](https://github.com/timothyb0912/pylogit), and [Choice-Learn](https://github.com/artefactory/choice-learn), a minimalistic implementation of MNL in PyTorch is still useful for **educational purposes** and for those who want to **keep a consistent data structure and workflow in PyTorch**. In addition, I found that doing it from scratch is not much more difficult than adapting to a new package, because PyTorch has resolved the most difficult part of automatic differentiation.

In this post, I will use PyTorch to replicate the [Swissmetro example](https://biogeme.epfl.ch/sphinx/auto_examples/swissmetro/plot_b01logit.html#sphx-glr-auto-examples-swissmetro-plot-b01logit-py) in Biogeme[^1]. Next, some theory and code for the significance testing of coefficients. Lastly, this post can be summarized into a [plug-and-play module](https://github.com/chengzhanhong/archive_notes/blob/main/MNL_pytorch_general.py) for significance tests in MNL and other similar regression models estimated by maximum likelihood estimation (MLE).

## Load Swissmetro dataset
Load the [Swissmetro dataset](https://transp-or.epfl.ch/pythonbiogeme/examples/swissmetro/swissmetro.pdf), where the participants were asked to choose between Swissmetro, train, and car.

```python
import pandas as pd

df = pd.read_table(
    "https://courses.edx.org/asset-v1:EPFLx+ChoiceModels2x+3T2021+type@asset+block@swissmetro.dat",
    sep="\t",
)

# Select relevant samples
df = df.loc[
    df.PURPOSE.isin([1, 3])  # Only keep 1: Commuter, 3: Business
    & (df.CHOICE > 0)  # Only keep samples with a valid choice
    & (df.CAR_AV == 1),  # Only keep samples with car availability
    :]

# With GA (annual ticket), the costs of Swissmetro and train are 0.
df.loc[df.GA == 1, "SM_CO"] = 0
df.loc[df.GA == 1, "TRAIN_CO"] = 0
# Normalize the travel time and cost
df[["TRAIN_TT", "TRAIN_CO", "SM_TT", "SM_CO", "CAR_TT", "CAR_CO"]] /= 100
```

Transform the input features and choice results into PyTorch tensors. Unlike the many machine learning models that use mini-batch training, the correct significance test requires zero gradient at the solution point, thus, the full dataset is used to estimate the model parameters. In PyTorch, this is done by using the entire dataset at once.


```python
import torch
# Using the travel time (TT) and cost (CO) of the three alternatives as input features:
x = df[["TRAIN_TT", "TRAIN_CO", "SM_TT", "SM_CO", "CAR_TT", "CAR_CO"]].values
x = torch.tensor(
    x.reshape([-1, 3, 2]), dtype=torch.float32
)  # (batch_size, num_choices, num_features)

y = df.CHOICE.values  # Choice is 1, 2, or 3 for the three alternatives
y = torch.tensor(y, dtype=torch.long) - 1  # In Python, index start at 0
```

## MNL in PyTorch
Define the following MNL model in PyTorch:

$$
\begin{align*}
V_1 &= \beta_1 \text{TRAIN_TT} + \beta_2 \text{TRAIN_CO} + ASC_1 \\
V_2 &= \beta_1 \text{SM_TT} + \beta_2 \text{SM_CO} + ASC_2 \\
V_3 &= \beta_1 \text{CAR_TT} + \beta_2 \text{CAR_CO} \\
P_i &= \frac{e^{V_i}}{\sum_{j=1}^{3}e^{V_j}} \\
\end{align*}
$$

The same $\beta_1$ and $\beta_2$ are used for all alternatives. The alternative specific constants (ASC) for TRAIN and SM are included in the model, but not for CAR (for model identification purposes).


```python
import torch.nn as nn
import torch.nn.functional as F

class MNL(nn.Module):
    def __init__(self, num_features, num_choices):
        super(MNL, self).__init__()
        self.beta = nn.Parameter(torch.randn(num_features), requires_grad=True)
        self.asc = nn.Parameter(torch.zeros(num_choices - 1), requires_grad=True)

    def forward(self, data):
        # data: (batch_size, num_choices, num_features)
        x = data @ self.beta  # (batch_size, num_choices)
        x[:, :-1] += self.asc  # Add ASC to TRAIN and SM, but not to CAR
        return F.log_softmax(x, dim=-1)  # (batch_size, num_choices)


model = MNL(num_features=2, num_choices=3)
```

The `F.log_softmax` is equivalent to `log(softmax(x))`, but if faster and more stable than the latter.

## Model estimation

The [LBFGS optimizer](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html) is used to maximize the log-likelihood function (minimize the [nll_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html)).


```python
optimizer = torch.optim.LBFGS(model.parameters())
# Estimate the model parameters using the training data by L-BFGS optimization
def closure():
    optimizer.zero_grad()
    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()
    return loss

for _ in range(100):
    optimizer.step(closure)
# Print the estimated parameters
print("Estimated parameters:")
print("Beta:", model.beta.data.numpy())
print("Asc:", model.asc.data.numpy())
print("log-likelihood:", -F.nll_loss(model(x), y, reduction='sum').item())
```

    Estimated parameters:
    Beta: [-1.2727097 -1.1552944]
    Asc: [-0.91749597  0.2504263 ]
    log-likelihood: -4382.490234375

Both $\beta_1$ and $\beta_2$ are negative, meaning that longer travel time and higher cost reduce the probability of choosing the alternative, which is expected.

## Significance test

### Standard errors
The theory of significance test in maximum likelihood estimation (MLE) can be found [here](https://discdown.org/microeconometrics/maximum-likelihood-estimation-1.html) and [Section 8.6](https://eml.berkeley.edu/books/choice2nd/Ch08_p183-204.pdf#page=18.33) of the book by Dr. Kenneth Train. I give an intuitive but not rigorous explanation. In MLE, the **Hessian matrix** (second derivative of the log-likelihood function)

$$
H(\hat{\beta}) = \frac{\partial^2 \mathcal{L}(\hat{\beta})}{\partial \hat{\beta}_i \partial \hat{\beta}_j}
$$

measures how “curved” the log-likelihood surface is around the estimated parameters. A steeper curve (i.e., higher curvature) along a parameter direction indicates that the log-likelihood changes more quickly with respect to that parameter, suggesting that the parameter is more important, the estimate is more certain, and has a smaller variance for that parameter. Conversely, if the curve is flat, it suggests that there’s more uncertainty about the parameter value, meaning the variance is larger.

Therefore, the significance test of the estimated parameters is based on the **Hessian matrix** at the solution point. The steps are as follows:
1. The **inverse of the negative Hessian**[^2] is the **covariance matrix** of the estimated parameters.
2. The **diagonal elements** of this covariance matrix give the variances of individual coefficients.
3. Taking the square root of these variances gives the **standard errors**.
4. Using the standard errors, we can compute the **t-statistics** $t_i = \hat{\beta}_i/\text{SE}(\hat{\beta}_i)$ and p-value for each coefficient.


```python
# PyTorch's functional API allows us to compute the Hessian of a function
# Therefore, we define a loss function that computes the NLL given the model parameters.
def loss_all(params):
    param_dict = {'beta': params[:2], 'asc': params[2:]}
    # Reconstruct the `model` with the parameters in `param_dict``, and apply it to input `x`
    log_probs = torch.func.functional_call(model, param_dict, x)
    nll = F.nll_loss(log_probs, y, reduction='sum')
    return nll


params = torch.cat([model.beta, model.asc])
# This gives the negative of the Hessian, as we minimize the NLL
H = torch.autograd.functional.hessian(loss_all, params)
H_inv = torch.inverse(H)  # Invert the Hessian matrix to get the covariance matrix
std_err = torch.sqrt(torch.diag(H_inv)).detach().numpy()  # Standard errors of the parameters


def significance_test(params, std_err, param_names):
    from scipy.stats import t
    t_values = params / std_err
    p_values = 2 * (
        1 - t.cdf(abs(t_values), df=len(x) - len(params))
    )  # p-value of a t-distribution with degrees of freedom = num_samples - num_params

    # organize the results in a pandas DataFrame
    results = {
        "params": params,
        "std_err": std_err,
        "t_values": t_values,
        "p_values": p_values,
    }

    return pd.DataFrame(results, index=param_names)

print(significance_test(params.detach().numpy(),
                        std_err,
                        ['beta_1', 'beta_2', 'asc_1', 'asc_2']))
```

              params   std_err   t_values      p_values
    beta_1 -1.272701  0.060907 -20.895693  0.000000e+00
    beta_2 -1.155296  0.053164 -21.730906  0.000000e+00
    asc_1  -0.917466  0.056707 -16.179033  0.000000e+00
    asc_2   0.250493  0.044582   5.618650  2.017313e-08


### Robust standard errors
The above standard error estimation is based on the assumption that the model is correctly specified and the errors are homoscedastic (constant variance). [PyLogit](https://github.com/timothyb0912/pylogit) and [Choice-Learn v1.1.0](https://github.com/artefactory/choice-learn) are based on this implementation. In practice, it is better to use **robust standard errors** to account for potential heteroscedasticity and model misspecification, as supported by [Biogeme](https://biogeme.epfl.ch/) and [Apollo](https://www.apollochoicemodelling.com/).

The robust standard errors are computed using heteroscedasticity consistent (HC) covariances:

$$
\text{Cov}(\hat{\beta}) = H^{-1} B H^{-1},
$$

where $H$ is the Hessian matrix at the solution point, and $B$ is outer product of individual scores (the gradient of the log-likelihood):

$$
B = \sum_{i=1}^{N} \left(\frac{\partial \mathcal{L}(\hat{\beta};x_i, y_i)}{\hat{\beta}}\right) \left(\frac{\partial \mathcal{L}(\hat{\beta};x_i, y_i)}{\hat{\beta}}\right)^T,
$$

where $\mathcal{L}(\hat{\beta};x_i, y_i)$ is the log-likelihood function for the $i$-th observation. This **sandwich estimator** can be readily computed in PyTorch.



```python
def loss_each(params):
    param_dict = {"beta": params[:2], "asc": params[2:]}
    log_probs = torch.func.functional_call(model, param_dict, x)
    nll = F.nll_loss(log_probs, y, reduction="none")
    return nll

jacobian = torch.autograd.functional.jacobian(loss_each, params) # (batch_size, num_params)

B = jacobian.T @ jacobian  # (n_params, n_params)
COV = H_inv @ B @ H_inv
robust_std_err = torch.sqrt(torch.diag(COV)).detach().numpy()
print(
    significance_test(
        params.detach().numpy(), robust_std_err, ["beta_1", "beta_2", "asc_1", "asc_2"]
    )
)
```

              params   std_err   t_values  p_values
    beta_1 -1.272701  0.117085 -10.869920  0.000000
    beta_2 -1.155296  0.071941 -16.059046  0.000000
    asc_1  -0.917466  0.063455 -14.458471  0.000000
    asc_2   0.250493  0.062681   3.996299  0.000065

## Summary
We can see that implementing MNL in PyTorch is not difficult. The notebook of this post is available [here](https://github.com/chengzhanhong/archive_notes/blob/main/MNL_pytorch_example.ipynb), and a more general module for significance test in MNL and other regression models estimated by maximum likelihood estimation (MLE) is available on [here](https://github.com/chengzhanhong/archive_notes/blob/main/MNL_pytorch_general.py).

[^1]: There are minor differences from the Biogeme example. 1. The Biogeme example has an availability condition for each alternative. For simplicity, I only use samples with car ownership; thus, everyone chooses from the three alternatives. 2. The Biogeme example adds alternative specific constants (ASC) to car and train, but here the ASC is added to train and Swissmetro.
[^2]: The log-likelihood is concave, and a negative sign must be used to ensure the Hessian is positive definite. This negative sign is not required if we use the Hessian of the **negative log-likelihood**, as shown in our implementation.