# Talk 4: Optimisation and gradient descent (supplementary notes)

- In the talk, we chiefly discuss smooth objective functions. But many objective functions arising from, e.g., neural networks, will only be differentable almost everywhere (e.g., networks using ReLU activation).
- $f$ is coercive if $\lim_{x \to \infty} f(x) = +\infty$.
- In practice, mini-batch stochastic gradient descent is performed by shuffling the data $\{(x_{i_{1}}, y_{i_{1}}), \dots, (x_{i_{N}}, y_{i_{N}})\}$ using some permutation of the indices $\{1, \dots, N\}$, and partitioning into batches of size $B$. But sampling the indices uniformly with replacement (as described in the notes) would also work, and would give an unbiased estimator of the gradient.
- The main motivation for SGD is that, for problems with a large dataset and high-dimensional parameter, a single step of "vanilla" gradient descent may be too costly (in terms of memory) to compute. For example, this would certainly be the case with one-billion neural-network parameters and one-billion data points! 
- For **non-convex** objectives, there is a tradeoff in the batch size: larger batches reduce the variance of the estimator, but some variance can be helpful to escape local minima in the objective function. For convex objectives, this is less of an issue, and so taking the batch size as large as possible may be preferable.


## Theory
- **Essential**: Bach 5.1; from start of 5.2 to start of 5.2.1; 5.2.5; from start of 5.4 to start of 5.4.1.

- **Nice to have**: 5.2.1; 5.4.2; 5.4.4

- **Not so important**: everything else in chapter 5.

## Code

We will revisit the least-squares linear regression example from talk 3.

- Load 1D data set as in that example.
- Instead of using the closed-form expression, we will use gradient descent to find parameters. To do this we will use `jax` (seen briefly in talk 1) to differentiate the objective function and compute gradients; can do this by hand ('vanilla' gradient descent / SGD) and possibly also introduce [`optax`](https://optax.readthedocs.io/en/latest/) library which has more sophisticated stochastic gradient descent algorithms.

Learning `jax` and `optax` will be a useful warm-up for the section on neural networks later.
