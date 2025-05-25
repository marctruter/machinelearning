# Kernel Methods
### [Alvaro Gonzalez Hernandez](https://alvarogohe.github.io/), 27th May 2025

Kernel methods are a powerful set of techniques in machine learning that are used for building non-linear prediction models.

In previous weeks of the study group, we saw how to construct linear models that allow us to find trends in data. However, as we know, most phenomena that machine learning excels at studying, from image recognition to natural language processing, are inherently non-linear.

Kernels provide a way to solve non-linear problems by transforming them into equivalent problems that can be solved using linear methods in high or infinite-dimensional vector spaces.

## Let's start with an example
A very illustrative example is the following:

Suppose that we want to build a model that allows to classify between two types of data and we have the following samples:
<div align="center">
<img src="data.png" alt="Some points in the plane" height="350" >
</div>
It is quite clear that using a line here to separate the data is not going to work well, but it seems like this data could be approximated well by a closed shape, like a circle. Now, here is the trick on how to classify the data. Instead of trying to fit a linear model on the data $\{x_1,x_2\}\in\mathbb{R}^2$, we generate points in $\mathbb{R}^5$ of the form $\{x_1,x_2,x_1^2,x_1x_2,x_2^2\}$ and try to use a hyperplane to separate the data.

Indeed, just by considering $\{x_1^2,x_2^2\}$, we can easily see that the data can be easily separated with a line:

<div align="center">
<img src="classification_lines.png" alt="The points can be separated by a line" height="300" >
</div>

Now, this line corresponds to a conic in the original space, which separates our original data:
<div align="center">
<img src="classification_circles.png" alt="The points can be separated by a line" height="300" >
</div>

In the exercise session, we will see how to implement this in practice using the `sklearn` library. But let's first explain the theory behind this.

## The theory 

As in the past, let $(x_i,y_i)\in\mathcal{X}\times\mathcal{Y}$ be a set of training data, where $\mathcal{X}\subseteq\mathbb{R}^d$ is the input space and $\mathcal{Y}$ is the output space, and our goal is to learn a function $f:\mathcal{X}\to\mathcal{Y}$ that approximates the relationship between the inputs and outputs. For today's talk, we will focus on the case where $\mathcal{Y}=\{-1,1\}$, which is the case of binary classification. As $\mathcal{Y}$ is discrete, what we do instead is that we learn a function $f_\theta:\mathcal{X}\to\mathbb{R}$ and then, we set $f(x)=\text{sign}(f_\theta(x))$. This function $f_\theta$ is called the *prediction function*, and we will assume that it is a linear function on some parameters $\{\theta_1,\dots,\theta_n\}$. Assume, we also have a *loss function* $\ell:\mathcal{Y}\times\mathbb{R}\to\mathbb{R}$ that measures how well the prediction function $f_\theta$ fits the data (for example, the squared loss $\ell(y,f_\theta(x))=(y-f_\theta(x))^2$).

Let us now choose a function $\varphi:\mathcal{C}\rightarrow\mathbb{R}^n$ that maps the input space $\mathcal{X}$ to a different space $\mathbb{R}^n$. This is what we will call the *feature map*.

Going back to our example, there, we defined the prediction function to be

$$\begin{align*}
\mathcal{X}\longrightarrow \mathbb{R}^3\\
(x_1,x_2)&\longmapsto (x_1^2, x_2^2, 1)
\end{align*}$$

and the function we are trying to learn is $f_\theta(x_1,x_2)=\theta_1 x_1^2 + \theta_2 x_2^2+\theta_3$, for some parameters $\{\theta_1,\theta_2,theta_3\}\in\mathbb{R}^3$. If we denote by $(x_i,y_i)$ our data, we can write $f_\theta(x_i)$ in terms of the usual inner product as $f_\theta(x_i)=\langle \varphi(x_i),\theta\rangle$. Therefore, we would like to find the parameters $\theta\in\mathbb{R}^d$ that minimize what is known as the *empirical risk*:

$$





# A remark


