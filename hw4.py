import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)
x_min=-3.
x_max=3.
n_iter=20

def generate_sample(x_min=-3., x_max=3., sample_size=10):
    x = np.linspace(x_min, x_max, num=sample_size)
    y = x + np.random.normal(loc=0., scale=.2, size=sample_size)
    y[-1] = y[-2] = y[-3] = -4  # outliers
    return x, y


def build_design_matrix(x):
    phi = np.empty(x.shape + (2,))
    phi[:, 0] = 1.
    phi[:, 1] = x
    return phi


def iterative_reweighted_least_squares(x, y, eta=1., n_iter=20):
    phi = build_design_matrix(x)
    # initialize theta using the solution of regularized ridge regression
    theta = theta_prev = np.linalg.solve(
        phi.T.dot(phi) + 1e-4 * np.identity(phi.shape[1]), phi.T.dot(y))
    fig1=plt.figure(figsize=(12,5))
    fig1.subplots_adjust(hspace=0.6, wspace=0.6)

    error_transition_hu=[]
    error_transition_tu=[]

    for _ in range(n_iter):
        r = np.abs(phi.dot(theta_prev) - y)
        w = np.diag(np.where(r > eta, eta / r, 1.))
        phit_w_phi = phi.T.dot(w).dot(phi)
        phit_w_y = phi.T.dot(w).dot(y)
        theta = np.linalg.solve(phit_w_phi, phit_w_y)
        theta_prev = theta
        error_transition_hu.append(np.sum((w/2).dot(r**2)))

    ax1=fig1.add_subplot(1,2,1)
    ax1.scatter(x, y, c='blue', marker='o')
    ax1.set_title('Huber Loss')
    X = np.linspace(x_min, x_max)
    Y = predict(X, theta)
    ax1.plot(X, Y, color='green')

    theta = theta_prev = np.linalg.solve(
        phi.T.dot(phi) + 1e-4 * np.identity(phi.shape[1]), phi.T.dot(y))
    for _ in range(n_iter):
        r = np.abs(phi.dot(theta_prev) - y)
        w = np.diag(np.where(r > eta, 0, (1-r**2/eta**2)**2))
        phit_w_phi = phi.T.dot(w).dot(phi)
        phit_w_y = phi.T.dot(w).dot(y)
        theta = np.linalg.solve(phit_w_phi, phit_w_y)
        theta_prev = theta
        error_transition_tu.append(np.sum((w/2).dot(r**2)))
    ax2=fig1.add_subplot(1,2,2)
    ax2.scatter(x, y, c='blue', marker='o')
    ax2.set_title('Tukey Loss')
    X = np.linspace(x_min, x_max)
    Y = predict(X, theta)
    ax2.plot(X, Y, color='green')
    plt.show()

    plt.figure()
    x = np.linspace(0,n_iter,20)
    plt.plot(x,error_transition_hu,color="red",label="Hubor")
    plt.plot(x,error_transition_tu,label="Tukey")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss transition')
    plt.legend(loc="best")
    plt.show()
    return theta

def predict(x, theta):
    phi = build_design_matrix(x)
    return phi.dot(theta)


x, y = generate_sample()
theta = iterative_reweighted_least_squares(x, y, eta=1.)

