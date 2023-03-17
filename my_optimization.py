import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from datetime import datetime as dt

def gen_A(d, mu, L):
    U = ortho_group.rvs(dim = d)
    A = mu * np.eye(d)
    A[0][0] = L
    A = U.T @ A @ U

    return A

def f(A, b, x):
    return 1/2*(x.T @ A @ x - b.T @ x)

def Get_grad(A, b, x):
    return 1 / 2 * np.matmul((A + A.T), x) - b

def Grad_Optimizer(n_iter, A, lr, b, x_0, batch_size = 1, optimizer = "GD", lr_decrease = False, 
                   a = 1, d = 1000):
    points = []
    x_old = x_0
    for i in range(n_iter):
        grad = Get_grad(A, b, x_old)
        if optimizer == "GD":
            x_new = x_old - lr * grad
        elif optimizer == "SGD":
            ksi_k = np.mean([np.random.normal(0, 10, len(b)) for _ in range(batch_size)])
            if lr_decrease == True:
                i_0 = round(n_iter/2)
                if n_iter < d/a:
                    lr = 1/d
                else:
                    if i < i_0:
                        lr = 1/d
                    else:
                        lr = 2/(a*(2*d/a + i - i_0))
                    
            x_new = x_old - lr * (grad + ksi_k)
            
        x_old = x_new
        points.append(x_old)
    return points

def criteria_points(A, b, points, x_true = None, criteria_type = "||x - x*||"):
    diff_arr = []
    crit_arr = []
    
    for i in range(len(points)):
        if x_true != None:
            diff_arr.append(points[i] - x_true)
            if criteria_type == "||x - x*||":
                crit_arr.append(np.linalg.norm(diff_arr[i]))
            elif criteria_type == "|f - f*|":
                crit_arr.append(np.abs(f(A, b, points[i]) - f(A, b, x_true)))
        if criteria_type == "||grad_f||^2":
            crit_arr.append((np.linalg.norm(Get_grad(A, b, points[i])))**2)
        
    return crit_arr

def plot_graphs(x, y, label, title, logscale = False, criteria_type = "||x - x*||"):
    for i in range(len(y)):
        plt.plot(x, y[i], label = label[i])
    if logscale == True:
        plt.yscale('log')
    plt.ylabel(criteria_type)
    plt.xlabel('n_iter')
    plt.title(title)
    plt.legend()
    plt.show()


def acc_n_iter_dependency(start, finish, step, optimizer, lambda_=None):
    acc_arr, n_arr = [], []
    for n_iter in range(start, finish, step):
        if GD_type == "basic":
            w_true = Grad_Descent(n_iter, X_train, 1 / L, y_train, np.ones(d), np.shape(X_train)[0], optimizer = optimizer)[-1]
        elif GD_type == "l1_ball":
            w_true = Grad_Descent_l1_ball(n_iter, X_train, 1 / L, y_train, np.ones(d), np.shape(X_train)[0], lambda_)[
                -1]

        y_pred = X_test @ w_true
        for i in range(len(y_pred)):
            y_pred[i] = round(y_pred[i])
        acc_arr.append(accuracy_score(y_test, y_pred))
        n_arr.append(n_iter)

    plt.plot(n_arr, acc_arr, label="accuracy")
    plt.xlabel('n_iter')
    plt.ylabel('accuracy')
    plt.title('Dependence of accuracy from n_iter')
    plt.legend()
    plt.show()


def mirror_descent(d, n_iter, X):
    x = np.array([1 / d for _ in range(d)])
    crit_arr = []
    times = []
    t_0 = dt.now()
    gamma = -1 / L

    for i in range(n_iter):
        x = x * np.exp(gamma * A @ x) / (np.sum(x * np.exp(gamma * A @ x)))

        crit_arr.append(gap_criteria(x, X))
        times.append(to_diff(t_0, dt.now()))

    return crit_arr, times

def to_diff(t1, t2):
    delta = t2 - t1
    diff_in_seconds = delta.total_seconds()
    return diff_in_seconds


def frank_wolfe(d, n_iter, X):
    x = np.array([1 / d for _ in range(d)])

    crit_arr = []
    times = []
    t_0 = dt.now()

    for k in range(n_iter):
        s = np.zeros(d)
        s[np.argmin(A @ x)] = 1
        x = x + 2 / (k + 2) * (s - x)

        crit_arr.append(gap_criteria(x, X))
        times.append(to_diff(t_0, dt.now()))

    return crit_arr, times