#Imports
import numpy as np
import matplotlib.pyplot as plt
#from scipy.stats import ortho_group
from datetime import datetime as dt
from numpy.linalg import norm
import time
#from sklearn.metrics import mean_squared_error, accuracy_score

#Class of Gradient Methods Optimizers
class GradientOptimizer:
    def __init__(self, f, grad_f, x_0, gamma_k, args, n_iter = 1000, n = 1, criterium = '||x_k - x^*||', 
                 eps = 1e-8, x_true = None, sgd_activate = False, batch_size = 1, svrg_activate = False, 
                 sarah_activate = False, csgd_activate = False, grad_f_j = None, is_independent = False, 
                 n_coord = 1, sega_activate = False, n_workers = 1, top_k_activate = False, 
                 rand_k_activate = False, ef_activate = False, top_k_param = None, rand_k_param = None):
        '''
        :parameter f: target function
        :parameter grad_f: target function gradient
        :parameter x_0: starting point
        :parameter gamma_k: learning rate function depending on the number of current iteration k
        :parameter n_iter: number of iterations
        :parameter n: number of workers (functions to optimize)
        :parameter args: includes parameters of functions
        :parameter criterium: criterium of convergence, options: '||x_k - x^*||', '|f(x_k) - f(x^*)|', '||grad_f(x_k)||'
        :parameter eps: target difference from x_k and x^*
        :parameter x_true: trueoptimum
        :parameter sgd_activate: activate sgd
        :parameter batch_size: number of batches (by default equals to 1)
        :parameter svrg_activate: activate svrg
        :parameter sarah_activate: activate sarah
        :parameter csgd_activate: activate csgd
        :parameter is_independent: use the independent coordinates in csgd if True
        :parameter grad_f_j: the j-th coordinate of a gradient        
        :parameter n_coord: the number of coordinates left in the csgd
        :parameter sega_activate: activate sega
        :parameter n_workers: number of workers in distributed optimization
        :parameter top_k_activate: activate top_k compressor
        :parameter rand_k_activate: activate rand_k compressor
        :parameter ef_activate: activate error feedback in compressor
        '''
        
        self.f = f
        self.grad_f = grad_f
        self.x_0 = x_0
        self.gamma_k = gamma_k
        self.args = args
        self.n_iter = n_iter
        self.n = n
        self.eps = eps
        self.x_true = x_true
        self.criterium = criterium
        self.sgd_activate = sgd_activate
        self.svrg_activate = svrg_activate
        self.sarah_activate = sarah_activate
        self.csgd_activate = csgd_activate
        self.grad_f_j = grad_f_j
        self.is_independent = is_independent
        self.n_coord = n_coord
        self.sega_activate = sega_activate
        self.n_workers = n_workers
        self.top_k_activate = top_k_activate
        self.top_k_param = top_k_param
        self.rand_k_activate = rand_k_activate
        self.rand_k_param = rand_k_param
        self.ef_activate = ef_activate

    #---COMPRESSORS------------------------------------------------------------------------------------------

    def grad_list_compressor(self, x_k):
        '''
        Realizes compression of a gradient 
        '''
        grad_list = self.grad_f(x_k, self.args)

        if self.top_k_activate is True:
            for i in range(self.n_workers):
                grad_list[i] = GradientOptimizer.top_k_compressor(self, grad_list[i])
        elif self.rand_k_activate is True:
            for i in range(self.n_workers):
                grad_list[i] = GradientOptimizer.rand_k_compressor(self, grad_list[i])
        else:
            pass

        return grad_list
            
        def top_k_compressor(self, grad):
            '''
            top_k compressor
            '''
            assert self.top_k_param is not None
            grad = np.array(grad)  # Convert grad to a numpy array
            compressed_grad = np.zeros(len(grad))
            grad_abs = np.abs(grad)
            top_indices = np.argsort(grad_abs)[-abs(self.top_k_param):]
            compressed_grad[top_indices] = grad[top_indices]
            return compressed_grad
    
    def rand_k_compressor(self, grad):
        '''
        rand_k compressor
        '''
        assert self.rand_k_param is not None
        compressed_grad = np.zeros(len(grad))

        all_indices = np.arange(len(grad))
        rand_indices = np.random.choice(all_indices, self.rand_k_param, replace=False)

        compressed_grad[rand_indices] = grad[rand_indices]
        return compressed_grad
    
    #---OPTIMIZATION ALGORITHMS' STEPS------------------------------------------------------------------------------------------

    def gd_step(self, x_k, k):
        '''
        Basic Gradient Descent step
        '''
        compressed_grad_list = GradientOptimizer.grad_list_compressor(self, x_k)

        master_grad = np.zeros(len(compressed_grad_list[0]))
        for i in range(self.n_workers):
            master_grad += compressed_grad_list[i]
        master_grad /= (2*self.n_workers) #IDK WHY 2n BUT IT IS IN HW7 TASK 1

        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        return x_k - gamma * master_grad
    
    def sgd_step(self, x_k, k):
        '''
        Stochastic Gradient Descent step
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        ksi_k = np.mean([np.random.normal(0, 10, len(x_k)) for _ in range(batch_size)])
        
        return x_k - gamma * (self.grad_f(x_k, self.args) + ksi_k)
    
    def csgd_step(self, x_k, k):
        '''
        Coordinate Stochastic Gradient Descent step
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        grad = np.zeros(x_k.shape[0])
     
        if self.is_independent is False:  
            for i in range(self.n_coord):
                j = np.random.randint(self.args['d'])
                grad[j] = self.grad_f_j(x_k, j, self.args).real
        else:
            s = set(range(self.args['d']))
            for i in range(self.n_coord):
                j = np.random.choice(list(s))
                grad[j] = self.grad_f_j(x_k, j, self.args).real
                s.discard(j)
                
        return x_k - gamma * grad

    
    def sega_step(self, x_k, h_k, k):
        '''
        SEGA Algorithm step
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)

        e_j = np.zeros(len(x_k)) 
        j = np.random.randint(self.args['d'])
        e_j[j] = 1

        grad_j = self.grad_f_j(x_k, j, self.args).real
        
        g_k = self.args['d'] * e_j * (grad_j - h_k[j]) + h_k
        h_k_new = h_k + e_j * (grad_j - h_k[j])
        
        return x_k - gamma*g_k, h_k_new
    
    #---DESCENT------------------------------------------------------------------------------------------
           
    def descent(self):
        '''
        This function realizes the descent to the optimum using one of the gradient-based methods
        Note that it is a master node
        '''
        x_k = np.copy(self.x_0)
        #g_k = self.grad_f(x_k, self.args)
        h_k = self.grad_f(self.x_0, self.args)

        #phi_k = np.array([self.x_0] * self.n)
        t_start = time.time()
        
        times_arr = []
        
        differences_arr = []
        points_arr = []
        
        for k in range(self.n_iter):     
            if self.sgd_activate is True:
                x_k = GradientOptimizer.sgd_step(self, x_k, k)
            elif self.csgd_activate is True:
                x_k = GradientOptimizer.csgd_step(self, x_k, k)
            elif self.sega_activate is True:
                x_k = GradientOptimizer.sega_step(self, x_k, h_k, k)
            else:
                x_k = GradientOptimizer.gd_step(self, x_k, k)
                
            points_arr.append(x_k)
            
            t_stop = time.time()
            if self.criterium == '||x_k - x^*||':
                differences_arr.append(norm(x_k - self.x_true, ord = 2))
            elif self.criterium == '|f(x_k) - f(x^*)|':
                differences_arr.append(self.f(x_k, self.args) - self.f(self.x_true, self.args))
            elif self.criterium == '||grad_f(x_k)||':
                differences_arr.append(norm(self.grad_f(x_k, self.args), ord = 2))
            else:
                AssertionError
                
            t_current = time.time()
            times_arr.append(t_current - t_start)

            '''                   
            if differences_arr[-1] <= self.eps:
                break
            '''
                
        return points_arr, differences_arr, times_arr

#Plot Graphs
def plot_graphs(x, y, x_label, lines_labels, title, logscale = False, specific_slice = False, criteria_type = "||x - x*||"):
    if specific_slice == False:
        specific_slice = range(len(y))
    
    plt.figure(figsize=(8, 8))
    for i in specific_slice:
        plt.plot(x[i], y[i], label = lines_labels[i])
    if logscale == True:
        plt.yscale('log')
    plt.ylabel(criteria_type)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()        
        
#Matrix Generation                
def gen_A(d, mu, L):
    #U = ortho_group.rvs(dim = d)
    A = mu * np.eye(d)
    A[0][0] = L
    #A = U.T @ A @ U

    return A

#Quadratic Function 
def f_quad(x, args):
    return 1/2*(x.T @ args['A'] @ x - args['b'].T @ x)

def f_quad_grad(x, args):
    return args['A'] @ x - args['b']

def f_quad_grad_j(x, j, args):
    return args['A'][j] @ x - args['b'][j]    

def SAGA(n_iter, lr, R, f_arr, nodes_count, x_0):
    arr_n = [i for i in range(nodes_count)]
    phi = [x_0]*nodes_count
    points = []
    x_old = x_0
    
    for k in range(n_iter):
        avg_grad = 0
        for i in range(nodes_count):
            np.sum(get_grad(f_arr[i], phi[i]))
        avg_grad /= nodes_count
        
        j = np.random.randint(nodes_count)
        phi_j_old = phi[j]
        phi[j] = x_old
             
        #g = get_autograd(f_arr[j], phi[j]) - get_autograd(f_arr[j], phi_j_old) + avg_grad
        
        x_new = prox_l2(x_old - lr*g, lr)
        x_old = x_new


def SVRG(n_iter, A, lr, b, x_0):
    pass    
'''
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
'''

#---------------------------------------------HW 3-----------------------------------------------------

def acc_n_iter_dependency(start, finish, step, optimizer, lambda_=None):
    acc_arr, n_arr = [], []
    for n_iter in range(start, finish, step):
        if GD_type == "basic":
            w_true = Grad_Descent_Logloss(n_iter, X_train, 1 / L, y_train, np.ones(d), np.shape(X_train)[0], optimizer = optimizer)[-1]
        elif GD_type == "l1_ball":
            w_true = Grad_Descent_l1_ball(n_iter, X_train, 1 / L, y_train, np.ones(d), np.shape(X_train)[0], lambda_)[-1]

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

def Logloss(w, X, y, n):
    ans = 0
    for i in range(n):
        ans += np.log(1 + np.exp(-(w @ X[i]) * y[i]))
    return ans / n

def Get_grad_Logloss (w, X, y, n):
    ans = np.zeros(w.size)
    for i in range(n):
        ans += - y[i] * X[i] * np.exp(-w.dot(X[i]) * y[i]) / (1 + np.exp(- w.dot(X[i]) * y[i]))
    return ans / n

def Grad_Descent_LogReg(n_iter, X, lr, y, w_0, n):
    points = []
    w_old = w_0
    for i in range(n_iter):
        grad = Get_grad_Logloss(X, y, w_old, n)
        
        w_new = w_old - lr*grad
        w_old = w_new 
        points.append(w_old)
    return points
             
            
#---------------------------------------------HW 4-----------------------------------------------------
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
