#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from datetime import datetime as dt
from numpy.linalg import norm
import time
from sklearn.metrics import mean_squared_error, accuracy_score
from tqdm import tqdm

#Class of Gradient Methods Optimizers
class GradientOptimizer:
    def __init__(self, f, grad_f, x_0, gamma_k, args, n_iter = 1000, n = 1, criterium = '||x_k - x^*||', 
                 y_lim = 1e-8, x_true = None, sgd_activate = False, batch_size = 1, svrg_activate = False, 
                 sarah_activate = False, csgd_activate = False, grad_f_j = None, is_independent = False, 
                 n_coord = 1, sega_activate = False, n_workers = 1, top_k_activate = False, 
                 rand_k_activate = False, ef_activate = False, top_k_param = 0, rand_k_param = 0, 
                 diana_activate = False, acc_k = None, upper_limit = 1e10, ef21_activate = False, marina_activate = False,
                 p_marina = 0.5, momentum_gd_activate = False, nesterov_momentum_activate = False, momentum_coeff_k = 0,
                 restart_activate = False):
        '''
        :parameter f: target function
        :parameter grad_f: target function gradient
        :parameter x_0: starting point
        :parameter gamma_k: learning rate function depending on the number of current iteration k
        :parameter n_iter: number of iterations
        :parameter n: number of workers (functions to optimize)
        :parameter args: includes parameters of functions
        :parameter criterium: criterium of convergence, options: '||x_k - x^*||', '|f(x_k) - f(x^*)|', '||grad_f(x_k)||'
        :parameter y_lim: target difference from x_k and x^*
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
        :parameter diana_activate: activate diana algorithm
        :parameter acc_k: calculate accuracy on each step depending on the function
        :parameter upper_limit: upper_limit on criterium
        :parameter ef21_activate: activate ef21
        :parameter marina_activate: activate marina
        :parameter p_marina: probability of transmitting the whole gradient (not compressed one)
        HW3
        :parameter momentum_gd_activate: activate momentum gradient descent
        :parameter nesterov_momentum_activate: activate nesterov momentum algorithm
        :parameter momentum_coeff_k: the coefficient in front of the momentum 
        :parameter restart_activate: activate restart method in accelerated method
        '''
        
        self.f = f
        self.grad_f = grad_f
        self.x_0 = x_0
        self.gamma_k = gamma_k
        self.args = args
        self.n_iter = n_iter
        self.n = n
        self.y_lim = y_lim
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
        self.diana_activate = diana_activate
        self.acc_k = acc_k
        self.upper_limit = upper_limit
        self.ef21_activate = ef21_activate
        self.marina_activate = marina_activate
        self.p_marina = p_marina
        #HW3
        self.momentum_gd_activate = momentum_gd_activate
        self.nesterov_momentum_activate = nesterov_momentum_activate
        self.momentum_coeff_k = momentum_coeff_k
        self.restart_activate = restart_activate

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
        assert self.top_k_param > 0

        compressed_grad = np.zeros(len(grad))
        top_indices = np.argsort(np.abs(grad))[-self.top_k_param:]
        compressed_grad[top_indices] = grad[top_indices]

        return compressed_grad
    
    def rand_k_compressor(self, grad):
        '''
        rand_k compressor
        '''
        assert self.rand_k_param > 0
        compressed_grad = np.zeros(len(grad))

        all_indices = np.arange(len(grad))
        rand_indices = np.random.choice(all_indices, self.rand_k_param, replace=False)

        compressed_grad[rand_indices] = grad[rand_indices]
        return compressed_grad
    
    #---OPTIMIZATION ALGORITHMS------------------------------------------------------------------------------------------

    def gd_step(self, x_k, k):
        '''
        Basic Gradient Descent step
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        if self.n_workers > 1:
            compressed_grad_list = GradientOptimizer.grad_list_compressor(self, x_k)

            master_grad = np.zeros_like(x_k)
            for i in range(self.n_workers):
                master_grad += compressed_grad_list[i]
            master_grad /= (self.n_workers) #IDK WHY 2n BUT IT IS IN HW7 TASK 1
            
            return x_k - gamma * master_grad
        else:
            grad = self.grad_f(x_k, self.args)
            return x_k - gamma * grad

    
    #---HW3----------------------------------------------------------------------------------------

    def momentum_gd_step(self, x_k, k, x_prev):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        momentum_coeff_k = self.momentum_coeff_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        grad = self.grad_f(x_k, self.args)

        return x_k - gamma * grad + momentum_coeff_k * (x_k - x_prev), x_k
    
    def nesterov_momentum_step(self, x_k, k, x_prev):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        momentum_coeff_k = self.momentum_coeff_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
   
        y_k = x_k + momentum_coeff_k*(x_k - x_prev)

        x_new = y_k - gamma * self.grad_f(y_k, self.args)

        return x_new, x_k
    
    #---Bonus--------------------------------------------------------------------------------------

    def restart_step(self, x_k, k, y_k, theta_k):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        q = self.args['q'] #\in [0; 1]

        x_new = y_k - gamma * self.grad_f(y_k, self.args)

        #solving quadratic equation theta_new^2 + theta_new * (theta_k^2 - q) - theta_k^2 = 0
        D = (theta_k**2 - q)**2 + 4*theta_k**2
        theta_new = (-1 * (theta_k**2 - q) + np.sqrt(D))/ 2

        beta_new = theta_k * (1 - theta_k) / (theta_k**2 + theta_new)

        y_new = x_new + beta_new * (x_new - x_k)

        return x_new, y_new, theta_new
    
    #---HW4----------------------------------------------------------------------------------------

    #---HW5----------------------------------------------------------------------------------------
    
    def sgd_step(self, x_k, k):
        '''
        Stochastic Gradient Descent step
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        ksi_k = np.mean([np.random.normal(0, 10, len(x_k)) for _ in range(self.batch_size)])
        
        return x_k - gamma * (self.grad_f(x_k, self.args) + ksi_k)
    
    #---HW6----------------------------------------------------------------------------------------
    
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
    
    #---HW7----------------------------------------------------------------------------------------
    
    def diana_step(self, x_k, k, h_list):
        '''
        DIANA step
        #each node generates h_i - new sequence
        #(starting) h_i = grad_i(x_0) and send to server
        #save delta_i = grad_i(x_k) - h_i
        #compressed_delta_i = Q(delta_i)
        #iteration h_i = h_i + alpha(= 1) * compressed_delta_i #both master and worker update
        #send to server compressed_delta_i

        #server calculates: x_k = x_k - gamma * 1/n * sum of (h_i + compressed_delta_i)
        '''
        grad_list = self.grad_f(x_k, self.args)
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        alpha = 1

        #Worker's calculation
        old_h_list = np.copy(h_list)
        compressed_delta_list = []

        for i in range(self.n_workers):
            delta_i = grad_list[i] - h_list[i]
            if self.rand_k_activate is True:
                compressed_delta_i = GradientOptimizer.rand_k_compressor(self, delta_i)
            else:
                compressed_delta_i = GradientOptimizer.top_k_compressor(self, delta_i)
            compressed_delta_list.append(compressed_delta_i)
            #old_h_ = h_list[i]
            h_list[i] = h_list[i] + alpha * compressed_delta_i
        
        #Master's calculations
        master_grad = np.zeros(len(grad_list[0]))
        for i in range(self.n_workers):
            master_grad += (old_h_list[i] + compressed_delta_list[i])
        master_grad /= self.n_workers

        return x_k - gamma * master_grad, h_list
    
    
    def ef_gd_step(self, x_k, k, errors_list):
        '''
        Error Feedback Gradient Descent step
        '''
        grad_list = self.grad_f(x_k, self.args)
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)

        #Worker's calculation of EF
        displaced_grad_list = np.zeros_like(grad_list)
        for i in range(self.n_workers):
            if self.rand_k_activate is True:
                displaced_grad_list[i] = GradientOptimizer.rand_k_compressor(self, errors_list[i] + gamma * grad_list[i])
            else:
                displaced_grad_list[i] = GradientOptimizer.top_k_compressor(self, errors_list[i] + gamma * grad_list[i])    
            errors_list[i] = errors_list[i] + gamma * grad_list[i] - displaced_grad_list[i]

        #Master's calculations
        master_grad = np.zeros(len(grad_list[0]))
        for i in range(self.n_workers):
            master_grad += displaced_grad_list[i]
        master_grad /= self.n_workers

        return x_k - master_grad, errors_list
    
    def ef21_step(self, x_k, k, g_list):
        '''
        The Error Feedback 21 approach
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)

        #Master's calculations
        g_k = np.zeros_like(g_list[0])
        for i in range(self.n_workers):
            g_k += g_list[i]
        g_k /= len(g_list[0])

        x_new = x_k - gamma * g_k

        grad_f_list_new = self.grad_f(x_new, self.args)
        g_list_new = np.zeros_like(g_list)

        #Worker's calculation
        for i in range(self.n_workers):
            if self.rand_k_activate is True:
                c_i = GradientOptimizer.rand_k_compressor(self, grad_f_list_new[i] - g_list[i])
            else:
                c_i = GradientOptimizer.top_k_compressor(self, grad_f_list_new[i] - g_list[i])
            
            g_list_new[i] = g_list[i] + c_i
            #Master computes via g_new = g + 1/n * \sum_{i = 1}^n c_i^t, 
            #which is the same as the previous line in my algorithm
        return x_new, g_list_new
    
    def marina(self, x_k, k, p_marina, g_k):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)

        grad_f_list_cur = self.grad_f(x_k, self.args)

        c_k = np.random.binomial(n = 1, p = p_marina)
        g_list_new = np.zeros_like(grad_f_list_cur)
        g_new = np.zeros_like(grad_f_list_cur[0])

        x_new = x_k - gamma*g_k

        for i in range(self.n_workers):
            grad_f_list_new = self.grad_f(x_new, self.args)
            if c_k == 1:
                g_list_new[i] = self.grad_f(x_new, self.args)[i]
            else:
                if self.rand_k_activate is True:
                    g_list_new[i] = GradientOptimizer.rand_k_compressor(self, grad_f_list_new[i] - grad_f_list_cur[i])
                else:
                    g_list_new[i] = GradientOptimizer.top_k_compressor(self, grad_f_list_new[i] - grad_f_list_cur[i])

            g_new += g_list_new[i]
        g_new /= self.n_workers

        return x_new, g_new
    
    #---DESCENT------------------------------------------------------------------------------------------
           
    def descent(self):
        '''
        This function realizes the descent to the optimum using one of the gradient-based methods
        '''
        x_k = np.copy(self.x_0)
        x_prev = np.copy(self.x_0)                                #for momentum algorithms
        grad_list = self.grad_f(self.x_0, self.args)
        h_k = grad_list
        errors_list = np.zeros_like(self.grad_f(x_k, self.args)) #for the ef_gd_step method
        h_list = grad_list                                        #for the diana_step 
        g_list = []                   
        if self.ef21_activate is True:                           #for ef21 step
            for i in range(self.n_workers):
                if self.rand_k_activate is True:
                    g_i = GradientOptimizer.rand_k_compressor(self, grad_list[i])
                else:
                    g_i = GradientOptimizer.top_k_compressor(self, grad_list[i])
                g_list.append(g_i)

        g_k = np.sum(grad_list, axis = 0) / len(grad_list)      #for marina step
        
        t_start = time.time()
        
        times_arr = []
        differences_arr = []
        points_arr = []
        acc_arr = []
        
        for k in tqdm(range(self.n_iter)):     
            if self.sgd_activate is True:
                x_k = GradientOptimizer.sgd_step(self, x_k, k)
            elif self.csgd_activate is True:
                x_k = GradientOptimizer.csgd_step(self, x_k, k)
            elif self.sega_activate is True:
                x_k = GradientOptimizer.sega_step(self, x_k, h_k, k)
            elif self.ef_activate is True:
                x_k, errors_list = GradientOptimizer.ef_gd_step(self, x_k, k, errors_list)
            elif self.diana_activate is True:
                x_k, h_list = GradientOptimizer.diana_step(self, x_k, k, h_list)
            elif self.ef21_activate is True:
                x_k, g_list = GradientOptimizer.ef21_step(self, x_k, k, g_list)
            elif self.marina_activate is True:
                x_k, g_k = GradientOptimizer.marina(self, x_k, k, self.p_marina, g_k)
            elif self.momentum_gd_activate is True:
                x_k, x_prev = GradientOptimizer.momentum_gd_step(self, x_k, k, x_prev)
            elif self.nesterov_momentum_activate is True:
                x_k, x_prev = GradientOptimizer.nesterov_momentum_step(self, x_k, k, x_prev)
            elif self.restart_activate is True:
                pass
            else:
                x_k = GradientOptimizer.gd_step(self, x_k, k)
                
            points_arr.append(x_k)

            if self.acc_k is not None:
                acc_arr.append(self.acc_k(k, self.f, self.grad_f, x_k, self.x_true, self.args))

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
                   
            if differences_arr[-1] <= self.y_lim:
                break

            if differences_arr[-1] >= self.upper_limit:
                print("The upper limit was broken!")
                break
            
        return points_arr, differences_arr, times_arr, acc_arr

#Plot Graphs
def plot_graphs(x, y, x_label, lines_labels, title, logscale = False, specific_slice = False, 
                criteria_type = "||x - x*||", idx_marker_arr = []):
    if specific_slice == False:
        specific_slice = range(len(y))
    
    plt.figure(figsize=(8, 8))
    for i in specific_slice:
        if i in idx_marker_arr:
            plt.plot(x[i], y[i], label = lines_labels[i], marker = '+')
        else:
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

#HW 5?
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
#[NEW]
def d_logloss_mushrooms(w, args):
    ans = 0
    for i in range(len(args['X_train'])):
        ans += np.log(1 + np.exp(-(w @ args["X_train"][i]) * args["y_train"][i]))
    return ans / len(args["X_train"])

def d_logloss_grad_mushrooms (w, args):
    grad_list = []
    for j in range(args['n_workers']):
        n_samples = len(args['X_train_list'][j]) 
        
        grad_j = np.zeros(w.size) 

        for i in range(n_samples):
            grad_j -= np.real(args['y_train_list'][j][i] * args['X_train_list'][j][i] * np.exp(- w.dot(args['X_train_list'][j][i]) * args["y_train_list"][j][i]) / (1 + np.exp(- w.dot(args['X_train_list'][j][i]) * args['X_train_list'][j][i])))
        grad_j /= n_samples
        grad_list.append(grad_j)
        
    return grad_list

def logloss_grad_mushrooms(w, args):
    n_samples = len(args['X_train']) 
    
    grad = np.zeros(w.size)

    for i in range(n_samples):
        grad -= np.real(args['y_train'][i] * args['X_train'][i] * np.exp(-w.dot(args['X_train'][i]) * args["y_train"][i]) / (1 + np.exp(- w.dot(args['X_train'][i]) * args['X_train'][i])))
    grad /= n_samples
            
    return grad

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

'''
#------------------------------------------from HW3 version 1
def Momentum_Grad_Descent(n_iter, X, lr, gamma, y, w_0, n):
    points = []
    w_old = w_0
    for i in range(n_iter):
        grad = Get_grad (X, y, w_old, n)
        if i < 2:
            w_new = w_old - lr*grad 
        else:
            w_new = w_old - lr*grad - gamma*(points[i] - points[i - 1])
        w_old = w_new 
        points.append(w_old)
    return points

def Momentum_Draw_Graph (n_iter, X, lr, gamma, y, w_0, n, w_true, criteria): 
    points = Momentum_Grad_Descent(n_iter, X, lr, gamma, y, w_0, n)
    diff_arr, crit_arr, x = [], [], []

    for i in range(n_iter):
        x.append(i + 1)
        diff_arr.append(points[i] - w_true)
        if criteria == "x_criteria":
            crit_arr.append(np.linalg.norm(diff_arr[i]))
        elif criteria == "f_criteria":
            crit_arr.append(np.abs(Logloss(X, y, points[i], n) - Logloss(X, y, w_true, n)))

    y = crit_arr
    if criteria == "x_criteria":
        plt.plot(x, y, label = "||w^k - w*||")
    elif criteria == "f_criteria":
        plt.plot(x, y, label = "|f(w^k) - f(w*)|")

    plt.xlabel('n_iter')
    plt.ylabel('criteria')
    plt.title('Standard axis convergence')
    plt.legend()
    plt.show()
'''