#Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from datetime import datetime as dt
from numpy.linalg import norm
import time
from sklearn.metrics import mean_squared_error, accuracy_score
from tqdm import tqdm
import scipy.optimize as spo

#---HW9-------------------------------------------------------------------------------------------------

class NewtonOptimizer:
    '''
    Class of Optimization Methods related to Newton's one
    '''
    def __init__(self, f, grad_f, x_0, gamma_k, args, n_iter = 5, criterium = '||x_k - x^*||', 
                 y_lim = 1e-5, x_true = None, newton_activate = False, hessian_f = None,
                 acc_k = None, upper_limit = 10**5, cubic_newton_activate = False, broyden_activate = False,
                 dfp_activate = False, bfgs_activate = False, l_bfgs_activate = False, m_l_bfgs = 5):
        #HW9
        self.f = f
        self.grad_f = grad_f
        self.x_0 = x_0
        self.gamma_k = gamma_k
        self.args = args
        self.n_iter = n_iter
        self.criterium = criterium
        self.y_lim = y_lim
        self.x_true = x_true
        self.newton_activate = newton_activate
        self.hessian_f = hessian_f
        self.acc_k = acc_k
        self.upper_limit = upper_limit
        self.cubic_newton_activate = cubic_newton_activate
        self.broyden_activate = broyden_activate
        self.dfp_activate = dfp_activate
        self.bfgs_activate = bfgs_activate
        self.l_bfgs_activate = l_bfgs_activate
        self.m_l_bfgs = m_l_bfgs

    def newton_step(self, x_k, k):
        grad = self.grad_f(x_k, self.args)
        hess = self.hessian_f(x_k, self.args)
        
        if len(x_k) == 1:
            hess_inv = 1 / hess
        else:
            hess_inv = np.linalg.inv(hess)

        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, hess_inv, self.args)

        return x_k - gamma * hess_inv @ grad
    
    def cubic_newton_step(self, x_k, k):
        grad = self.grad_f(x_k, self.args)
        hess = self.hessian_f(x_k, self.args)
        
        '''if len(x_k) == 1:
            hess_inv = 1 / hess
        else:
            hess_inv = np.linalg.inv(hess)'''

        def phi(x):
            return self.f(x_k, self.args) + grad @ (x - x_k) + \
            1/2 * (x - x_k) @ (hess @ (x - x_k)) + \
            self.args['M'] / 6 * norm(x_k - x, ord = 2)**3
        
        res = spo.minimize_scalar(phi)
        return res.x
    
    def broyden_step(self, x_k, H_k, s_k, y_k, k):
        #quasinewton: s_k = H_new @ y_k, where s_k = x_new - x_k, y_k = grad(x_k) - grad(x_new)
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        q_k = s_k - H_k @ y_k
        mu_k = 1 / (q_k.T @ y_k)

        delta_H_k = mu_k * q_k @ q_k.T
        H_k = H_k + delta_H_k
        
        return x_k - gamma * H_k @ self.grad_f(x_k, self.args), H_k, s_k, y_k

    def dfp_step(self, x_k, H_k, s_k, y_k, k):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)

        mu_1 = 1 / (s_k.T @ y_k)
        mu_2 = - 1 / ((H_k @ y_k).T @ y_k)

        delta_H_k = mu_1 * s_k @ s_k.T + mu_2 * H_k @ y_k @ (H_k @ y_k).T
        H_k = H_k + delta_H_k

        return x_k - gamma * H_k @ self.grad_f(x_k, self.args), H_k, s_k, y_k
    
    def bfgs_step(self, x_k, H_k, s_k, y_k, alpha_k, k):
        grad = self.grad_f(x_k, self.args)
        p_k = - H_k @ grad

        def find_alpha_k(alpha_k, x_k, p_k):
            c_1 = 1e-4
            c_2 = 0.9
            alpha_k = 0.01
            while self.f(x_k + alpha_k * p_k, self.args) <= self.f(x_k, self.args) + \
                c_1 * alpha_k * self.grad_f(x_k, self.args).T @ p_k and + \
                self.grad_f(x_k + alpha_k * p_k, self.args).T @ p_k >= c_2 * self.grad_f(x_k, self.args).T @ p_k:
                alpha_k *= 1.5
            return alpha_k
        
        alpha_k = find_alpha_k(alpha_k, x_k, p_k)

        x_new = x_k + alpha_k * p_k
        s_k = x_new - x_k
        y_k = self.grad_f(x_new, self.args) - grad

        rho_k = 1 / (y_k.T @ s_k)

        I = np.eye(len(x_k))
        H_new = (I - rho_k * s_k @ y_k.T) @ H_k @ (I - rho_k * y_k @ s_k.T) + rho_k * s_k @ s_k.T

        return x_new, H_new, s_k, y_k, alpha_k
    
    def l_bfgs_step(self, x_k, H_0, s_k_arr, y_k_arr, rho_k_arr, V_k_arr, k, m):
        m_hat = min(k, m - 1)
        H_new = H_0
        for i in range(k - m_hat, k + 1):
            H_new = V_k_arr[i].T @ H_new @ V_k_arr[i]
        left, right = np.eye(len(x_k)), np.eye(len(x_k))
        for i in range(k, k - m_hat - 1, -1):
            H_new += (((left @ s_k_arr[i].reshape(-1,1)) @ (s_k_arr[i].reshape(1,-1))) @ right)
        
            left = left @ V_k_arr[i].T
            right = V_k_arr[i] @ right

        d_k = - H_new @ self.grad_f(x_k, self.args)
        def find_alpha_k_l_bfgs(x_k, d_k):
            beta_ = 1e-4
            beta = 0.9
            alpha_k = 0.1
            while self.f(x_k + alpha_k * d_k, self.args) <= self.f(x_k, self.args) + \
                beta_ * alpha_k * self.grad_f(x_k, self.args).T @ d_k and + \
                self.grad_f(x_k + alpha_k * d_k, self.args).T @ d_k >= beta * self.grad_f(x_k, self.args).T @ beta:
                alpha_k *= 1.5
            return alpha_k
        
        alpha_k = find_alpha_k_l_bfgs(x_k, d_k)
        x_new = x_k + alpha_k * d_k

        s_k_arr.append(x_new - x_k)
        y_k_arr.append(self.grad_f(x_new, self.args) - self.grad_f(x_k, self.args))
        rho_k_arr.append(1/(y_k_arr[-1].T @ s_k_arr[-1]))
        V_k_arr.append(np.eye(len(x_k)) - rho_k_arr[-1] * y_k_arr[-1] @ s_k_arr[-1].T)

        return x_new, s_k_arr, y_k_arr, rho_k_arr, V_k_arr

    def gd_step(self, x_k, k):
        '''
        Basic Gradient Descent step
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        grad = self.grad_f(x_k, self.args)
            
        return x_k - gamma * grad
    
    def descent(self):
        '''
        This function realizes the descent to the optimum using one of the Newton-based methods
        '''
        #for every method
        x_k = np.copy(self.x_0) 
        grad = np.copy(self.grad_f(x_k, self.args))
        #for quasi-newton methods
        if  self.broyden_activate is True or self.dfp_activate is True or self.bfgs_activate is True:
            H_k = np.copy(self.hessian_f(x_k, self.args))
            #gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
            gamma = 1
            x_new = x_k - gamma * H_k @ grad
            s_k = x_new - x_k
            y_k = self.grad_f(x_new, self.args) - grad
            alpha_k = 0.1 #for bfgs
        if self.l_bfgs_activate is True: #for l_bfgs
            H_0 = np.copy(self.hessian_f(x_k, self.args))
            gamma = 1
            x_new = x_k - gamma * H_0 @ grad
            s_k_arr = [np.array(x_new - x_k)]
            y_k_arr = [np.array(self.grad_f(x_new, self.args) - grad)]
            rho_k_arr = [1/(y_k_arr[-1].T @ s_k_arr[-1])]
            V_k_arr = [np.eye(len(x_k)) - rho_k_arr[-1] * y_k_arr[-1] @ s_k_arr[-1].T]
        
        t_start = time.time()
        
        times_arr = []
        differences_arr = []
        points_arr = []
        acc_arr = []
        
        for k in tqdm(range(self.n_iter)):     
            if self.newton_activate is True:
                x_k = NewtonOptimizer.newton_step(self, x_k, k)
            elif self.cubic_newton_activate is True:
                x_k = NewtonOptimizer.cubic_newton_step(self, x_k, k)
            elif self.broyden_activate is True:
                x_k, H_k, s_k, y_k = NewtonOptimizer.broyden_step(self, x_k, H_k, s_k, y_k, k)
            elif self.dfp_activate is True:
                x_k, H_k, s_k, y_k = NewtonOptimizer.dfp_step(self, x_k, H_k, s_k, y_k, k)
            elif self.bfgs_activate is True:
                x_k, H_k, s_k, y_k, alpha_k = NewtonOptimizer.bfgs_step(self, x_k, H_k, s_k, y_k, alpha_k, k)
            elif self.l_bfgs_activate is True:
                x_new, s_k_arr, y_k_arr, rho_k_arr, V_k_arr = NewtonOptimizer.l_bfgs_step(self, x_k, H_0, s_k_arr, y_k_arr, rho_k_arr, V_k_arr, k, self.m_l_bfgs)
            else:
                x_k = NewtonOptimizer.gd_step(self, x_k, k)

            points_arr.append(x_k)

            if self.acc_k is not None:
                acc_arr.append(self.acc_k(k, self.f, self.grad_f, x_k, self.x_true, self.args))

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

#---HW8-------------------------------------------------------------------------------------------------

class VariatonalOptimizer:
    '''
    Class of Optimization Methods for Variatonal Inequality Problems
    '''
    def __init__(self, f, grad_f_x, grad_f_y, x_0, y_0, gamma_k, args, n_iter = 100, criterium = '||x_k - x^*||', 
                 y_lim = 1e-5, z_true = None, projection_activate = False, projection_operator = None,
                 extragradient_activate = False, smp_activate = False):
        '''
        :parameter f: target function
        :parameter grad_f_x and grad_f_y: function gradients by x and y
        :parameter x_0 and y_0: starting point
        :parameter args: the arguments of the optimization problem
        :parameter n_iter: number of iterations
        :parameter criterium: criterium of convergence, options: 'relative ||z_k - z^*||', '||z_k - z^*||'
        :parameter y_lim: target difference from z_k and z^*
        :parameter z_true: true optimum
        :parameter projection_activate: activate projection in descent algorithm
        :parameter projection_operator: projection operator in descent algorithm
        :parameter extragradient_activate: activate extragradient method
        :parameter smp_activate: activate smp
        '''
        #HW8
        self.f = f
        self.grad_f_x = grad_f_x
        self.grad_f_y = grad_f_y
        self.x_0 = x_0
        self.y_0 = y_0
        self.gamma_k = gamma_k
        self.args = args
        self.n_iter = n_iter
        self.criterium = criterium
        self.y_lim = y_lim
        self.z_true = z_true
        self.projection_activate = projection_activate
        self.projection_operator = projection_operator
        self.extragradient_activate = extragradient_activate
        self.smp_activate = smp_activate

    def gd_step(self, x_k, y_k, k):
        '''
        Basic Gradient Descent step
        '''
        gamma = self.gamma_k(k, self.args)

        x_new = x_k - gamma * self.grad_f_x(x_k, y_k, self.args)
        y_new = y_k + gamma * self.grad_f_y(x_k, y_k, self.args)

        if self.projection_activate is True:
            x_new = self.projection_operator(x_new, self.args)
            y_new = self.projection_operator(y_new, self.args)

        return x_new, y_new
    
    def extragradient_step(self, x_k, y_k, x_temp, y_temp, k):
        '''
        Extragradient step
        '''
        gamma = self.gamma_k(k, self.args)

        x_temp = x_k - gamma * self.grad_f_x(x_temp, y_temp, self.args)
        y_temp = y_k + gamma * self.grad_f_y(x_temp, y_temp, self.args)
        
        if self.projection_activate is True:
            x_temp = self.projection_operator(x_temp, self.args)
            y_temp = self.projection_operator(y_temp, self.args)
        
        x_new = x_k - gamma * self.grad_f_x(x_temp, y_temp, self.args)
        y_new = y_k + gamma * self.grad_f_y(x_temp, y_temp, self.args)

        if self.projection_activate is True:
            x_new = self.projection_operator(x_new, self.args)
            y_new = self.projection_operator(y_new, self.args)
            '''x_temp = self.projection_operator(x_temp, self.args)
            y_temp = self.projection_operator(y_temp, self.args)'''

        return x_new, y_new, x_temp, y_temp

    def smp_step(self, w_x_k, w_y_k, r_x_k, r_y_k, k):
        '''
        Realization of the step od SMP (Stochastic Mirror Prox algorithm: https://arxiv.org/pdf/0809.0815.pdf)
        '''
        gamma = self.gamma_k(k, self.args)

        def smp_prox_operator(z, ksi):
            prox = np.zeros_like(z)
            normalize = np.sum(z * np.exp(ksi))
            for j in range(len(z)):
                prox[j] = z[j] * np.exp(ksi[j]) / normalize
            return prox
        
        w_x_k = smp_prox_operator(r_x_k, gamma * self.grad_f_x(r_x_k, r_y_k, self.args))
        w_y_k = smp_prox_operator(r_y_k, gamma * ( - self.grad_f_y(r_x_k, r_y_k, self.args)))
        r_x_k = smp_prox_operator(r_x_k, gamma * self.grad_f_x(w_x_k, w_y_k, self.args))
        r_y_k = smp_prox_operator(r_y_k, gamma * ( - self.grad_f_y(w_x_k, w_y_k, self.args)))

        if self.projection_activate is True:
            w_x_k = self.projection_operator(w_x_k, self.args)
            w_y_k = self.projection_operator(w_y_k, self.args)
            r_x_k = self.projection_operator(r_x_k, self.args)
            r_y_k = self.projection_operator(r_y_k, self.args)

        return w_x_k, w_y_k, r_x_k, r_y_k, gamma


    def descent(self):
        '''
        Function to descent to the optimum
        '''
        differences_arr, points_arr, gradient_calls_arr = [], [], []
        z_0 = np.hstack([self.x_0, self.y_0])
    
        x_k, y_k = np.copy(self.x_0), np.copy(self.y_0)
        #for extragradient
        x_temp, y_temp = np.copy(self.x_0), np.copy(self.y_0)
        #for smp 
        w_x_k, w_y_k, r_x_k, r_y_k = np.copy(self.x_0), np.copy(self.y_0), np.copy(self.x_0), np.copy(self.y_0)
        gamma_sum, gamma_w_x_sum, gamma_w_y_sum = 0, np.zeros_like(self.x_0), np.zeros_like(self.y_0)

        gradient_calls = 0 #consider an operator's gradient call
        for k in tqdm(range(self.n_iter)):     
            if self.extragradient_activate is True:
                x_k, y_k, x_temp, y_temp = VariatonalOptimizer.extragradient_step(self, x_k, y_k, x_temp, y_temp, k)
                gradient_calls += 2
            elif self.smp_activate is True:
                w_x_k, w_y_k, r_x_k, r_y_k, gamma = VariatonalOptimizer.smp_step(self, w_x_k, w_y_k, r_x_k, r_y_k, k)
                
                gradient_calls += 2
                gamma_sum += gamma
                gamma_w_x_sum += gamma * w_x_k
                gamma_w_y_sum += gamma * w_y_k
                
                x_k = gamma_w_x_sum / gamma_sum
                y_k = gamma_w_y_sum / gamma_sum
            else:
                x_k, y_k = VariatonalOptimizer.gd_step(self, x_k, y_k, k)
                gradient_calls += 1
                
            z_k = np.hstack([x_k, y_k])
            points_arr.append(z_k)
            gradient_calls_arr.append(gradient_calls)

            if self.criterium == '||z_k - z^*||':
                differences_arr.append(np.linalg.norm(z_k - self.z_true, ord = 2))
            elif self.criterium == 'relative ||z_k - z^*||':
                differences_arr.append(np.linalg.norm(z_k - self.z_true, ord = 2) / np.linalg.norm(z_0 - self.z_true, ord = 2))
            elif self.criterium == 'Errvi':
                differences_arr.append(np.max(self.args['A'] @ x_k) - np.min(self.args['A'] @ y_k))
            else:
                AssertionError
        return points_arr, differences_arr, gradient_calls_arr

#---HW1-7-------------------------------------------------------------------------------------------------

class GradientOptimizer:
    '''
    Class of Gradient Methods Optimizers for regular optimization problems
    '''
    def __init__(self, f, grad_f, x_0, gamma_k, args, n_iter = 1000, n = 1, criterium = '||x_k - x^*||', 
                 y_lim = 1e-8, x_true = None, csgd_activate = False, grad_f_j = None, is_independent = False, 
                 n_coord = 1, sega_activate = False, n_workers = 1, top_k_activate = False, 
                 rand_k_activate = False, ef_activate = False, top_k_param = 0, rand_k_param = 0, 
                 diana_activate = False, acc_k = None, upper_limit = 1e10, ef21_activate = False, marina_activate = False,
                 p_marina = 0.5, momentum_gd_activate = False, nesterov_momentum_activate = False, momentum_coeff_k = 0,
                 restart_activate = False, noisy_gradient_activate = False, ksi_sigma = 10, sgd_activate = False, 
                 batch_size = 1, saga_activate = False, svrg_activate = False, p_svrg = 0.5, sarah_activate = False, p_sarah = 0.5,
                 proj_activate = False, proj_func = None, prox_activate = False, prox_func = None, frank_wolfe_activate = False,
                 mirror_descent_activate = False, accelerated_fw_activate = False, server_ef_activate = False,
                 master_rand_k_activate = False):
        '''
        :parameter f: target function
        :parameter grad_f: target function gradient
        :parameter x_0: starting point
        :parameter gamma_k: learning rate function depending on the number of current iteration k
        :parameter args: the arguments of the optimization problem
        :parameter n_iter: number of iterations
        :parameter n: number of workers (functions to optimize)
        :parameter args: includes parameters of functions
        :parameter criterium: criterium of convergence, options: '||x_k - x^*||', '|f(x_k) - f(x^*)|', '||grad_f(x_k)||', 'gap'
        :parameter y_lim: target difference from x_k and x^*
        :parameter x_true: true optimum
        :parameter grad_f_j: the j-th coordinate of a gradient  
        :parameter proj_activate: activate projection operator
        :parameter proj_func: projection operator
        :parameter prox_activate: activate proximal operator
        :parameter prox_func: proximal operator
        HW3
        :parameter momentum_gd_activate: activate momentum gradient descent
        :parameter nesterov_momentum_activate: activate nesterov momentum algorithm
        :parameter momentum_coeff_k: the coefficient in front of the momentum 
        :parameter restart_activate: activate restart method in accelerated method
        HW4
        :parameter frank_wolfe_activate: activate frank_wolfe
        :parameter mirror_descent_activate: activate mirror_descent
        :parameter accelerated_fw_activate: activate accelerated frank wolfe algorithm
        HW5
        :parameter noisy_gradient_activate: activate noisy gradient method
        :parameter ksi_sigma: sqrt(variance of ksi) in noisy gradient method
        :parameter sgd_activate: activate sgd
        :parameter batch_size: number of batches (by default equals to 1)
        :parameter saga_activate = activate saga
        :parameter svrg_activate: activate svrg
        :parameter p_svrg: probability in svrg method
        :parameter sarah_activate: activate sarah
        :parameter p_sarah: probability in sarah method
        HW6
        :parameter csgd_activate: activate csgd
        :parameter is_independent: use the independent coordinates in csgd if True
        :parameter n_coord: the number of coordinates left in the csgd
        :parameter sega_activate: activate sega
        HW7
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
        :parameter server_ef_activate: activate error feedback on the server
        :parameter master_rand_k_activate: the compressor on the master is done by rand_k if true, else by top_k
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
        self.csgd_activate = csgd_activate
        self.grad_f_j = grad_f_j
        self.is_independent = is_independent
        self.n_coord = n_coord
        self.sega_activate = sega_activate
        self.proj_activate = proj_activate
        self.proj_func = proj_func
        self.prox_activate = prox_activate
        self.prox_func = prox_func
        #HW3
        self.momentum_gd_activate = momentum_gd_activate
        self.nesterov_momentum_activate = nesterov_momentum_activate
        self.momentum_coeff_k = momentum_coeff_k
        self.restart_activate = restart_activate
        #HW4
        self.frank_wolfe_activate = frank_wolfe_activate
        self.mirror_descent_activate = mirror_descent_activate
        self.accelerated_fw_activate = accelerated_fw_activate
        #HW5
        self.noisy_gradient_activate = noisy_gradient_activate
        self.ksi_sigma = ksi_sigma
        self.batch_size = batch_size
        self.sgd_activate = sgd_activate
        self.saga_activate = saga_activate
        self.svrg_activate = svrg_activate
        self.p_svrg = p_svrg
        self.sarah_activate = sarah_activate
        self.p_sarah = p_sarah
        #HW7
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
        self.server_ef_activate = server_ef_activate
        self.master_rand_k_activate = master_rand_k_activate

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

    def frank_wolfe_step(self, x_k, k):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)

        #note: this works for probabilistic simplex. Generally s = argmin_s<grad_f, s - x_k>
        s = np.zeros(len(x_k))
        s[np.argmin(self.grad_f(x_k, self.args))] = 1

        return x_k + gamma * (s - x_k)

    def mirror_descent_step(self, x_k, k):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        #Note: only for simplex
        return x_k * np.exp(- gamma * self.grad_f(x_k, self.args)) /(x_k @ np.exp(- gamma * self.grad_f(x_k, self.args)))

    def accelerated_fw_step(self, x_k, k):       
        return GradientOptimizer.cgs(self, x_k, k)

    def cgs(self, x_s, s):
        x_k, y_k = x_s, x_s

        N = int(2 * np.sqrt((6*self.args['L']) / self.args['mu'])) + 1

        def gamma(k):
            return 2 / (k + 2)

        def beta(k):
            return (2 * self.args['L']) / (k + 1)

        def eta(k, s, N):
            return (8 * self.args['L'] * self.args['delta0'] * (2**(-s - 1))) / (self.args['mu'] * N * (k + 1))
         
        for k in range(N):
            gamma_k = gamma(k)
            beta_k = beta(k)
            eta_ksN = eta(k, s, N)

            z_k = (1 - gamma_k) * y_k + gamma_k * x_k
            x_k = GradientOptimizer.CndG(self.grad_f(z_k, self.args), x_k, beta_k, eta_ksN)
            y_k = (1 - gamma_k) * y_k + gamma_k * x_k
    
        return y_k

    def CndG(gamma, u, beta, eta):
        ut = np.array(u)
        vt = GradientOptimizer.max_simplex(gamma, beta, ut, u)
        V = (gamma + beta * (ut - u)) @ (ut - vt)
        while V > eta:
            temp = ((beta * (u - ut) - gamma)@(vt - ut))/(beta * ((np.linalg.norm(vt - ut))**2))
            alphat = min(1, temp)
            ut = (1 - alphat) * ut + alphat * vt
            vt = GradientOptimizer.max_simplex(gamma, beta, ut, u)
            V = (gamma + beta * (ut - u)) @ (ut - vt)
        return ut

    def max_simplex(gamma, beta, ut, u):
        vector = gamma + beta * (ut - u)
        x = np.zeros(len(vector))
        x[np.argmin(vector)] = 1
        return x

    #---HW5----------------------------------------------------------------------------------------
    
    def noisy_gradient_step(self, x_k, k):
        '''
        Noisy Gradient Descent step
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        ksi_k = np.mean([np.random.normal(0, self.ksi_sigma, len(x_k)) for _ in range(self.batch_size)])
        
        return x_k - gamma * (self.grad_f(x_k, self.args) + ksi_k)

    def sgd_step(self, x_k, k):
        '''
        Stochastic Gradient Descent step
        '''
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        j = np.random.randint(0, self.args['n'])
        
        return x_k - gamma * self.grad_f_j(x_k, j, self.args)
    
    def saga_step(self, x_k, phi_k, k):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)

        j = np.random.randint(0, self.args['n'])
        phi_j_new = x_k

        g_additional = np.zeros_like(x_k)
        for i in range(self.args['n']):
            g_additional += self.grad_f_j(phi_k[i], i, self.args)
        g_additional /= self.args['n']

        g_k = self.grad_f_j(phi_j_new, j, self.args) - self.grad_f_j(phi_k[j], j, self.args) + g_additional

        phi_k[j] = phi_j_new

        return x_k - gamma * g_k, phi_k
    
    def svrg_step(self, x_k, w_k, g_k, k):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        i = np.random.randint(0, self.args['n'])

        g = self.grad_f_j(x_k, i, self.args) - self.grad_f_j(w_k, i, self.args) + g_k

        x_k = x_k - gamma * g

        p = np.random.random()
        if self.p_svrg >= p:
            w_k = x_k
            g_k = self.grad_f(x_k, self.args)
        
        return x_k, w_k, g_k        
    
    def sarah_step(self, x_k, x_old, g_k, k):
        gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
        
        j = np.random.randint(0, self.args['n'])

        p = np.random.random()
        if self.p_sarah >= p:
            g_k = self.grad_f_j(x_k, j, self.args) - self.grad_f_j(x_old, j, self.args) + g_k
        else:
            g_k = self.grad_f(x_k, self.args)

        x_k = x_k - gamma * g_k

        return x_k, g_k
    
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

        g_k = h_k 
        h_k_new = h_k
        j_list = [np.random.randint(self.args['d']) for i in range(self.n_coord)]
        for j in j_list:
            e_j = np.zeros(len(x_k))
            e_j[j] = 1

            grad_j = self.grad_f_j(x_k, j, self.args).real

            h_k_new += e_j * (grad_j - h_k[j])
            g_k += self.args['d'] * e_j * (grad_j - h_k[j])

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

        return x_k - master_grad, errors_list, displaced_grad_list
    
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

        if self.ef21_activate is True:                           #for ef21 step
            g_list = []   
            for i in range(self.n_workers):
                if self.rand_k_activate is True:
                    g_i = GradientOptimizer.rand_k_compressor(self, grad_list[i])
                else:
                    g_i = GradientOptimizer.top_k_compressor(self, grad_list[i])
                g_list.append(g_i)
            grad_list = g_list

        g_k = np.sum(grad_list, axis = 0) / len(grad_list)      #for marina step
        y_k = np.copy(self.x_0) #for restart
        theta_k = 1             #for restart

        phi_k = [np.copy(self.x_0) for i in range(self.args['n'])] #for saga
        w_k = np.copy(self.x_0) #for svrg
        g_k_stoch = self.grad_f(self.x_0, self.args) #for svrg and sarah  
        x_old = x_k #for sarah     
        h_k_sega = np.zeros_like(self.grad_f(self.x_0, self.args)) #for sega

        server_error = np.zeros_like(self.x_0) #for server ef

        t_start = time.time()
        
        times_arr = []
        differences_arr = []
        points_arr = []
        acc_arr = []
        
        for k in tqdm(range(self.n_iter)):     
            if self.sgd_activate is True:
                x_k = GradientOptimizer.sgd_step(self, x_k, k)
            elif self.saga_activate is True:
                x_k, phi_k = GradientOptimizer.saga_step(self, x_k, phi_k, k)
            elif self.svrg_activate is True:
                x_k, w_k, g_k_stoch = GradientOptimizer.svrg_step(self, x_k, w_k, g_k_stoch, k)
            elif self.sarah_activate is True:
                 x_new, g_k_stoch = GradientOptimizer.sarah_step(self, x_k, x_old, g_k_stoch, k)
                 x_old = x_k
                 x_k = x_new
            elif self.csgd_activate is True:
                x_k = GradientOptimizer.csgd_step(self, x_k, k)
            elif self.sega_activate is True:
                x_k, h_k_sega = GradientOptimizer.sega_step(self, x_k, h_k_sega, k)
            elif self.ef_activate is True:
                x_k, errors_list, grad_list = GradientOptimizer.ef_gd_step(self, x_k, k, errors_list)
            elif self.diana_activate is True:
                x_k, grad_list = GradientOptimizer.diana_step(self, x_k, k, grad_list)
            elif self.ef21_activate is True:
                x_k, grad_list = GradientOptimizer.ef21_step(self, x_k, k, grad_list)
            elif self.marina_activate is True:
                x_k, g_k = GradientOptimizer.marina(self, x_k, k, self.p_marina, g_k)
            elif self.momentum_gd_activate is True:
                x_k, x_prev = GradientOptimizer.momentum_gd_step(self, x_k, k, x_prev)
            elif self.nesterov_momentum_activate is True:
                x_k, x_prev = GradientOptimizer.nesterov_momentum_step(self, x_k, k, x_prev)
            elif self.restart_activate is True:
                x_k, y_k, theta_k = GradientOptimizer.restart_step(self, x_k, k, y_k, theta_k)
            elif self.noisy_gradient_activate is True:
                x_k = GradientOptimizer.noisy_gradient_step(self, x_k, k)
            elif self.frank_wolfe_activate is True:
                x_k = GradientOptimizer.frank_wolfe_step(self, x_k, k)
            elif self.mirror_descent_activate is True:
                x_k = GradientOptimizer.mirror_descent_step(self, x_k, k)
            elif self.accelerated_fw_activate is True:
                x_k = GradientOptimizer.accelerated_fw_step(self, x_k, k)
            else:
                x_k = GradientOptimizer.gd_step(self, x_k, k)

            if self.server_ef_activate is True: #Bonus 2 HW7
                gamma = self.gamma_k(k, self.f, self.grad_f, x_k, self.x_true, self.args)
                g_master, g_master_compressed = np.zeros_like(self.x_0), np.zeros_like(self.x_0)   
                for i in range(self.n_workers):
                    g_master += grad_list[i]
                g_master /= self.n_workers

                if self.master_rand_k_activate is True:
                    g_master_compressed = GradientOptimizer.rand_k_compressor(self, server_error + gamma * g_master)
                else:
                    g_master_compressed = GradientOptimizer.top_k_compressor(self, server_error + gamma * g_master)
                server_error = server_error + gamma * g_master - g_master_compressed

                x_k = x_k - g_master_compressed

            if self.proj_activate is True:
                x_k = self.proj_func(x_k, k, self.args)
            elif self.prox_activate is True:
                x_k = self.prox_func(x_k, k, self.args)

            points_arr.append(x_k)

            if self.acc_k is not None:
                acc_arr.append(self.acc_k(k, self.f, self.grad_f, x_k, self.x_true, self.args))

            if self.criterium == '||x_k - x^*||':
                differences_arr.append(norm(x_k - self.x_true, ord = 2))
            elif self.criterium == '|f(x_k) - f(x^*)|':
                differences_arr.append(self.f(x_k, self.args) - self.f(self.x_true, self.args))
            elif self.criterium == '||grad_f(x_k)||':
                differences_arr.append(norm(self.grad_f(x_k, self.args), ord = 2))
            elif self.criterium == 'gap':
                differences_arr.append(self.grad_f(x_k, self.args) @ x_k - np.min(self.grad_f(x_k, self.args)))
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
    U = ortho_group.rvs(dim = d)
    A = mu * np.eye(d)
    A[0][0] = L
    A = U.T @ A @ U

    return A

#Quadratic Function 
def f_quad(x, args):
    return 1/2*(x.T @ args['A'] @ x - args['b'].T @ x)

def f_quad_grad(x, args):
    return args['A'] @ x - args['b']

def f_quad_grad_j(x, j, args):
    return args['A'][j] @ x - args['b'][j]    

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