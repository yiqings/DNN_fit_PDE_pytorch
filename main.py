import torch
from torch import Tensor, optim
import numpy
from numpy import array, arange, zeros, sum, sqrt, linspace, ones, absolute, meshgrid
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_in_cube_time_dependent,\
    generate_uniform_points_on_cube, generate_uniform_points_on_cube_time_dependent,\
    generate_uniform_points_in_sphere, generate_uniform_points_in_sphere_time_dependent,\
    generate_uniform_points_on_sphere, generate_uniform_points_on_sphere_time_dependent,\
    generate_learning_rates
import network_3 as network_file
from selection_network_setting import selection_network
from matplotlib.pyplot import plot, show, clf, pause, subplot, xlim, semilogy
import pickle
import time
import os
from solution_Poisson_poly import true_solution, Du, Du_ft, Bu_ft, f, g, h0, h1, domain_shape, domain_parameter, time_dependent_type, time_interval

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

 
torch.set_default_tensor_type('torch.cuda.DoubleTensor')
device = torch.device('cuda') # run the codes on GPU devices
########### Plot parameters  #############
n_epoch_plot = 25 # frequency to plot the solutions
n_epoch_plot_loss = 25 # frequency to plot the loss 
flag_plot_selection_network = False 
flag_plot_truesolution  = True # plot true solution
flag_plot_solution = True
########### Saving parameters  #############
# please change the path to the folder where you save the codes
save_path = '/home/shenyiqing/pde/test/{}/'.format(current_name) # path to save the figure of the training netwrok
# sub names
current_name = 'test_plot' # name of the folder to save the figures
save_network_path = save_path+'/model/' # path to save the weights of the model
solution_network_name = 'solution_network' 
seltection_network_1_name = 'selection_network_1'
seltection_network_2_name = 'selection_network_2'
########### Set parameters #############
d = 2  # dimension of problem
m = 100  # number of nodes in each layer of solution network

n_epoch = 2000  # number of outer iterations
N_inside_train = 1000 #1000 # number of training sampling points inside the domain in each epoch (batch size)
n_update_each_batch = 1 # number of iterations in each epoch (for the same batch of points)
lrseq = generate_learning_rates(-3,-4,n_epoch,0.8,5) # set the learning rates for each epoch

lambda_term = 1

m_sn = 20   # number of nodes in each layer of the selection network
penalty_parameter = 0.01 # the epsilon in the SelectNet loss function  
maxvalue_sn = 5  # upper bound for the selection net
minvalue_sn = 0.5  # lower bound for the selection net
lr_sn = 0.01  # learning rate for the selection net
n_update_each_batch_sn = 1 # number of updates for selection net in each epoch
loss_sn1_threshold = 1e-5  # stopping criteria for training the selection net1 (inside the domain)
loss_sn2_threshold = 1e-5  # stopping criteria for training the selection net2 (boudanry or initial)
    
    
activation = 'ReLU3'  # activation function for the solution net
boundary_control = 'none' #'convection_dominated_1'  # if the solution net architecture satisfies the boundary condition automatically 
flag_preiteration_by_small_lr = False  # If pre iteration by small learning rates
lr_pre = 1e-4  # learning rates in preiteration
n_update_each_batch_pre = 10 # number of iterations in each epoch in preiteration
h_Du_t = 0.01  # time length for computing the first derivative of t by finite difference (for the hyperbolic equations)
flag_reset_select_net_each_epoch = False  # if reset selection net for each outer iteration
selectnet_initial_constant = 1  # if selectnet is initialized as constant one

########### Problem parameters  #############
time_dependent_type = time_dependent_type()   ## If this is a time-dependent problem
domain_shape = domain_shape()  ## the shape of domain 
if domain_shape == 'cube':  
    domain_intervals = domain_parameter(d)
elif domain_shape == 'sphere':
    R = domain_parameter(d)
    
if not time_dependent_type == 'none':    
    time_interval = time_interval()
    T0 = time_interval[0]
    T1 = time_interval[1]
    
########### Interface parameters #############
flag_compute_loss_each_epoch = True # if compute loss after each epoch
n_epoch_show_info = max([round(n_epoch/100),1]) # how many epoch will it show information
N_test = N_inside_train #10000 # number of testing points
flag_l2error = True
flag_maxerror = True
flag_givenpts_l2error = True#False
flag_givenpts_maxerror = True #False
if flag_givenpts_l2error == True or flag_givenpts_maxerror == True:
    if time_dependent_type == 'none':
        given_pts = zeros((1,d))
    else:
        given_pts = zeros((1,d+1))
flag_show_sn_info = False # if show information for the selection net training
flag_show_plot = True # if show plot during the training
flag_output_results = False # if save the results as files in current directory
    
########### Depending parameters #############
u_net = network_file.network(d,m, activation_type = activation, boundary_control_type = boundary_control)
if u_net.if_boundary_controlled == False:
    flag_boundary_term_in_loss = True  # if loss function has the boundary residual
else:
    flag_boundary_term_in_loss = False
if time_dependent_type == 'none':
    flag_initial_term_in_loss = False  # if loss function has the initial residual
else:
    if u_net.if_initial_controlled == False:
        flag_initial_term_in_loss = True
    else:
        flag_initial_term_in_loss = False
if flag_boundary_term_in_loss == True or flag_initial_term_in_loss == True:
    flag_IBC_in_loss = True  # if loss function has the boundary/initial residual
    N_IBC_train = 0  # number of boundary and initial training points
else:
    flag_IBC_in_loss = False
if flag_boundary_term_in_loss == True:
    if domain_shape == 'cube':
        if d == 1 and time_dependent_type == 'none':
            N_each_face_train = 1
        else:
            N_each_face_train = N_inside_train  #max([1,int(round(N_inside_train/2/d))]) # number of sampling points on each domain face when training
        N_boundary_train = 2*d*N_each_face_train
    elif domain_shape == 'sphere':
        if d == 1 and time_dependent_type == 'none':
            N_boundary_train = 2
        else:
            N_boundary_train = N_inside_train # number of sampling points on each domain face when training
    N_IBC_train = N_IBC_train + N_boundary_train
else:
    N_boundary_train = 0
if flag_initial_term_in_loss == True:          
    N_initial_train = max([1,int(round(N_inside_train/d))]) # number of sampling points on each domain face when training
    N_IBC_train = N_IBC_train + N_initial_train
########### Construct the saving document #############
isExists=os.path.exists(save_path)
if not isExists:
    os.makedirs(save_path)

########### Set functions #############
# function to evaluate the discrete L2 error (input x_batch is a 2d numpy array; output is a scalar Tensor)
def evaluate_rel_l2_error(model, x_batch):
    l2error = sqrt(sum((model.predict(x_batch) - true_solution(x_batch))**2)/x_batch.shape[0])
    u_l2norm = sqrt(sum((true_solution(x_batch))**2)/x_batch.shape[0])
    return l2error/u_l2norm

def evaluate_rel_max_error(model, x_batch):
    maxerror = numpy.max(absolute(model.predict(x_batch) - true_solution(x_batch)))
    u_maxnorm = numpy.max(absolute(true_solution(x_batch)))
    return maxerror/u_maxnorm

def evaluate_l2_error(model, x_batch):
    l2error = sqrt(sum((model.predict(x_batch) - true_solution(x_batch))**2)/x_batch.shape[0])
    return l2error

def evaluate_max_error(model, x_batch):
    maxerror = numpy.max(absolute(model.predict(x_batch) - true_solution(x_batch)))
    return maxerror

if not time_dependent_type == 'none':   
    def Du_t_ft(model, tensor_x_batch):
        h = 0.01 # step length ot compute derivative
        s = torch.zeros(tensor_x_batch.shape[0])
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:,0] = 1
        s = (3*model(tensor_x_batch+2*h*ei)-4*model(tensor_x_batch+h*ei)+model(tensor_x_batch))/2/h
        return s
#functions to plot loss
def plotloss(loss,l2error,maxerror,save_path=save_path):
        plt.figure()
        ax = plt.gca()
        y1 = loss
        y2 = l2error
        y3 = maxerror
        plt.plot(y1,'ro',label='loss')
        plt.plot(y2,'g*',label='l2error')
        plt.plot(y3,'b*',label='maxerror')
        ax.set_xscale('log')
        ax.set_yscale('log')                
        plt.legend(fontsize=18)
        plt.title('loss',fontsize=15)
        save_path_name = save_path + 'loss.png'
        plt.savefig(save_path_name)
        plt.close()

flag_can_plot_dim2 = False # only plots when spatial dimenstion + time dimension = 2

if d ==2 and time_dependent_type == 'none':
    flag_can_plot_dim2 = True

if d ==1 and time_dependent_type != 'none':
    flag_can_plot_dim2 = False

if flag_can_plot_dim2:
    X = numpy.arange(-1, 1.01, 0.01)
    Y = numpy.arange(-1, 1.01, 0.01)
    X1, Y1 = numpy.meshgrid(X, Y)
    Z = numpy.zeros((len(X),len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            test_tensor = numpy.zeros((1,2))
            test_tensor[0,0] = X[i]
            test_tensor[0,1] = Y[j]
            Z[i,j]=true_solution(test_tensor)

        
def plot_the_network (model,save_path,image_name,X1,Y1,Z):
            
    X2 = numpy.arange(-1, 1.05, 0.05)
    Y2 = numpy.arange(-1, 1.05, 0.05)
    X3, Y3 = numpy.meshgrid(X2, Y2)
    Z2 = numpy.zeros((len(X2),len(Y2)))
    for i in range(len(X2)):
        for j in range(len(Y2)):
            test_tensor = numpy.zeros((1,2))
            test_tensor[0,0] = X2[i]
            test_tensor[0,1] = Y2[j]
            Z2[i,j]=model.predict(test_tensor)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X3, Y3, Z2, cmap='rainbow')

    if flag_plot_truesolution :
        ax.plot_surface(X1, Y1, Z, rstride=1, cstride=1, cmap='rainbow',alpha=0.5)
        
    save_path_name = save_path + image_name + '.png'
    plt.savefig(save_path_name)
    plt.close()





#################### Start ######################
optimizer = optim.Adam(u_net.parameters(),lr=lrseq[0])

lossseq = zeros((n_epoch,))
l2errorseq = zeros((n_epoch,))
maxerrorseq = zeros((n_epoch,))
givenpts_l2errorseq = zeros((n_epoch,))
givenpts_maxerrorseq = zeros((n_epoch,))

if time_dependent_type == 'none':
    if domain_shape == 'cube':
        x_test = generate_uniform_points_in_cube(domain_intervals,N_test)     
    elif domain_shape == 'sphere':
        x_test = generate_uniform_points_in_sphere(d,R,N_test)
else:
    if domain_shape == 'cube':
        x_test = generate_uniform_points_in_cube_time_dependent(domain_intervals,time_interval,N_test)
    elif domain_shape == 'sphere':
        x_test = generate_uniform_points_in_sphere_time_dependent(d,R,time_interval,N_test)

N_plot = 101
if time_dependent_type == 'none':
    x_plot = zeros((N_plot,d))
    if domain_shape == 'cube':
        x_plot[:,0] = linspace(domain_intervals[0,0],domain_intervals[0,1],N_plot)
    elif domain_shape == 'sphere':
        x_plot[:,0] = linspace(-R,R,N_plot)
else:
    x_plot = zeros((N_plot,d+1))
    x_plot[:,0] = linspace(T0,T1,101)

# Training
k = 0
while k < n_epoch:
    ## generate training and testing data (the shape is (N,d)) or (N,d+1) 
    ## label 1 is for the points inside the domain, 2 is for those on the bondary or at the initial time
    if time_dependent_type == 'none':
        if domain_shape == 'cube':
            x1_train = generate_uniform_points_in_cube(domain_intervals,N_inside_train)
            if flag_IBC_in_loss == True:
                x2_train = generate_uniform_points_on_cube(domain_intervals,N_each_face_train)        
        elif domain_shape == 'sphere':
            x1_train = generate_uniform_points_in_sphere(d,R,N_inside_train)
            if flag_IBC_in_loss == True:
                x2_train = generate_uniform_points_on_sphere(d,R,N_boundary_train)
    else:
        if domain_shape == 'cube':
            x1_train = generate_uniform_points_in_cube_time_dependent(domain_intervals,time_interval,N_inside_train)
            if flag_IBC_in_loss == True:
                # x2_train for boudanry samplings; x2_train for initial samplings
                x2_train, x3_train = generate_uniform_points_on_cube_time_dependent(domain_intervals,time_interval,N_each_face_train,N_initial_train)     
        elif domain_shape == 'sphere':
            x1_train = generate_uniform_points_in_sphere_time_dependent(d,R,time_interval,N_inside_train)
            if flag_IBC_in_loss == True:
                # x2_train for boudanry samplings; x2_train for initial samplings
                x2_train, x3_train = generate_uniform_points_on_sphere_time_dependent(d,R,time_interval,N_boundary_train,N_initial_train)
        
    tensor_x1_train = Tensor(x1_train)
    tensor_x1_train.requires_grad=False
    tensor_f1_train = Tensor(f(x1_train))
    tensor_f1_train.requires_grad=False
    if flag_boundary_term_in_loss == True:
        tensor_x2_train = Tensor(x2_train)
        tensor_x2_train.requires_grad=False
        tensor_g2_train = Tensor(g(x2_train))
        tensor_g2_train.requires_grad=False
        
    if flag_initial_term_in_loss == True:
        tensor_x3_train = Tensor(x3_train)
        tensor_x3_train.requires_grad=False
        if time_dependent_type == 'parabolic' or time_dependent_type == 'hyperbolic':
            # h0 for u(x,0) = h0(x)
            tensor_h03_train = Tensor(h0(x3_train))
            tensor_h03_train.requires_grad=False
        if time_dependent_type == 'hyperbolic':
            # h1 for u_t(x,0) = h1(x)
            tensor_h13_train = Tensor(h1(x3_train))
            tensor_h13_train.requires_grad=False
            
    ## Set learning rate
    for param_group in optimizer.param_groups:
        if flag_preiteration_by_small_lr == True and k == 0:
            param_group['lr'] = lr_pre
        else:
            param_group['lr'] = lrseq[k]
        
    ## Train the selection net inside the domain
    if flag_reset_select_net_each_epoch == True or k == 0:
        if time_dependent_type == 'none':
            select_net1 = selection_network(d,m_sn,maxvalue_sn,minvalue_sn, initial_constant = selectnet_initial_constant) # selection_network inside the domain
        else:
            select_net1 = selection_network(d+1,m_sn,maxvalue_sn,minvalue_sn, initial_constant = selectnet_initial_constant)  # selection_network for initial and boudanry conditions
        optimizer_sn1 = optim.Adam(select_net1.parameters(),lr=lr_sn)
    const_tensor_residual_square_x1_train = Tensor((Du(u_net,x1_train)-f(x1_train))**2)
    const_tensor_residual_square_x1_train.requires_grad=False
    old_loss = 1e5
    for i_update_sn in range(n_update_each_batch_sn):
        ## Compute the loss  
        loss_sn = -1/torch.sum(const_tensor_residual_square_x1_train)*torch.sum((select_net1(tensor_x1_train)*const_tensor_residual_square_x1_train)) + 1/penalty_parameter*(1/N_inside_train*torch.sum(select_net1(tensor_x1_train))-1).pow(2)
        ## Show information
        if flag_show_sn_info == True:
            if i_update_sn%10 == 0:
                print("select_net1: ",end='')
                temp = (1/N_inside_train*torch.sum(select_net1(tensor_x1_train))-1).pow(2)
                print("i_update_sn = %d, loss_sn = %6.3f, penalty term = %6.3f" %(i_update_sn,-loss_sn.item(),temp.item()))
        ## If loss_sn is stable, then break
        if abs(loss_sn.item()-old_loss)<loss_sn1_threshold:
            break
        old_loss = loss_sn.item()
        ## Update the network
        optimizer_sn1.zero_grad()
        loss_sn.backward(retain_graph=False)
        optimizer_sn1.step()
            
    ## Train the selection net on the boudnary or on the initial slice
    if flag_IBC_in_loss == True:
        if flag_reset_select_net_each_epoch == True or k == 0:
            if time_dependent_type == 'none':
                select_net2 = selection_network(d,m_sn,maxvalue_sn,minvalue_sn, initial_constant = selectnet_initial_constant)  # selection_network for initial and boudanry conditions
            else:
                select_net2 = selection_network(d+1,m_sn,maxvalue_sn,minvalue_sn, initial_constant = selectnet_initial_constant)  # selection_network for initial and boudanry conditions
            optimizer_sn2 = optim.Adam(select_net2.parameters(),lr=lr_sn)
        if flag_boundary_term_in_loss == True:
            const_tensor_residual_square_x2_train = (Bu_ft(u_net,tensor_x2_train)-tensor_g2_train)**2
        if flag_initial_term_in_loss == True:
            const_tensor_residual_square_x3_train = (u_net(tensor_x3_train)-tensor_h03_train)**2
        old_loss = 1e5
        for i_update_sn in range(n_update_each_batch_sn):
            ## Compute the loss  
            loss_sn = Tensor([0])
            tensor_IBC_sum_term = Tensor([0])
            if flag_boundary_term_in_loss == True:
                loss_sn = loss_sn - 1/torch.sum(const_tensor_residual_square_x2_train)*torch.sum((select_net2(tensor_x2_train)*const_tensor_residual_square_x2_train)) 
                tensor_IBC_sum_term = tensor_IBC_sum_term + torch.sum(select_net2(tensor_x2_train))
            if flag_initial_term_in_loss == True:
                loss_sn = loss_sn - 1/torch.sum(const_tensor_residual_square_x3_train)*torch.sum((select_net2(tensor_x3_train)*const_tensor_residual_square_x3_train)) 
                tensor_IBC_sum_term = tensor_IBC_sum_term + torch.sum(select_net2(tensor_x3_train))
            
            loss_sn = loss_sn + 1/penalty_parameter*(1/N_IBC_train*tensor_IBC_sum_term-1).pow(2)
            ## Show information
            if flag_show_sn_info == True:
                if i_update_sn%10 == 0:
                    print("select_net2: ",end='')
                    temp = (1/N_IBC_train*tensor_IBC_sum_term-1).pow(2)
                    print("i_update_sn = %d, loss_sn = %6.3f, penalty term = %6.3f" %(i_update_sn,-loss_sn.item(),temp.item()))
            ## If loss_sn is stable, then break
            if abs(loss_sn.item()-old_loss)<loss_sn2_threshold:
                break
            old_loss = loss_sn.item()
            ## Update the network
            optimizer_sn2.zero_grad()
            loss_sn.backward(retain_graph=True)
            optimizer_sn2.step()
    
    ## Train the solution net
    const_tensor_sn_x1_train = Tensor(select_net1.predict(x1_train))
    const_tensor_sn_x1_train.requires_grad=False
    if flag_boundary_term_in_loss == True:
        const_tensor_sn2_x2_train = Tensor(select_net2.predict(x2_train))
        const_tensor_sn2_x2_train.requires_grad=False
    if flag_initial_term_in_loss == True:
        const_tensor_sn2_x3_train = Tensor(select_net2.predict(x3_train))
        const_tensor_sn2_x3_train.requires_grad=False
        
    if flag_preiteration_by_small_lr == True and k == 0:
        temp = n_update_each_batch_pre
    else:
        temp = n_update_each_batch
    for i_update in range(temp):
        if flag_compute_loss_each_epoch == True or i_update == 0:
            ## Compute the loss  
            loss1 = 1/N_inside_train*torch.sum(const_tensor_sn_x1_train*(Du_ft(u_net,tensor_x1_train)-tensor_f1_train)**2)
            loss = loss1

            if flag_IBC_in_loss == True:
                loss2 = Tensor([0])
                if flag_boundary_term_in_loss == True:
                    loss2 = loss2 + torch.sum(const_tensor_sn2_x2_train*(Bu_ft(u_net,tensor_x2_train)-tensor_g2_train)**2)
                if flag_initial_term_in_loss == True:
                    loss2 = loss2 + torch.sum(const_tensor_sn2_x3_train*(u_net(tensor_x3_train)-tensor_h03_train)**2)
                    if time_dependent_type == 'hyperbolic':
                            loss2 = loss2 + torch.sum((Du_t_ft(u_net,tensor_x3_train)-tensor_h13_train)**2)
                loss2 = lambda_term/N_IBC_train*loss2
                loss = loss1 + loss2


#        if i_update%10 == 0:
#            print("i_update = %d, loss = %6.3f, L2 error = %5.3f" %(i_update,loss.item(),evaluate_rel_l2_error(u_net, x1_train)))
        
        ## Update the network
        optimizer.zero_grad()
        loss.backward(retain_graph=not flag_compute_loss_each_epoch)
        optimizer.step()
        
        if flag_show_plot == True:
            if i_update%10 == 0:
                # Plot the slice for xd
                clf()
                plot(x_plot[:,0],u_net.predict(x_plot),'r')
                plot(x_plot[:,0],true_solution(x_plot),'b')
                plot(x_plot[:,0],select_net1.predict(x_plot),'g')
                show()
                pause(0.02)
        

    # Save loss and L2 error
    lossseq[k] = loss.item()
    if flag_l2error == True:
        l2error = evaluate_rel_l2_error(u_net, x_test)
        l2errorseq[k] = l2error
    if flag_maxerror == True:
        maxerror = evaluate_rel_max_error(u_net, x_test)
        maxerrorseq[k] = maxerror
    if flag_givenpts_l2error == True:
        givenpts_l2error = evaluate_rel_l2_error(u_net, given_pts)
        givenpts_l2errorseq[k] = givenpts_l2error
    if flag_givenpts_maxerror == True:
        givenpts_maxerror = evaluate_max_error(u_net, given_pts)
        givenpts_maxerrorseq[k] = givenpts_maxerror
    
    ## Show information
    if k%n_epoch_show_info==0:
        if flag_compute_loss_each_epoch:
            print("epoch = %d, loss = %2.5f" %(k,loss.item()), end='')
            if flag_IBC_in_loss == True:
                print(", loss1 = %2.5f, loss2 = %2.5f" %(loss1.item(),loss2.item()), end='')
            print('')
        else:
            print("epoch = %d" % k, end='')
        if flag_l2error == True:
            print("l2 error = %2.3e" % l2error, end='')
        if flag_maxerror == True:
            print(", max error = %2.3e" % maxerror, end='')
        if flag_givenpts_l2error == True:
            print(", givenpts l2 error = %2.3e" % givenpts_l2error, end='')
        if flag_givenpts_maxerror == True:
            print(", givenpts max error = %2.3e" % givenpts_maxerror, end='')
        print("\n")
    
    if flag_preiteration_by_small_lr == True and k == 0:
        flag_start_normal_training = True
        if flag_l2error == True and l2error>0.9:
            flag_start_normal_training = False
        if flag_maxerror == True and maxerror>0.9:
            flag_start_normal_training = False
        if flag_start_normal_training == True:
            k = 1
    else:
        k = k + 1

    if k % n_epoch_plot == 0 and flag_can_plot_dim2 and flag_plot_solution:

        ## plot the network
        solution_network_name_now =  solution_network_name +'_' + str(k)
        seltection_network_1_name_now = seltection_network_1_name +'_' +str(k)
        
        seltection_network_2_name_now = seltection_network_2_name + '_'+str(k)
        plot_the_network(u_net,save_path,solution_network_name_now,X1,Y1,Z)

        if flag_plot_selection_network:
            plot_the_network(select_net1,save_path,seltection_network_1_name_now)
            plot_the_network(select_net2,save_path,seltection_network_2_name_now)


    if k % n_epoch_plot_loss ==0:        
        l1 = list(lossseq)
        l2 = list(l2errorseq)
        l3 = list(maxerrorseq)
        plotloss(l1[0:k],l2[0:k],l3[0:k])

        

## save the learnt weights of the model
save_solution_network_path = save_network_path + solution_network_name + '.pkl'
#seltection_network_1_path = save_network_path + seltection_network_1_name + '.pkl'
#seltection_network_2_path = save_network_path + seltection_network_2_name + '.pkl'
torch.save(u_net.state_dict(), save_solution_network_path) # save the solution network
#torch.save(select_net1.state_dict(), seltection_network_1_path)
#torch.save(select_net2.state_dict(), seltection_network_2_path)


#print the minimal L2 error
if flag_l2error == True:
    print('min l2 error =  %2.3e,  ' % min(l2errorseq), end='')
if flag_maxerror == True:
    print('min max error =  %2.3e,  ' % min(maxerrorseq), end='')
if flag_givenpts_l2error == True:
    print('min givenpts l2 error =  %2.3e,  ' % min(givenpts_l2errorseq), end='')
if flag_givenpts_maxerror == True:
    print('min givenpts max error =  %2.3e,  ' % min(givenpts_maxerrorseq), end='')

if flag_output_results == True:
    #save the data
    localtime = time.localtime(time.time())
    time_text = str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)
    filename = 'result_d_'+str(d)+'_'+time_text+'.data'
    lossseq_and_errorseq = zeros((5,n_epoch))
    lossseq_and_errorseq[0,:] = lossseq
    lossseq_and_errorseq[1,:] = l2errorseq
    lossseq_and_errorseq[2,:] = maxerrorseq
    lossseq_and_errorseq[3,:] = givenpts_l2errorseq
    lossseq_and_errorseq[4,:] = givenpts_maxerrorseq
    f = open(filename, 'wb')
    pickle.dump(lossseq_and_errorseq, f)
    f.close()
    
    #save the solution and error mesh
    if (not time_dependent_type == 'none') or (d>=2):
        N_plot = 101
        if time_dependent_type == 'none':
            if domain_shape == 'cube':
                x0_plot = linspace(domain_intervals[0,0],domain_intervals[0,1],N_plot)
                x1_plot = linspace(domain_intervals[1,0],domain_intervals[1,1],N_plot)
            elif domain_shape == 'sphere':
                x0_plot = linspace(-R,R,N_plot)
                x1_plot = linspace(-R,R,N_plot)
            [X0_plot,X1_plot] = meshgrid(x0_plot,x1_plot)
            x_plot = zeros((N_plot*N_plot,d))
            x_plot[:,0] = X0_plot.reshape((N_plot*N_plot))
            x_plot[:,1] = X1_plot.reshape((N_plot*N_plot))
        else:
            t_plot = linspace(T0,T1,N_plot)
            if domain_shape == 'cube':
                x0_plot = linspace(domain_intervals[0,0],domain_intervals[0,1],N_plot)
            elif domain_shape == 'sphere':
                x0_plot = linspace(-R,R,N_plot)
            [T_plot,X0_plot] = meshgrid(t_plot,x0_plot)
            x_plot = zeros((N_plot*N_plot,d+1))
            x_plot[:,0] = T_plot.reshape((N_plot*N_plot))
            x_plot[:,1] = X0_plot.reshape((N_plot*N_plot))
        
        u_net_plot = u_net.predict(x_plot)
        u_net_plot = u_net_plot.reshape((N_plot,N_plot))       
        u_exact_plot = true_solution(x_plot)
        u_exact_plot = u_exact_plot.reshape((N_plot,N_plot))
        select_net_plot = select_net1.predict(x_plot)
        select_net_plot = select_net_plot.reshape((N_plot,N_plot))      
        f = open('result_d_'+str(d)+'_u_net_'+time_text+'.data', 'wb')
        pickle.dump(u_net_plot, f)
        f.close()
        f = open('result_d_'+str(d)+'_u_exact_'+time_text+'.data', 'wb')
        pickle.dump(u_exact_plot, f)
        f.close()
        f = open('result_d_'+str(d)+'_select_net_'+time_text+'.data', 'wb')
        pickle.dump(select_net_plot, f)
        f.close()
    
    #save parameters
    text = 'Parameters:\n'
    text = text + 'd = ' + str(d) +'\n'
    text = text + 'm = ' + str(m) +'\n'
    text = text + 'n_epoch = ' + str(n_epoch) +'\n'
    text = text + 'N_inside_train = ' + str(N_inside_train) +'\n'
    text = text + 'n_update_each_batch = ' + str(n_update_each_batch) +'\n'
    text = text + 'lrseq[0] = ' + str(lrseq[0]) +'\n'
    text = text + 'lrseq[-1] = ' + str(lrseq[-1]) +'\n'
    text = text + 'lambda_term = ' + str(lambda_term) +'\n'
    text = text + 'activation = ' + activation +'\n'
    text = text + 'boundary_control = ' + boundary_control +'\n'
    text = text + 'flag_preiteration_by_small_lr = ' + str(flag_preiteration_by_small_lr) +'\n'
    text = text + 'lr_pre = ' + str(lr_pre) +'\n'
    text = text + 'n_update_each_batch_pre = ' + str(n_update_each_batch_pre) +'\n'
    text = text + 'h_Du_t = ' + str(h_Du_t) +'\n'
    
    text = text + 'm_sn = ' + str(m_sn) +'\n'
    text = text + 'penalty_parameter = ' + str(penalty_parameter) +'\n'
    text = text + 'maxvalue_sn = ' + str(maxvalue_sn) +'\n'
    text = text + 'minvalue_sn = ' + str(minvalue_sn) +'\n'
    text = text + 'lr_sn = ' + str(lr_sn) +'\n'
    text = text + 'n_update_each_batch_sn = ' + str(n_update_each_batch_sn) +'\n'
    text = text + 'loss_sn1_threshold = ' + str(loss_sn1_threshold) +'\n'
    text = text + 'loss_sn2_threshold = ' + str(loss_sn2_threshold) +'\n'
    text = text + 'flag_reset_select_net_each_epoch = ' + str(flag_reset_select_net_each_epoch) +'\n'
    text = text + 'selectnet_initial_constant = ' + str(selectnet_initial_constant) +'\n'

    text = text + '\n'+'min loss = ' + str(min(lossseq)) + ', '
    if flag_l2error == True:
        text = text + 'min l2 error = ' + str(min(l2errorseq)) + ', '
    if flag_maxerror == True:
        text = text + 'min max error = ' + str(min(maxerrorseq)) + ', '
    if flag_givenpts_l2error == True:
        text = text + 'min givenpts l2 error = ' + str(min(givenpts_l2errorseq)) + ', '
    if flag_givenpts_maxerror == True:
        text = text + 'min givenpts max error = ' + str(min(givenpts_maxerrorseq)) + ', '
    with open('Parameters_'+time_text+'.log','w') as f:   
        f.write(text)  
    
# empty the GPU memory

torch.cuda.empty_cache()



