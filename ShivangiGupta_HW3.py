#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 00:04:18 2018

@author: shivangi
"""

import numpy as np

import matplotlib.pyplot as plt

from numpy.linalg import inv

#Linear Discriminant Analysis
def LDA(m1,m2,sigma):
    theta_0=(np.dot(m1,np.dot(sigma,m1.T)))-(np.dot(m2,np.dot(sigma,m2.T)))
    theta_1, theta_2=(np.dot(sigma,(m2.T-m1.T)))-(np.dot((m1-m2),sigma))
    
    print("LDA Boundary Equation: \n ({0})*X1 + ({1})*X2 + ({2}) = 0 \n".format(theta_1,theta_2,theta_0))

    x=np.linspace(-3,1.5,200)
    line=[-((theta_1*val+theta_0)/theta_2) for val in x]
    plt.plot(x,line,color='red',linewidth=1)
    plt.xlim(-2.1,1.1)
    
    
#Quadratic Discriminant Analysis
def QDA(m1,m2,sigma1,sigma2):
    theta0=(np.dot(m1,np.dot(sigma1,m1.T)))-(np.dot(m2,np.dot(sigma2,m2.T)))
    theta1, theta2=((np.dot(sigma2,m2.T))-(np.dot(sigma1,m1.T)))-((np.dot(m1,sigma1))-(np.dot(m2,sigma2)))
    sigma=sigma1-sigma2
    theta1_sq=sigma[0][0]
    theta2_sq=sigma[1][1]
    theta1_theta2=sigma[0][1]+sigma[1][0]
    
    print("QDA Boundary Equation: \n ({0})*X1^2 + ({1})*X2^2 + ({2})*X1*X2 + ({3})*X1 + ({4})*X2 + ({5}) = 0 \n".format(theta1_sq,theta2_sq,theta1_theta2,theta1,theta2,theta0))
    x1=np.linspace(-3,1.5,200)
    x2=np.linspace(-1.5,5,200)
    X1,X2=np.meshgrid(x1,x2)
    Eq=theta1_sq*(  X1**2)+theta2_sq*(X2**2)+theta1_theta2*X1*X2+theta1*X1+theta2*X2+theta0
    plt.contour(X1,X2,Eq,[0])
    
    
    
#Main Code
x = np.linspace(0,1,200)

y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]

x[100:200] = np.cos(4*np.pi*x)[100:200]

y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)

label= np.ones_like(x)

label[0:100]=0

plt.scatter(x,y,c=label)


c1=np.vstack((x[0:100],y[0:100])).T
c2=np.vstack((x[100:200],y[100:200])).T
data=np.vstack((x,y)).T

mean_c1=np.mean(c1,axis=0)
mean_c2=np.mean(c2,axis=0)

cov_c1=np.cov(c1.T)
cov_c2=np.cov(c2.T)
cov=np.cov(data.T)

sigma1=inv(cov_c1)
sigma2=inv(cov_c2)
sigma=inv(cov)

LDA(mean_c1,mean_c2,sigma)
QDA(mean_c1,mean_c2,sigma1,sigma2)

plt.title("LDA, QDA Decision Boundary")
plt.xlabel("X")
plt.xlim(-3,1)
plt.ylim(-1,5)
plt.ylabel("Y")








