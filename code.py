# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 23:56:26 2023

@author: Gideon
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.rcParams['figure.dpi']="100" 

grid_point=100
domain_size=1
n_iterations=100
kinematic_viscoscity=0.5
density=1
vel=10

 

pressure_poisson_iterations=100
#@jit (nopython=True)

element_length=domain_size/(grid_point-1)
x=np.linspace(0.0,domain_size,grid_point)
y=np.linspace(0.0,domain_size,grid_point)

X,Y=np.meshgrid(x,y)

################## Time Step Limiting ##############
time_step = min(0.25*(element_length**2)/kinematic_viscoscity,
              4.0*kinematic_viscoscity/(vel**2))

## Initial Conditions ##



####################### Central diff method discretized for Square Mesh ###########################

def central_diff_x(f):   
    diff=np.zeros_like(f)
    diff[1:-1,1:-1]=(
        
       f[1:-1 , 2: ] - f[1:-1, 0:-2]
        
    )/(
        2*element_length
    )
    return diff

def central_diff_y(f):   
    diff=np.zeros_like(f)
    diff[1:-1, 1:-1]=(
    
        f[2: , 1:-1] - f[0:-2 , 1:-1]
    
    )/(
        2*element_length
    )
    return diff



######################### Laplace operator discretized for Square Mesh #############################
 
def laplace(f):    
    diff=np.zeros_like(f)
    diff[1:-1,1:-1]=(
        f[1:-1, 0:-2]
        +
        f[0:-2,1:-1]
        -
        4
        *
        f[1:-1,1:-1]
        +
        f[1:-1,2: ]
        +
        f[2: ,1:-1]
    )/(
    element_length**2
    )
    return diff




def main():
       
        
    u_prev=np.zeros_like(X)
    v_prev=np.zeros_like(X)
    p_prev=np.zeros_like(X)    
    
    for _ in tqdm(range(n_iterations)):
        
        
        u_prev[ 0, :]=0.0
        u_prev[ :, 0]=0.0
        u_prev[:, -1]=0.0
        u_prev[-1, :]=0.0
        v_prev[ 0, :]=0.0
        v_prev[ :, 0]=0.0
        v_prev[ :,-1]=0.0
        v_prev[-1, :]=0.0
        
        
    
        u_prev[40:60, 0]= vel
        u_prev[48:52,-1]= vel
        
        ######################### Solving for velocity ignoring the Pressure Gradient #############################      
       
        u_int=(
            
                u_prev
                +
                time_step*(
                  -  
                  (
                     
                       u_prev*central_diff_x(u_prev)
                       +
                       v_prev*central_diff_y(u_prev)
                  )
                  +
                  kinematic_viscoscity*laplace(u_prev)
                )
            )

        v_int=(
            
                v_prev
                +
                time_step*(
                  - 
                  (
                       u_prev*central_diff_x(v_prev)
                       +
                       v_prev*central_diff_y(v_prev)
                  )
                  +
                  kinematic_viscoscity*laplace(v_prev)
                )
            )
        
       
        
        ######################### Pressure Correction Equation #############################
        
        RHS=(
            density/time_step
            *
            (
                central_diff_x(u_int) + central_diff_y(v_int)
            )
        )
        
        for _ in range(pressure_poisson_iterations):
            p_next=np.zeros_like(p_prev)
            p_next[1:-1,1:-1]=1/4*(
                +
                p_prev[1:-1,0:-2]
                +
                p_prev[0:-2,1:-1]
                +
                p_prev[1:-1, 2: ]
                +
                p_prev[2: , 1:-1]
                -
                element_length**2
                *
                RHS[1:-1, 1:-1]
                
            )
            
        ######################### Pressure Boundary Conditions (Neumann) #############################
            p_next[:,-1]=p_next[:,-2]
            p_next[0,:]=p_next[1,:]
            p_next[:,0]=p_next[:,1]
            p_next[-1,:]=p_next[0,:]
            
            p_prev=p_next
                    
        ######################### Velocity Correction Step #############################
        
        u_next=(
            u_int
            -
            time_step/density
            *
            central_diff_x(p_next)
        )
        
        v_next=(
            v_int
            -
            time_step/density
            *
            central_diff_y(p_next)
        )
        
        ######################### Channel Boundary Conditions #############################
      
        u_next[8:12,20:80]=0
        v_next[8:12,20:80]=0
        
        u_next[18:22,20:80]=0   
        v_next[18:22,20:80]=0
        
        u_next[28:32,20:80]=0   
        v_next[28:32,20:80]=0
        
        u_next[38:42,20:80]=0
        v_next[38:42,20:80]=0
        
        u_next[48:52,20:80]=0
        v_next[48:52,20:80]=0
        
        u_next[58:62,20:80]=0
        v_next[58:62,20:80]=0
        
        u_next[68:72,20:80]=0
        v_next[68:72,20:80]=0
        
        u_next[78:82,20:80]=0
        v_next[78:82,20:80]=0
        
        u_next[88:92,20:80]=0
        v_next[88:92,20:80]=0
        

        ######################### Advance In Time #############################
        
        u_prev=u_next
        v_prev=v_next
        p_prev=p_next
            

    plt.figure()
     
    plt.contourf(X,Y,p_next)
    plt.colorbar()

    plt.streamplot(X,Y,u_next,v_next,color="red",density=5,linewidth=0.3)  
    
    ######################### Channel Boundary Display #############################
    
    plt.axhspan(0.08,0.12,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    plt.axhspan(0.18,0.22,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    plt.axhspan(0.28,0.32,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    plt.axhspan(0.38,0.42,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    plt.axhspan(0.48,0.52,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    plt.axhspan(0.58,0.62,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    plt.axhspan(0.68,0.72,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    plt.axhspan(0.78,0.82,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    plt.axhspan(0.88,0.92,0.25,0.8 , color='black', alpha=0.75, lw=0.1)
    
    
    plt.show()
    
   
if __name__ =="__main__":
       main()     
        
        

    
    
    
