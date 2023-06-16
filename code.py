"""
Created on Wed Jun 14 23:56:26 2023

@author: Gideon
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

grid_point=100
domain_size=1
n_iterations=500
kinematic_viscoscity=0.1
density=2.0
vel=10



pressure_poisson_iterations=100

def main():
    element_length=domain_size/(grid_point-1)
    x=np.linspace(0.0,domain_size,grid_point)
    y=np.linspace(0.0,domain_size,grid_point)
    
    X,Y=np.meshgrid(x,y)
    
    time_step = min(0.25*element_length*element_length/kinematic_viscoscity,
                  4.0*kinematic_viscoscity/vel/vel)
  
  ## Initial Conditions ##
    
    u_prev=np.zeros_like(X)
    v_prev=np.zeros_like(X)
    p_prev=np.zeros_like(X) 


   ## Descritised Differential Operators ##

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
    
    
        
    
    
    for _ in tqdm(range(n_iterations)):
      
       
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
        
        #### Velocity BCs (Drichlet) ####
        u_int[ 0, :]=0.0
        u_int[ :, 0]=0.0
        u_int[:, -1]=0.0
        u_int[-1, :]=0.0
        v_int[ 0, :]=0.0
        v_int[ :, 0]=0.0
        v_int[ :,-1]=0.0
        v_int[-1, :]=0.0
        
        u_int[40:60,0]=vel
        
        
        
        
        
        
        #### Pressure Correction ####
        
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
            
        #### Pressure Boundary Conditions (Neumann) ####
            p_next[:,-1]=p_next[:,-2]
            p_next[0,:]=p_next[1,:]
            p_next[:,0]=p_next[:,1]
            p_next[-1,:]=p_next[0,:]
        
            p_prev=p_next
            
        
        #### Velocity Correction ####
        
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
        
        #### Velocity BCs (Drichlet) ####
        u_next[ 0, :]=0.0
        u_next[ :, 0]=0.0
        u_next[:, -1]=0.0
        u_next[-1, :]=0.0
        v_next[ 0, :]=0.0
        v_next[ :, 0]=0.0
        v_next[ :,-1]=0.0
        v_next[-1, :]=0.0
        
        u_next[40:60,0]=vel
        
        
        
        
        
        
        
        #### Advance in time ####
        
        u_prev=u_next
        v_prev=v_next
        p_prev=p_next
            
    
        plt.figure()
        plt.contourf(X,Y,p_next)
        plt.colorbar()
    
        plt.streamplot(X,Y,u_next,v_next,color="red")
        
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
        
        
    plt.show()
    
   
if __name__ =="__main__":
       main()     
        
        

    
    
