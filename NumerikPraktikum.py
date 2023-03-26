import tkinter as tk
import math
from tracemalloc import start
import numpy as np
import traceback
# delta=0.00001
point=np.array([-3.0,10.0 ])
delta_2=0.01
def twentynine(input_array,params):
    l=params[0]
    output=np.array([])
    sum=np.sum(input_array)
    for x in input_array:
        value=x-math.exp(l*math.cos(x*sum))
        output=np.append(output,value)
    return output
g=lambda x: twentynine(x,[1])

def four_dot_nine(input_array,params):
    p=params[0]
    s=np.sum(input_array)
    x=input_array
    output=np.array([x[0]+x[3]-3,2*x[0]+x[1]+x[3]+x[6]+x[7]+x[8]+2*x[9]-p,2*x[1]+2*x[4]+x[5]+x[6]-8,
    2*x[2]+x[8]-4*p,
    x[0]*x[4]-0.193*x[1]*x[3],
    (x[5]**2)*x[0]-0.67444*(10**(-5))*x[1]*x[3]*s,
    (x[6]**2)*x[3]-0.1189*(10**(-4))*x[0]*x[1]*s,
    x[7]*x[3]-0.1799*(10**(-4))*x[0]*s,
    ((x[8]*x[3])**2)-0.4644*(10**(-7))*(x[0]**2)*x[2]*s,
    x[9]*(x[3]**2)-0.3846*(10**(-4))*(x[0]**2)*s])
    return output

def Brusselator(input_array,params):
    x=input_array[0]
    y=input_array[1]
    a=params[0]
    b=params[1]
    return np.array([(a-(b+1)*x+(x**2)*y),b*x-(x**2)*y])
f=lambda x: Brusselator(x,[1,1])


def apply_partial_diff(f,delta,point,direction):
    transformed_array=point.copy()
    transformed_array[direction]+=delta
    output=np.add(f(transformed_array),float(-1)*f(point))
    output=output/delta
    # print("jo")
    return np.array([output])

def compute_jacobian(f,delta,point):
    d=0
    output=apply_partial_diff(f,delta,point,d)
    while True:
        try:
            d+=1
            output=np.append(output,apply_partial_diff(f,delta,point,d),axis=0)
        except IndexError:
            break
    output=np.transpose(output)
    return output


def easy_newton(f,delta,start_point,cutoff=5000,acc=0.001):
    residual_array=np.array([math.sqrt(np.dot(f(start_point),f(start_point)))])
    differences_array=np.array([0.0])
    point_array=[np.array(start_point)]
    jac=compute_jacobian(f,delta,start_point)
    inv_jac=np.linalg.inv(jac)
    x=start_point
    y=0
    i=0
    while True:
        if i==cutoff:
            return [cutoff,residual_array,differences_array]
        i+=1
        y=x-np.dot(inv_jac,f(x))
        # print("der aktuelle Punkt ist")
        # print(y)
        point_array.append(y)
        residual_array=np.append(residual_array,[math.sqrt(np.dot(f(y),f(y)))])
        diff=x-y
        differences_array=np.append(differences_array,[math.sqrt(np.dot((diff),(diff)))])
        if math.sqrt(np.dot(f(y),f(y)))<=acc:
            break
        x=y
        # 'except:
        #     traceback.print_exc()
        #     i=cutoff
        #     break'
        
    # print("die antwort ist \n")
    # print(y)
    # print("f(x) ist ungefähr \n")
    # print(f(y))
    return [i,residual_array,differences_array,point_array]

def normal_newton(f,delta,start_point,cutoff=5000,acc=0.001):
    x=start_point
    y=0
    i=0
    residual_array=np.array([math.sqrt(np.dot(f(start_point),f(start_point)))])
    differences_array=np.array([0.0])
    point_array=[np.array(start_point)]
    try:
        while True:
            # print(np.amax(f(x)**2))
            i+=1
            jac=compute_jacobian(f,delta,x)
            inv_jac=np.linalg.inv(jac)
            y=x-np.dot(inv_jac,f(x))
            diff=y-x
            point_array.append(y)
            residual_array=np.append(residual_array,math.sqrt(np.dot(f(y),f(y))))
            differences_array=np.append(differences_array,math.sqrt(np.dot(diff,diff)))
            # print("der aktuelle Punkt ist")
            # print(y)
            if  math.sqrt(np.dot(f(y),f(y)))<=acc:
                break
            if i==cutoff:
                return [cutoff]
            x=y
        # print("die antwort ist \n")
        # print(y)
        # print("f(x) ist ungefähr \n")
        # print(f(y))
    except np.linalg.LinAlgError: 
        return [cutoff]
    return [i,residual_array,differences_array,point_array]

def quasi_newton(f,delta,start_point,cutoff=5000,acc=0.001):
    residual_array=np.array([math.sqrt(np.dot(f(start_point),f(start_point)))])
    differences_array=np.array([0.0])
    point_array=[np.array(start_point)]
    jac=compute_jacobian(f,delta,start_point)
    inv_jac=np.linalg.inv(jac)
    dimension=inv_jac.shape[0] 
    
    # x=start_point
    # y=start_point-np.dot(inv_jac,f(start_point))
    # z=0
    # A_inv_x=inv_jac
    # delta_bar_x=0
    # delta_x=(-1)*np.dot(inv_jac,f(x))
    # delta_bar_y=(-1)*np.dot(inv_jac,y)
    
    # delta_y=(delta_bar_y/(1-((np.dot(delta_bar_y,delta_x))/(np.dot(delta_x,delta_x)))))

    # A_inv_y=np.dot((np.identity(dimension)+(np.outer(delta_y,delta_x)/(np.dot(delta_x,delta_x)))),A_inv_x)

    # A_inv_z=0
    # delta_z=0
    # delta_bar_z=0

    delta_x=(-1)*np.dot(inv_jac,f(start_point))
    A_x=inv_jac
    x=start_point
    i=0
    while True:
        i+=1
        y=x+delta_x
        # print("der aktuelle Punkt ist")
        # print(y)
        # print(f(x))
        delta_bar_y=(-1)*np.dot(A_x,f(y))
        alpha=np.dot(delta_bar_y,delta_x)*(1/np.dot(delta_x,delta_x))
        delta_y=delta_bar_y/(1-alpha)
        matrix=(np.outer(delta_y,delta_x)*(1/np.dot(delta_x,delta_x)))
        A_y=np.dot(matrix,A_x)+A_x
        # print(np.amax(f(x)**2))
        diff=y-x
        point_array.append(y)
        residual_array=np.append(residual_array,math.sqrt(np.dot(f(y),f(y))))
        differences_array=np.append(differences_array,math.sqrt(np.dot(diff,diff)))
        if i==cutoff:
            return [cutoff]
        if math.sqrt(np.dot(f(y),f(y)))<=acc:
            # print(f(y))
            break
        x=y
        delta_x=delta_y
        A_x=A_y
        # except np.linalg.LinAlgError:
        #     traceback.print_exc()
        #     return cutoff

    # print("this is old quasi newton")
    # print("die antwort ist \n")
    # print(y)
    # print("f(x) ist ungefähr \n")
    # print(f(y))

    return [i,residual_array,differences_array,point_array]

# def new_quasi_newton(f,delta,start_point,cutoff=5000):
#     jac=compute_jacobian(f,delta,start_point)
#     i=0
#     x=start_point
#     inv_jac=np.linalg.inv(jac)
#     y=start_point-np.dot(inv_jac,f(start_point))
#     Jn=jac
#     while True:
#         i=i+1
#         midterm=f(y)-f(x)-np.dot(Jn,y-x)
#         parameter=(1/np.dot(y-x,y-x))
#         midterm=midterm*parameter
#         midterm=np.outer(midterm,y-x)
#         Jn=Jn+midterm
#         # print(i)
#         z=y-np.dot(np.linalg.inv(Jn),f(y))
#         if np.abs(f(y)).max()<=delta_2:
#             break
#         if i==cutoff:
#             return cutoff
#         x=y
#         y=z
#     print("this is new quasi newton")
#     print("die antwort ist \n")
#     print(y)
#     print("f(x) ist ungefähr \n")
#     print(f(y))
#     return i
#     pass

# quasi_array=[]
# normal_array=[]
# easy_array=[]
# delta=0.000001
# zeros=np.zeros(10)
# ones=[1,1,1,1,1,1,1,1,1,1]
# for i in np.arange (1.0,3.0,0.1):
#     l=lambda x: four_dot_nine(x,[i])
#     print(i)
#     print(quasi_newton(l,delta,ones))
#     print("quasi done")
#     normal_array.append(normal_newton(l,delta,ones))
#     print("normal done")
#     easy_array.append(easy_newton(l,delta,ones))
#     print("easy done")

# # def gogogo():
#     global procedure_var
#     global which_var
#     if which_var.get()==0:
#         p=[float(Elambda.get())]
#         f=lambda x: twentynine(x,p)
#     if which_var.get()==1:
#         p=Elambda.get().splilt(" ")
#         p=[float(x) for x in p]
#         f= lambda x: Brusselator(x,p)
#     if which_var.get()==2:
#         p=[float(Elambda.get())]
#         f=lambda x: four_dot_nine(x,p)

    
#     start_point_string=Estart.get()
#     start_point=start_point_string.split(" ")
#     start_point=[float(x) for x in start_point]
#     start_point=np.array(start_point)

#     if procedure_var.get()==0:
#         easy_newton(f,0.00001,start_point)
#     if procedure_var.get()==1:
#         normal_newton(f,0.00001,start_point)
#     if procedure_var.get()==2:
#         quasi_newton(f,0.00001,start_point)
    


# root=tk.Tk()
# which_var=tk.IntVar()

# SLabel=tk.Label(root,text="Welche Aufgabe soll bearbeitet werden")
# SLabel.grid(column=0,row=0)

# Radio1=tk.Radiobutton(root,text="Beispiel 4.30",variable=which_var,value=0)
# Radio2=tk.Radiobutton(root,text="Beispiel 4.31",variable=which_var,value=1)
# Radio3=tk.Radiobutton(root,text="Aufgabe 4.9",variable=which_var,value=2)
# Radio1.grid(column=0,row=1)
# Radio2.grid(column=0,row=2)
# Radio3.grid(column=0,row=3)

# procedure_var=tk.IntVar()

# PLabel=tk.Label(root,text="Welches Verfahren soll angewendet werden?")
# PLabel.grid(column=1,row=0)

# PRadio1=tk.Radiobutton(root,text="vereinfachtes Newton-Verfahren",variable=procedure_var,value=0)
# PRadio2=tk.Radiobutton(root,text="normales Newton-Verfahren",variable=procedure_var,value=1)
# PRadio3=tk.Radiobutton(root,text="Quasi-Newton-Verfahren",variable=procedure_var,value=2)

# PRadio1.grid(column=1,row=1)
# PRadio2.grid(column=1,row=2)
# PRadio3.grid(column=1,row=3)

# lambda_label=tk.Label(root,text="wie groß soll der Parameter sein?\n (bei Beispiel 4.31 zwei Parameter mit Leerzeichen getrennt eingeben)")
# lambda_label.grid(column=2,row=0)

# Elambda=tk.Entry(root)
# Elambda.grid(column=2,row=1)


# start_label=tk.Label(root,text="die Startkoordinaten (mit Leerzeichen getrennt)")
# start_label.grid(column=2,row=2)
# Estart=tk.Entry(root)
# Estart.grid(column=2,row=3)

# myButton=tk.Button(root,text="Los",command=gogogo)
# myButton.grid(row=4,column=1)

# ones=[1,1,1,1,1,1,1,1,1,1]
# l=lambda x: four_dot_nine(x,[3.213])
# print(normal_newton(l,delta,ones))


# root.mainloop()

# quit()