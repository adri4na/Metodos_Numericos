#! /usr/bin/python3

from numpy import *

import matplotlib.pyplot as plt # from pylab import plot,show
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings("ignore")


################################################
################################################
################################################
################################################
'''
            FUNCIONES PARA EL PROBLEMA 1 
'''
################################################

#Comienzo a graficar en un rango "grande" para ver aproximadamente donde se encuentra una raiz
#primero analicé la imagen a gran escala y luego decidí por este rango pequeño de la imagen

def err_ap(xup,xlow):                  #funcion para el error aparente
    return abs((xup-xlow)/xup)

def fun_pot_normalizado(x): 
    return (1/x)**6 - exp(-x)

def derivfun_pot_normalizado(x): 
    return -6*(1/x)**7 + exp(-x)


'''
    METODO DE BISECCION
'''          
def biseccion(ai,bi,f): #rango inicial dado [ai,bi], f:funcion ingresada    
    
    E_bisec=[]   #array para errores

    ea_bsx = 1
    ep = 0.00001
    while ea_bsx > ep :
        m = (bi+ai)/2
        if f(ai)*f(m)<0 :
            bi=m
        if f(m)*f(bi)<0 :
            ai=m
        ea_bsx = err_ap(bi,ai)
        E_bisec.append(ea_bsx)
            
    return(E_bisec,ai,bi,ep)

#####################
#   ITEM 1.3.1      
#####################
'''
    REGRESION LINEAL
'''
def coeficientes_linreg(x,y): #halla los coefs de la ecuacion f= a0 + a1x 
    
    n = len(x)   
    
    Zx = 0 #para almacenar la suma de todos los x
    for i in range(n):
        Zx+= x[i]
    x_media = Zx/n
    
    Zy = 0 #para almacenar la suma de todos los y
    for i in range(n):
        Zy+= y[i]
    y_media = Zy/n
    
    temp=0
    for i in range(n):
        temp += x[i]**2
    Zcuadx = temp
    
    temp=0
    for i in range(n):
        temp += x[i]*y[i]
    Zprod = temp
    
    #hallando a0 y a1
    a1 = (n*Zprod - Zx*Zy)/(n*Zcuadx - Zx**2)
    
    a0 = y_media - a1*x_media
    
    return(a0,a1)


#####################
#    ITEM 1.2
#####################
def f_ex2_pot_normalizado(x): 
    return (1/x)**6 - exp(-x) + 0.1 #se hará 0 cuando la funcion f(x)=-0.1

'''
    NEWTON-RAPHSON
'''
def newton_raphson(ep,xor,f,derivada): #xor:valor inicial aproximado == guess,
    #f:funcion a la que se quiere aplicar el metodo  #derivada: derivada de f
    E_nr=[]#array para errores
    e_nr = 1
    while e_nr > ep:
        df=derivada(xor)
        xnr= xor - f(xor)/(df)
        e_nr=err_ap(xnr,xor)
        E_nr.append(e_nr)
        xd=xor                 #para poder imprimir el valor
        xor=xnr
        
    return(xd,xnr,E_nr)



################################################
################################################
################################################
################################################
'''
            FUNCIONES PARA EL PROBLEMA 2 
'''
################################################
def f_ptolagrge(x):
    return 3.987*(10**14)*((3.844*(10**8)-x)**2) - 4.92408*(10**12)*(x**2) - (7.086244*(10**-12))*(x**3)*((3.844*(10**8)-x)**2)

def deriv_fptolag(x):
    return -7.974*(10**14)*(3.844*(10**8)-x)-9.84816*(10**12)*x-(7.086244*(10**-12))*((5*x**4)-3.0752*(10**9)*(x**3)+4.4329*(10**9)*x**2)




################################################
################################################
################################################
################################################
'''
            FUNCIONES PARA EL PROBLEMA 3 
'''
################################################

def imprimir_problema(M,N):  #funcion que imprime las matrices del problema de forma ordenada
    lineas = len(M)
    #creando matriz de variables
    X=[]
    linea_igual = lineas % 2
    if linea_igual != 0 :
        linea_igual = 0.5 + lineas/2
    for k in range(lineas):
        p= "x"+str(k)
        X.append(p)
    for i in range (lineas):
        if i == (linea_igual-1):
            print(M[i]," ","[",X[i],"]","=","[",N[i],"]")
        else : 
            print(M[i]," ","[",X[i],"]"," ","[",N[i],"]") 
    return " "

'''
    ELIMINACION DE GAUSS
'''
def Elim_Gauss(Eqs_a,Eqs_b):
    
    rangoA= len(Eqs_b)
    
    for i in range(0, rangoA-1,1):
        
        for k in range(i+1, rangoA,1):
            #if Eqs_a[i,i] != 0.0: ya se analizo que siempre es != 0.0
                p_elem = Eqs_a[k][i]/ Eqs_a[i][i]  #se maneja las filas dividiendolas entre el valor deel elemento diagonal 
                for j in range (0,rangoA,1):
                    Eqs_a[k][j] =  Eqs_a[k][j] - p_elem*Eqs_a[i][j]
                Eqs_b[k]=Eqs_b[k]-p_elem*Eqs_b[i]
    print("Las nuevas matrices serán")
    print("M=",Eqs_a)
    print("N=",Eqs_b)
    return ("")


def sust_inv(M,N):
    rangoM= len(M)
    voltgs = zeros(rangoM)  #se inicia array para los valores finales
    for i in range(rangoM-1,-1, -1):   #se hace conteo inverso
            temp = N[i]
            for k in range(rangoM-1,i, -1):
                temp -= M[i][k]*voltgs[k]
            voltgs[i] = temp/M[i][i]    
    return voltgs


################################################
################################################
################################################
################################################
'''
            FUNCIONES PARA EL PROBLEMA 4 
'''
################################################
'''
    DESCOMPOSICION LU
'''
def Descomp_LU(M,N):
    
    #hallamos las matrices L y U

    dim_M = M.shape
    rangoM = len(M)
    L = zeros(dim_M) #inicio matriz L con tamaño dim_M=(rangoM,rangoM) por ser M cuadrada
    U = zeros(dim_M) #inicio U
    for i in range(rangoM):
        L[i,i] = 1.0    #llena la diagonal de L de 1's
        for j in range(i+1,rangoM):
            L[j,i] = M[j,i]/M[i,i]  #Genera la fila j de la matriz
            for k in range(i+1,rangoM):
                M[j,k] = M[j,k] - L[j,i]*M[i,k]
        for k in range(i,rangoM):
            U[i,k] = M[i,k]  #genera la fila i de la matriz U
        
    #obtenemos la matriz d (de la eq Ld = b)
    #para esta funcion: b = N
    #usamos la sustitución forward 
    
    d = []  #inicio la matriz d
    rangoN = len(N)
    for i in range(rangoN):
        temp = 0
        d.append(N[i])
        for j in range(i):
            temp = temp + L[i, j]*d[j]
        d[i] = (N[i] - temp)/L[i, i]
       
    
    #obtenida d, hallamos x (de la eq Ux = d)
    #usamos sustitución backwards/sust_inv del ejercicio anterior
    
    x = zeros(rangoM)  #se inicia matriz x
    for i in range(rangoM-1,-1, -1):   #se hace conteo inverso
            temp = d[i]
            for k in range(rangoM-1,i, -1):
                temp -= U[i][k]*x[k]
            x[i] = temp/U[i][i]  
    return (x)

'''
    REGRESION MULTIPLE
'''
def reg_pol3(y,x1,x2): #regresion pol para 2 variables indp 1 dep
    #para determinar la suma de residuos:   
    #generamos la matriz para obtener los ai's
    i=0
    j=0
    A=zeros((3, 3)) #por ser regresion multiple de 3 variables
    
    n= len(y)
    A[i][j]=n
    
    j+=1
    Zx1 = 0 #para almacenar la suma de todos los x1
    for k in range(n):
        Zx1+= x1[k]
    A[i][j]=Zx1
    A[j][i]=Zx1
    
    j+=1
    Zx2 = 0 #para almacenar la suma de todos los x
    for k in range(n):
        Zx2+= x2[k]
    A[i][j]=Zx2
    A[j][i]=Zx2
    
    i+=1
    j-=1
    Zcuadx1=0
    for k in range(n):
        Zcuadx1 += x1[k]**2
    A[i][j]=Zcuadx1 
    
    j+=1
    Zx1x2=0
    for k in range(n):
        Zx1x2 += x1[k]*x2[k]
    A[i][j]=Zx1x2 
    A[j][i]=Zx1x2 
    
    i+=1
    Zcuadx2=0
    for k in range(n):
        Zcuadx2 += x2[k]**2
    A[i][j]=Zcuadx2 
    
    #rellenando b del sistema Ax=b
    b=zeros((3, 1))
    
    j=0
    Zy = 0 #para almacenar la suma de todos los x
    for k in range(n):
        Zy+= y[k]
    b[j][0]=Zy
    
    j+=1
    Zyx1 = 0 #para almacenar la suma de todos los x
    for k in range(n):
        Zyx1+= y[k]*x1[k]
    b[j][0]=Zyx1
    
    j+=1
    Zyx2 = 0 #para almacenar la suma de todos los x
    for k in range(n):
        Zyx2+= y[k]*x2[k]
    b[j][0]=Zyx2
    
    #para resolver el sistema usamos el método de
    X= Descomp_LU(A,b)
    
    a0=X[0]
    a1=X[1]
    a2=X[2]
    return a0,a1,a2
