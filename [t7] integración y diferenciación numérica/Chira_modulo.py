#! /usr/bin/python3

from numpy import *

import matplotlib.pyplot as plt # from pylab import plot,show
from matplotlib.pyplot import figure

import warnings
warnings.filterwarnings("ignore")

'''
############################################################
############################################################
####################      EX. FINAL      ####################
############################################################
############################################################ 
'''

############################################################
####################   EJERCICIO 1      ####################
############################################################

#####################################
'''
    CUADRATURA DE GAUSS - LEGENDRE [para una funcion con parametro variable]
'''
#####################################


#funcion cuadratura GL con solo puntos para n=2,3,4,5 con sus respectivos pesos
def cuad_GL_var(func,a,b,n): #funcion, rango [a,b], # de pts
    
    #puntos
    # GLn=[[0](n=2), [1](n=3), [2](n=4), [3](n=5)] , puntos para diferente n
    GLn=array([(-3**-0.5,3**-0.5),
            (-(3/5)**0.5, 0, (3/5)**0.5),
            (-((3/7) + (2/7)*(6/5)**0.5)**0.5,-((3/7) - (2/7)*(6/5)**0.5)**0.5,((3/7) - (2/7)*(6/5)**0.5)**0.5,((3/7) + (2/7)*(6/5)**0.5)**0.5  ),
            (-1/3*(5 + 2*(10/7)**0.5)**0.5,-1/3*(5 - 2*(10/7)**0.5)**0.5,0,1/3*(5 - 2*(10/7)**0.5)**0.5,1/3*(5 + 2*(10/7)**0.5)**0.5  )]) 
    #pesos
    GLw=array([(1,1),
          (5/9,8/9,5/9),
          ( (18-30**0.5)/36 , (18+30**0.5)/36 , (18+30**0.5)/36 , (18-30**0.5)/36 ),
          ( (322 - 13*70**0.5)/900 , (322 + 13*70**0.5)/900 , 128/225 , (322 + 13*70**0.5)/900 , (322 - 13*70**0.5)/900 )])

    
    flag = n-2
    
    I=0
    for i in range(n):
        I = I + GLw[flag][i]*func(b, 0.5*(b-a)*GLn[flag][i] + (a+b)*0.5 ) #funcion con param variable b
    I= I*(b-a)/2
    
    return I


##repite la cuadratura de Gl para diferentes valores, retorna diversas soluciones de la integral en un intvlo 
def recurr_GL(a,b,n,f): #[a,b]: limites de integracion , n : #pts de funcion
                        #f : funcion a analizar 
    h=0.05 #paso
    
    vals=[] #lista donde se almacenarán los valores despues de la integracion
    xs=[] #lista
    
    #para la funcion del examen graficaremos desde <-b> hasta <b> 
    #pero solo sombrearemos la pare pedidad: [0,b]
    
    #de ida ( [-b,a] )  - por dato sabemos que la funcion es oscilatoria respecto a x=0 
    for i in arange(-b,a,h):
        temp = cuad_GL_var(f,i,a,n)
        vals.append(temp)
        xs.append(i)
        
    #de regreso ( [a,b] )  - por dato sabemos que la funcion es oscilatoria respecto a x=0 
    for i in arange(a+0.001,b+h,h):
        temp = cuad_GL_var(f,a,i,n)
        vals.append(temp)
        xs.append(i)
        
    return xs, vals



############################################################
####################   EJERCICIO 2      ####################
############################################################

##Se usaron las funciones encontradas  [mas abajo]
##func reordenar [en la tarea 8]
##func RK4 para sistemas de ODEs [PC3]



############################################################
####################   EJERCICIO 3      ####################
############################################################

#######################
'''
   Forward Time Central Space-FTCS - Elipticas
'''
#######################
def FTCS_elipticas(alfa,C,ht,hx,at,bt,ax,bx,Ti,cdextr_a,cdextr_b): # ht: paso t, hx:paso x, [at,bt] :rango del tiempo
                                       # [ax,bx] : rango de x    
    lamda=alfa*ht/(hx**2)   
    
    m=1+int((bx-ax)/hx) #nro de pasos para t
    n=1+int((bt-at)/ht)  
    xs=arange(ax,bx+hx,hx)
    ts=arange(at,bt+ht,ht)
    Ts=[]        
    Ts.append(Ti)  
    
    #T para otros t (T para t=i*ht)
    for i in range(1,n):
        Ti=[]
        Ti.append(cdextr_a)
        for j in range(1,m-1):
            temp=(1+ht*C)*Ts[i-1][j] + lamda*( Ts[i-1][j+1] - 2*Ts[i-1][j] + Ts[i-1][j-1] ) 
            Ti.append(temp)       
        Ti.append(cdextr_b)
        Ts.append(Ti)
        
    return xs,ts,Ts
    

#######################
'''
    REGRESION LINEAL
'''
#######################

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



###[FIN]

'''
############################################################
############################################################
####################      TAREA 8       ####################
############################################################
############################################################ 
'''

############################################################
####################   EJERCICIO 1      ####################
############################################################

##########################
'''
    METODO DE RK4
'''
##########################
def met_RK4(a,b,yt0,f,h=0.5):    #limites del rango [a,b], cond inic: y(t=0),h:paso, f:funcion edo del problema
    
    t=array(arange(a,b+h,h))
    lim=len(t)
    y=[]                #lista para almacenar los valores y encontrados
    y.append(yt0)       #se ingresa el valor dato a la lista
    
    for i in range(lim-1):
        
        k1 = f( t[i], y[i] ) 
        k2 = f( t[i] + h/2, y[i] + (h*k1)/2 )
        k3 = f( t[i] + h/2, y[i] + (h*k2)/2 )
        k4 = f( t[i] + h  , y[i] +  h*k3    )
        
        y_temp = y[i] + ( k1 + 2*k2 + 2*k3 + k4 )*h/6        
        y.append(y_temp)
            
    return t,y  


##########################
'''
    METODO DE EULER
'''
##########################

def met_euler(a,b,yt0,f,h=0.5):    #limites del rango [a,b], cond inic: y(t=0), f:funcion del problema
                                   #si no se ingresa h, por default será h=0.5
    t=array(arange(a,b+h,h))  #cambia respecto al paso (debido al float)
    lim=len(t)
    y=[]                #lista para almacenar los valores "y" encontrados
    y.append(yt0)       #se ingresa el valor dato a la lista
    
    for i in range(lim-1):
        y_temp = y[i] + f(t[i],y[i])*h
        y.append(y_temp)
        
    return t,y 

##########################
'''
    METODO DE HEUN (no iterado)
'''
##########################
def met_heun_noit(a,b,yt0,f,h=0.5):    #limites del rango [a,b], cond inic: y(t=0),h:paso, f:funcion del problema
    
    t=array(arange(a,b+h,h))
    lim=len(t)
    y=[]                #lista para almacenar los valores y encontrados
    y.append(yt0)       #se ingresa el valor dato a la lista
    
    for i in range(lim-1):
        
        y_pred = y[i] + f(t[i],y[i])*h  #predictor
        
        y_temp = y[i] + ( f(t[i],y[i]) + f(t[i+1],y_pred))*h/2 #corrector        
        y.append(y_temp)
            
    return t,y  


##########################
'''
    METODO DE RALSON
'''
##########################
def met_ralson(a,b,yt0,f,h=0.5):    #limites del rango [a,b], cond inic: y(t=0),h:paso, f:funcion edo del problema
    
    t=array(arange(a,b+h,h))
    lim=len(t)
    y=[]                #lista para almacenar los valores y encontrados
    y.append(yt0)       #se ingresa el valor dato a la lista
    
    for i in range(lim-1):
        
        k1 = f(t[i],y[i]) 
        k2 = f(t[i]+(3*h)/4,y[i]+(3*h*k1)/4)
        
        y_temp = y[i] + ( k1 + 2*k2 )*h/3        
        y.append(y_temp)
            
    return t,y  

############################################################
####################   EJERCICIO 2      ####################
############################################################

##########################
'''
   ELIMINACION DE GAUSS
'''
##########################

def solucion_met_elim_Gauss(A,B):
    
    M=copy(A)
    N=copy(B) #N es una lista que simula a la matriz N(nx1)
    
    rangoA= len(N)
    
    for i in range(0, rangoA-1,1):
        
        for k in range(i+1, rangoA,1):
            #if M[i,i] != 0.0: ya se analizo que siempre es != 0.0
                p_elem = M[k][i]/ M[i][i]  #se maneja las filas dividiendolas entre el valor deel elemento diagonal 
                for j in range (0,rangoA,1):
                    M[k][j] =  M[k][j] - p_elem*M[i][j]
                N[k]=N[k]-p_elem*N[i]
                
    #inicio de la sustitucion inversa
    rangoM= len(M)
    x = zeros(rangoM)  #se inicia array para los valores finales
    for i in range(rangoM-1,-1, -1):   #se hace conteo inverso
            temp = N[i]
            for k in range(rangoM-1,i, -1):
                temp -= M[i][k]*x[k]
            x[i] = temp/M[i][i]  
            
    return x


#funcion que creará una lista que almacenara los valores de cada variable 
#en un solo elemento-vector
#por ejemplo: si r= [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)], entonces rdist será
#rdist = [(x1,x2,x3),(y1,y2,y3),(z1,z2,z3)]
def reordenar (r): #funcion que separa los datos
    #separamos los datos 
    num_vects_generados=len(r) #nro de datos (vectores) generados con el método
    num_datos_en_vector=len(r[1]) #nro de elementos que contiene cualquier vector en r
    
    rdlist=[] #lista que almacenara los valores de cada variable en un solo elemento-vector
    
    for i in range (num_datos_en_vector):
        elem_temp=[]
        for j in range (num_vects_generados):
            temp1 = r[j][i]
            elem_temp.append(temp1)
        rdlist.append(elem_temp)
    
    return rdlist 


#############################
'''
    METODO DE EULER IMPLICITO
'''
#############################

def met_euler_implicito_sist(a,b,yt0s,f_coefs,h=0.5):   
    #limites del rango [a,b], cond inic: y(t=0), f_coefs:devuelve los coefs del problema
    #si no se ingresa h, por default será h=0.5
    
    t=array(arange(a,b+h,h))  #cambia respecto al paso (debido al float)
    lim=len(t)
    
    y=[]                #lista para almacenar los valores "y" encontrados
    y.append(yt0s)       #se ingresa el valor dato a la lista, yt0s es un vector    
    
    cyis= f_coefs() #obtiene los coeficientes del sistema a resolver    
    
    #creamos el sistema de matrizces M(nxn){yi+1}= N(2xn) del sistema lineal
    n = len(cyis) #orden de la matriz
    M = zeros((n,n))
    N = zeros(n)
        
    for i in range(lim-1):
        
        #ingresando los valores de yi a N
        for r in range(n):
            N[r]= yt0s[r] + h*cyis[r][2] # == yi + ci3*h
        
        #ingresando valores de M
        for k in range(n):
            for j in range(n):
                if j == k :
                    M[k][j]= 1 - h*cyis[k][j]
                else:
                    M[k][j]= -h*cyis[k][j]
                   
        Y = solucion_met_elim_Gauss(M,N)
        
        y.append(list(Y)) #guardamos los datos en la lista de solucion
        
        yt0s = [] #vaciamos la lista        
        #ahora, la reescribimos con los valores encontrados para y_{i+1}
        yt0s = list(Y)
    
    ydist=reordenar(y) #cambiando a array por seguridad                 
          
    return t,ydist


##########################
'''
    METODO DE EULER (exp) PARA SISTEMAS
'''
##########################

def sist_met_euler_exp(a,b,yt0,f,h=0.5):    #limites del rango [a,b], cond inic: y(t=0), f:funcion del problema
                                   #si no se ingresa h, por default será h=0.5
    t=array(arange(a,b+h,h))  #cambia respecto al paso (debido al float)
    lim=len(t)
    y=[]                #lista para almacenar los valores "y" encontrados
    y.append(yt0)       #se ingresa el valor dato a la lista
    n=len(yt0) #nro de variables del sistema
    
    for i in range(lim-1):
        
        y_temp=[]
        vals = f(t[i],yt0)
        
        for k in range(n): 
            temp = y[i][k] + h*vals[k]
            y_temp.append(temp)
        
        yt0=y_temp
        y.append(y_temp)
    
    ydist=reordenar(y)
    
    return t,ydist


##############################
'''
    METODO DE HEUN MODIFICADO (no iterado)
'''
##############################
def heun_modificado_noit(a,b,yt0,f,cvg,h=0.5):
    
    t=array(arange(a,b+h,h))
    lim=len(t)
    y=[]                #lista para almacenar los valores y encontrados
    y.append(yt0[0])       #se ingresa el valor dato a la lista
    y.append(yt0[1])
    k=0 #contador
    e=1000
    for i in range(1,lim-1):   
        y_pred = y[i-1] + 2*f(t[i],y[i])*h  #predictor
        
        y_temp = y[i] + ( f(t[i],y[i]) + f(t[i+1],y_pred))*h/2 #corrector        
        y.append(y_temp)
    return t,y


##############################
'''
    METODO DE HEUN MODIFICADO (cvg)
'''
##############################
def heun_modificado_it(a,b,yt0,f,cvg,h=0.5):
    
    t=array(arange(a,b+h,h))
    lim=len(t)
    y=[]                #lista para almacenar los valores y encontrados
    y.append(yt0[0])       #se ingresa el valor dato a la lista
    y.append(yt0[1])
    
    e=1000
    
    for i in range(2,lim+1):        
        
        y_temp = y[i-2] + 2*f(t[i-1],y[i-1])*h  #predictor
        k=0  #contador
        
        while (e >= cvg and k < 15 ):
            
            old_pred = y_temp #guarda el predictor para usalo con el crit.cvg
            y_temp = y[i-1] + ( f(t[i-1],y[i-1]) + f(t[i],y_temp))*h/2 #corrector        
            
            #criterio de convergencia
            e = abs((y_temp - old_pred)/y_temp)*100
        
            k+=1
            
        y.append(y_temp)
        
    y.remove(y[0])
    
    return t,y  


##########################
'''
    METODO DE HEUN (cvg)
'''
##########################
def met_heun_it(a,b,yt0,f,cvg,h=0.5):    #limites del rango [a,b], cond inic: y(t=0),h:paso, f:funcion del problema
    
    t=array(arange(a,b+h,h))
    lim=len(t)
    y=[]                #lista para almacenar los valores y encontrados
    y.append(yt0)       #se ingresa el valor dato a la lista
    e=1000
    
    for i in range(1,lim):
        
        yn = y[i-1] + f(t[i-1],y[i-1])*h  #predictor
        k=0                 #contador
        
        while (e >= cvg and k < 15 ): 
            
            old_pred = yn #guarda el predictor para usalo con el crit.cvg      
            yn = y[i-1] + ( f(t[i-1],y[i-1]) + f(t[i],yn))*h/2 #corrector        
            
            
            #criterio de convergencia
            e = abs((yn - old_pred)/yn)*100
                
            k+=1    
    
        y.append(yn)
    
    return t,y  
 


'''
############################################################
############################################################
####################      PC 03        ####################
############################################################
############################################################ 
'''
####################    FUNCIONES DEL   ####################
############################################################
####################   EJERCICIO 1      ####################
############################################################
####################                    ####################

##########################
'''
    RK4 PARA SISTEMAS
'''
##########################
def sists_RK4(a,b,rt0,F,h=0.5):    #limites del rango [a,b], cond inic: y(t=0),h:paso, f:funcion edo del problema
    
    t=array(arange(a,b+h,h))
    lim=len(t)
    r=[]                #lista para almacenar los valores y encontrados
    r.append(rt0)       #se ingresa el valor dato a la lista
    
    for i in range(lim-1):
        
        k1 = F( t[i], r[i] )
        k2 = F( t[i] + h/2, r[i] + (h*k1)/2 )
        k3 = F( t[i] + h/2, r[i] + (h*k2)/2 )
        k4 = F( t[i] + h  , r[i] +  h*k3    )
        
        r_temp = r[i] + ( k1 + 2*k2 + 2*k3 + k4 )*h/6        
        r.append(r_temp)
    
    #separamos los datos 
    num_vects_generados=len(r) #nro de datos (vectores) generados con el método
    num_datos_en_vector=len(r[1]) #nro de elementos que contiene cualquier vector en r
    
    rdlist=[] #lista que almacenara los valores de cada variable en un solo elemento-vector
             #por ejemplo: si r= [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)], entonces rdist será
             #rdist = [(x1,x2,x3),(y1,y2,y3),(z1,z2,z3)]
    
    for i in range (num_datos_en_vector):
        elem_temp=[]
        for j in range (num_vects_generados):
            temp1 = r[j][i]
            elem_temp.append(temp1)
        rdlist.append(elem_temp)
    
    rdist=array(rdlist) #cambiando a array por seguridad
    
    return t,rdist  


####################    FUNCIONES DEL   #################### + RK4
############################################################
####################   EJERCICIO 2      ####################
############################################################
####################                    ####################
#eqs energias:
def Ek(r,m,g,l): #Energia cinetica , el vector r ingresado es de la forma r= [th1,th2,w1,w2}]
    th1 = r[0]
    th2 = r[1]
    w1 = r[2]
    w2 = r[3]
    
    Ek= 0.5*m*(l**2)*(2*(w1**2)+(w2**2)+2*w1*w2*cos(th1-th2))    
    return Ek

def Epot(r,m,g,l):
    th1 = r[0]
    th2 = r[1]
    w1 = r[2]
    w2 = r[3]
    
    Epot=-m*g*l*(2*cos(th1) + cos(th2))
    return Epot

def Em(Ek,Epot):
    return Ek+Epot


####################    FUNCIONES DEL   ####################
############################################################
####################   EJERCICIO 3      ####################
############################################################
####################                    ####################

####################
'''
    MÉTODO DEL TRAPECIO COMPUESTO
'''
#################### 
def trap_comp(a,b,func,n):
    
    h=(b-a)/n
    
    xi=[a] #lista para los x_{i}
    for i in range(1,n):
        xi.append(a + i*h)
    xi.append(b)
       
    #integral
    I=0
    for i in range(1,n):
        I=I+2*func(xi[i])
    I = 0.5*h*(func(xi[0]) + I + func(xi[n]))
    
    return I

####################
'''
    MÉTODO DEL SIMPSON 3/8
'''
#################### 

def simp38_comp(a,b,func,n): #n debe ser multiplo de 3, esto será evaluado dentro de la func
    
    resto = n%3 #si el resto es diferente de 0 -> entra al if y lo obliga a ser multiplo de 3
    if resto != 0:
        n = n - resto
        print("El # de intervalos no es múltiplo de 3")
        print("Se redefinio el valor de n ---> n=",str(n))
        
    h=(b-a)/n
    
    xi=[a] #lista para los x_{i}
    for i in range(1,n):
        xi.append(a + i*h)
    xi.append(b)
    
    #integral
    I=0
    for i in range(1,n,3):
        I=I + 3*func(xi[i])
                 
    for i in range(2,n,3):
        I=I + 3*func(xi[i])
        
    for i in range(3, n-1,3):
        I=I + 2*func(xi[i])
        
    I = 3*h*(func(xi[0]) + I + func(xi[n]))/8   
    
    return I,n







##FIN FUNCIONES PC03

'''
############################################################
############################################################
####################      TAREA 7       ####################
############################################################
############################################################ 
'''

####################                    ####################
############################################################
####################   EJERCICIO 7.1    ####################
############################################################
####################                    ####################

#Funcion que calcula el error porcentual entre dos valores ingresados
def porc_err(valreal,valx):
    return 100*abs(valreal-valx)/valreal

####################
'''
    MÉTODO DEL TRAPECIO
'''
####################

def trapecio(a,b,func): # del rango[a,b] para la funcion func(x), seg_derv= func''(x)
    
    #integral
    I = (b-a)*(func(a)+func(b))/2
    
    return I

#OBS: comentado, usado arriba con algunas variaciones
# ####################
# '''
#     MÉTODO DEL TRAPECIO COMPUESTO
# '''
# #################### 
# def trap_comp(a,b,n,func):
    
#     h=(b-a)/n
    
#     xi=[a] #lista para los x_{i}
#     for i in range(1,n):
#         xi.append(a + i*h)
#     xi.append(b)
      
#     #integral
#     I=0
#     for i in range(1,n):
#         I=I+2*func(xi[i])
#     I = 0.5*h*(func(xi[0]) + I + func(xi[n]))
    
#     return I

####################

#Funcion que grafica la evolucion del error para el método del trapecio compuesto
def evolucionreal_trapcomp(a,b,func,v_real): #error dependiente de la iteracion
    Errores=[]
    ns=[]
    hs=[]
    
    for n in range(1,36):
        
        h=(b-a)/n
    
        xi=[a] #lista para los x_{i}
        for i in range(1,n):
            xi.append(a + i*h)
        xi.append(b)
    
        #error
        I=0
        for i in range(1,n):
            I=I+2*func(xi[i])
        I = 0.5*h*(func(xi[0]) + I + func(xi[n]))
        
        Err=0 #limpia la variable de la iteracion anterior
        Err=porc_err(v_real,I)
        
        Errores.append(Err)
        ns.append(n)
        hs.append(h)
     
    plt.plot(ns,Errores)
    plt.plot(ns,Errores,'b*')
    plt.xlabel('# divisiones (n)')
    plt.ylabel('Error')
    plt.title('Evolución del error para Int.f(x) según {n} elegido')  
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.show()
    
    plt.plot(hs,Errores)
    plt.plot(hs,Errores,'b*')
    plt.xlabel('valor de h')
    plt.ylabel('Error')
    plt.title('Evolución del error para Int.f(x) según {h}') 
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.show()
    
    print("[Obs] %Error(n=35)=",Errores[34])
    print("       Valor(n=35)=",I)
    return ("###")


####################
'''
    MÉTODO DEL SIMPSON 1/3
'''
#################### 
def simp13(a,b,func):
    
    h=(b-a)/2
    
    xi=[a]
    xi.append(a + h)
    xi.append(b)
    
    #integral
    I=(b-a)*(func(xi[0]) + 4*func(xi[1]) + func(xi[2]))/6
    
    
    return I


####################
'''
    MÉTODO DEL SIMPSON 3/8
'''
#################### 
def simp38(a,b,func):
    
    h=(b-a)/3
    
    xi=[a] #lista para los x_{i}
    for i in range(1,4):
        xi.append(a + i*h)
    xi.append(b)
    
    #integral
    I=(b-a)*(func(xi[0]) + 3*(func(xi[1])+func(xi[2])) + func(xi[3]))/8
    
    return I

####################                    ####################
############################################################
####################   EJERCICIO 7.2    ####################
############################################################
####################                    ####################

####################
'''
    MÉTODO DEL TRAPECIO COMPUESTO PARA PTS NO EQUIDISTANTES
'''
#################### 
def trap_comp_ptsnoeq(puntos,fpuntos): #lista de puntos, valor de la funcion en los puntos
       
    #integral
    I=0
    n=len(puntos)
    for i in range(0,n-1):
        h = puntos[i+1] - puntos[i]
        I = I + h*(fpuntos[i]+fpuntos[i+1])/2
        
    return I


####################                    ####################
############################################################
####################   EJERCICIO 7.3    ####################
############################################################
####################                    ####################
####################
'''
    EXTRAPOLACION DE RICHARDSON
'''
#################### 

def extrapol_rich(I1,n1,I2,n2,a,b): #Se ingresan las integrales con su respectivo n 
    
    h1=(b-a)/n1
    h2=(b-a)/n2
    
    I= I2 + (I2 - I1)/( (h1/h2)**2 - 1)
    
    return I

####################                    ####################
############################################################
####################   EJERCICIO 7.4    ####################
############################################################
####################                    ####################
####################
'''
    CUADRATURA DE GAUSS
'''
#################### 

#GAUSS LEGENDRE
#funcion cuadratura GL con solo puntos para n=2,3,4,5 con sus respectivos pesos
def cuad_GL(func,a,b,n): #funcion, rango [a,b], # de pts
    
    #puntos
    # GLn=[[0](n=2), [1](n=3), [2](n=4), [3](n=5)] , puntos para diferente n
    GLn=array([(-3**-0.5,3**-0.5),
            (-(3/5)**0.5, 0, (3/5)**0.5),
            (-((3/7) + (2/7)*(6/5)**0.5)**0.5,-((3/7) - (2/7)*(6/5)**0.5)**0.5,((3/7) - (2/7)*(6/5)**0.5)**0.5,((3/7) + (2/7)*(6/5)**0.5)**0.5  ),
            (-1/3*(5 + 2*(10/7)**0.5)**0.5,-1/3*(5 - 2*(10/7)**0.5)**0.5,0,1/3*(5 - 2*(10/7)**0.5)**0.5,1/3*(5 + 2*(10/7)**0.5)**0.5  )]) 
    #pesos
    GLw=array([(1,1),
          (5/9,8/9,5/9),
          ( (18-30**0.5)/36 , (18+30**0.5)/36 , (18+30**0.5)/36 , (18-30**0.5)/36 ),
          ( (322 - 13*70**0.5)/900 , (322 + 13*70**0.5)/900 , 128/225 , (322 + 13*70**0.5)/900 , (322 - 13*70**0.5)/900 )])

    
    flag = n-2
    
    I=0
    for i in range(n):
        I = I + GLw[flag][i]*func( 0.5*(b-a)*GLn[flag][i] + (a+b)*0.5 )
    I= I*(b-a)/2
    
    return I

#GAUSS-RADAU-LEGENDRE
def cuad_GRL(func,a,b,n):
    
    #raices
    # GRLn=[[0](n=3), [1](n=4), [2](n=5)]
    GRLn=array([(-1.000000,-0.289898,0.689898),
               (-1.000000,-0.575319,0.181066,0.822824),
               (-1.000000,-0.720480,-0.167181,0.446314,0.885792)])
    #pesos
    GRLw=array([ (0.222222, 1.0249717, 0.7528061),
               (0.125000,0.657689,0.776387,0.440924),
               (0.080000,0.446208,0.623653,0.562712, 0.287427)])
    
    flag = n-3
    
    I=0
    for i in range(n):
        I = I + GRLw[flag][i]*func( 0.5*(b-a)*GRLn[flag][i] + (a+b)*0.5 )
    I= I*(b-a)/2
    
    return I

#GAUSS-LOBATTO-LEGENDRE
def cuad_GLL(func,a,b,n):
    
    #raices
    # GRLn=[[0](n=2), [1](n=3), [2](n=4), [3](n=5)]
    GLLn=array([(-1,1),
               (-1,0,1),
               (-1,-0.447213595499958,0.447213595499958,1),
               (-1,-0.654653670707977,0,0.654653670707977,1)])
    
    #pesos
    GLLw=array([(1,1),
               (0.333333333333333,1.333333333333333,0.333333333333333),
               (0.166666666666667,0.833333333333333,0.833333333333333,0.166666666666667),
               (0.1,0.544444444444444,0.711111111111111,0.544444444444444,0.1)])
    
    flag = n-2
    
    I=0
    for i in range(n):
        I = I + GLLw[flag][i]*func(  0.5*(b-a)*GLLn[flag][i] + (a+b)*0.5 )
    I= I*(b-a)/2
    
    return I


####################                    ####################
############################################################
####################   EJERCICIO 7.5    ####################
############################################################
####################                    ####################
def accs(dr_Ox,dth_Ox,r,t):
    dim=dr_Ox.shape #ambos vects tienen la misma dimension, (x,2)
    lim=dim[0]
    for i in range(lim):
        acc=[(dr_Ox[i][1]-r[i]*dth_Ox[i][0]**2),(r[i]*dth_Ox[i][1]+2*dr_Ox[i][0]*dth_Ox[i][0])]
        print("Aceleración(t=",t[i],")=",acc[0],"er + ",acc[1],"e",u"\u03B8")
    return("")
    

def vels(dr_Ox,dth_Ox,r,t):
    dim=dr_Ox.shape #ambos vects tienen la misma dimension, (x,2)
    lim=dim[0]
    for i in range(lim):
        vel=[dr_Ox[i][0],r[i]*dth_Ox[i][0]]
        print("Velocidad(t=",t[i],")=",vel[0],"er + ",vel[1],"e",u"\u03B8")
    return("")

####################
'''
    DIF FORWARD
'''
#################### 
def dif_forw(x,Ox,t): #funcion, puntos, O(x), orden de deriv

    h=t[1]-t[0] 
    dervs=[]
    lim=len(x)
    
    if Ox == 1:
        
        for i in range(lim-3):            
            temp=[]
            
            #1ra derivada
            d1f = ( x[i+1]-x[i] )/h
            temp.append(d1f)

            #2da derv
            d2f= ( x[i+2]-2*x[i+1] +x[i])/(h**2)
            temp.append(d2f)

            dervs.append(temp)
     
        dervs_fOx=copy(dervs)
    
    if Ox == 2:
        
        for i in range(lim-3):            
            temp=[]
            h=(t[i+1]-t[i])
            
            #1ra derivada
            d1f = (-x[i+2] +4*x[i+1] -3*x[i])/( 2*h )
            temp.append(d1f)

            #2da derv
            d2f= ( -x[i+3] +4*x[i+2] -5*x[i+1] +2*x[i])/ (h**2)
            temp.append(d2f)

            dervs.append(temp)

        dervs_fOx=copy(dervs)
    
    return dervs_fOx #lista

#funcion para elegir los puntos que pueden ser usados en la diferenciacion forward
def eligeforw(vector):
    lim=len(vector)
    puntoselegidos=[]
    for i in range(lim-3):
        puntoselegidos.append(vector[i])
    return(puntoselegidos)


####################
'''
    DIF CENTRADA
'''
#################### 
def dif_cent(x,Ox,t): #funcion, puntos, O(x), orden de deriv
    
    h=t[1]-t[0] #el paso se mantiente para cualquier i 
    dervs=[]
    lim=len(x) 
    
    if Ox == 1:
                
        for i in range(1,lim-2):
            temp=[]
            
            #1ra derivada
            d1f = ( x[i+1]-x[i-1] )/(2*h)
            temp.append(d1f)

            #2da derv
            d2f= ( x[i+1] -2*x[i] +x[i-1])/(h**2)
            temp.append(d2f)

            dervs.append(temp)

            
        dervs_fOx=copy(dervs)
    
    if Ox == 2:
        
        for i in range(2,lim-2):            
            temp=[]
            
            #1ra derivada
            d1f = (-x[i+2] +8*x[i+1] -8*x[i-1] +x[i-2])/( 12*h )
            temp.append(d1f)
            
            #2da derv
            d2f= ( -x[i+2] +16*x[i+1] -30*x[i] +16*x[i-1] -x[i-2] )/ (12*(h**2))
            temp.append(d2f)
                        
            dervs.append(temp)
    
        dervs_fOx=copy(dervs)
    
    return dervs_fOx  # devuelve una lista (de valores de las derivadas)

#funcion para elegir los puntos que pueden ser usados en la diferenciacion centrada
def eligecent(vector):
    lim=len(vector)
    puntoselegidos=[]
    for i in range(1,lim-2):
        puntoselegidos.append(vector[i])
    return(puntoselegidos)



####################
'''
    DIF BACKWARDS
'''
#################### 
def dif_back(x,Ox,t): #funcion, puntos, O(x), orden de deriv
    
    h=t[1]-t[0]
    dervs=[]
    lim=len(x) #para los indices de x[i]
    
    if Ox == 1:  
        
        for i in range(3,lim): #3 pues necesita 3 puntos anteriores {0,1,2}            
            temp=[]
            
            #1ra derivada
            d1f = ( x[i]-x[i-1] )/h
            temp.append(d1f)
    
            #2da derv
            d2f= (x[i] -2*x[i-1] +x[i-2])/( h**2 )
            temp.append(d2f)
        
            dervs.append(temp)
     
        dervs_fOx=copy(dervs)
        
    if Ox == 2:
        
        for i in range(3,lim):            
            temp=[]
            
            #1ra derivada
            d1f = ( 3*x[i] -4*x[i-1] +x[i-2] )/(2*h)
            temp.append(d1f)

            #2da derv
            d2f= ( 2*x[i] -5*x[i-1] +4*x[i-2] -x[i-3] )/ (h**2)
            temp.append(d2f)
        
            dervs.append(temp)
                
        dervs_fOx=copy(dervs)
    
    return dervs_fOx # devuelve una lista (de valores de las derivadas)

#funcion para elegir los puntos que pueden ser usados en la diferenciacion backward
def eligeback(vector):
    lim=len(vector)
    puntoselegidos=[]
    for i in range(3,lim):
        puntoselegidos.append(vector[i])
    return(puntoselegidos)
