import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize

#optimizador con la implementacion de Nocedal
def bfgs_method(f, fprime, x0, maxiter=None, epsi=10e-3):
    """
    Función para minimizar la función f usando el BFGS
    Parametros
    ----------
    func : f(x)- Función que se quiere minimizar
    fprime : fprime(x) - gradiente de la función
    x0 : ndarray - Punto inicial de la búsqueda
    maxiter: int - Número máximo de iteraciones
    epsi: double - Criterio de convergencia de la norma del gradiente
    """
    if maxiter is None:
        #Se asigna un límite al numero de iteraciones propocional al numero de variables
        maxiter = len(x0) * 200
    # valores iniciales para la iteración 0
    #gfk es el gradiente de la función evaluado en x0, es un array de dos dimensiones, una por cada variable
    k = 0
    gfk = fprime(x0)
    N = len(x0)
    # Se define la matriz identidad que es la estimación inicial de la inversa de la Hessiana = Hk
    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
    #si la norma del gradiente en Xk es superior a epsilon, continua iterando y si no se ha superado el numero maximo de iteraciones
    while ln.norm(gfk) > epsi and k < maxiter:
        # pk - se calcula dirección de la búsqueda
        pk = -np.dot(Hk, gfk)
        # Se calcula la constante ak para la búsqueda lineal y que cumpla las condiciones de Wolfe
        #se usa el método line_search de la clase optimize de la libreria scipy con el alias sp
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html
        line_search = sp.optimize.line_search(f, f1, xk, pk)
        alpha_k = line_search[0]
        # actualización de Xk a Xk+1
        xkp1 = xk + alpha_k * pk
        #se calcula sk necesario para la actualización de la Hessiana estimada
        sk = xkp1 - xk
        # se actualiza xk para la proxima iteración
        xk = xkp1
        #se calcula el gradiente en el nuevo punto xk
        gfkp1 = fprime(xkp1)
        # se calcula yk necesario para la actualización de la Hessiana estimada
        yk = gfkp1 - gfk
        # se actualiza gfk para la proxima iteración
        gfk = gfkp1
        #se actualiza la iteración
        k += 1
        # se actualiza el parametro r0
        ro = 1.0 / (np.dot(yk, sk))
        #actualización de la Hessiana inversa propuesta por Nocedal
        A1 = I - ro * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - ro * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (ro * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])
    return (xk, k, ln.norm(gfk))