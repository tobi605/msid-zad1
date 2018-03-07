# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    mean_error = 0
    values = polynomial(x,w)
    for i in range(values.shape[0]):
        mean_error += (np.square(y[i]-values[i]))
    mean_error = mean_error/values.shape[0]
    return float(mean_error)


def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    matrix = np.zeros(shape=(M+1, x_train.shape[0]))
    for j in range(M+1):
        for i in range(x_train.shape[0]):
            matrix[j][i]=x_train[i]**j
    return(matrix.T)
    #pass


def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    phi = design_matrix(x_train, M)
    w = (np.linalg.inv(phi.T.dot(phi))).dot(phi.T).dot(y_train)
    err = mean_squared_error(x_train, y_train, w)
    return (w,err)
    #pass


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    
    phi = design_matrix(x_train, M)
    w = (np.linalg.inv(phi.T.dot(phi)+(regularization_lambda*np.eye(phi.shape[1])))).dot(phi.T).dot(y_train)
    err = mean_squared_error(x_train, y_train, w)
    return (w,err)
    #pass


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    params = []
    for M in M_values:
        params.append(least_squares(x_train,y_train,M)[0])
    val_error = mean_squared_error(x_val, y_val, params[0])
    best_param = params[0]
    for param in params:
        if(mean_squared_error(x_val,y_val,param)<val_error):
            val_error = mean_squared_error(x_val,y_val,param)
            best_param = param
    return(best_param, mean_squared_error(x_train, y_train, best_param), val_error)
    pass


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''
    params = []
    for lamb in lambda_values:
        params.append(regularized_least_squares(x_train, y_train, M, lamb)[0])
    val_error = mean_squared_error(x_val, y_val, params[0])
    best_lambda = lambda_values[0]
    best_param = params[0]
    for i in range(len(lambda_values)):
        if(mean_squared_error(x_val, y_val, params[i])<val_error):
            val_error = mean_squared_error(x_val, y_val, params[i])
            best_param = params[i]
            best_lambda = lambda_values[i]
    return(best_param, mean_squared_error(x_train, y_train, best_param), val_error, best_lambda)
    pass
