# -*- coding: utf-8 -*-
import numpy as np
from itertools import islice
import pandas as pd
############################################################
#        Functions for reading and writing grid files
############################################################

def read_grid_file(path_file):
    
    '''
    Description
    -----------

    This function reads the paramater files that are in grid form
   
    Parameters
    ----------
    path_file : file path of the the file to read

    Return
    ------
    x (list of lists) : Each element is a list with x coordinates of a layer 
    y (list of lists) : Each element is a list with y coordinates of a layer 
    grid_layer(list)  : Each element is a numpy.ndarray with parameter values 
    delc : widths of the columns
    delr : heights of the lines 

    Example
    -----------
    x,y,grid,delc,delr = read_grid_file(file_path)
    
    '''
    grid_layer = []
    x = []
    y = []   

    #open the file
    data = open(path_file,"r")

    #begin of each grid
    lookup_begin  = '[Data]' 
    lookup_begin2 = '[Constant_Data]' 

    #end of each grid
    lookup_end = '[End_Grid]'

    for num, line in enumerate(data, 1):
        #search for line number with begin mark
        if lookup_begin in line:
            begin = num + 1
            grid  = True 
        if lookup_begin2 in line:
            begin = num 
            grid  = False
        #search for line number with end mark
        if lookup_end in line:
            if num  > begin: 
                end = num -1

                if grid == True :
                    table_split = []
                    param_grid  = []
                    # select table
                    with open(path_file,"r") as text_file:
                        for line in islice(text_file, begin,  end ):
                            table_split.append(line.split())
                    for l in table_split :
                        param_grid.append([float(v) for v in l])
                    # select yrows, xcols, delr, delc, param in param_grid
                    x_val = param_grid[0]
                    del x_val[0:2]
                    y_val = list(zip(*param_grid))[1]
                    y_val = y_val[1:-1]
                    delr  = param_grid[-1][2:]
                    delc  = (param_grid)[2:]
                    delc  = [c[-1] for c in  zip(*delc)]
                    param_liste  = [c[0:-1] for c in  zip(*param_grid[1:])]
                    del param_liste[0:2]
                    param_tab = pd.DataFrame(param_liste).values
                    
                if grid == False :
                    table_split = []
                    param_grid  = []
                    # select table
                    with open(path_file,"r") as text_file:
                        for line in islice(text_file, begin,  end ):
                            table_split.append(line.split())
                    constant_value = (float(table_split[0][0].split("=")[1]))
                    param_tab = np.full((148,128), constant_value)
                    
                    # select yrows, xcols, delr, delc, param in param_grid
                    x_val = table_split[3]
                    x_val = list(np.array(x_val).astype(np.float))
                    y_val = table_split[7]
                    y_val = list(np.array(y_val).astype(np.float))

                grid_layer.append(param_tab)
                x.append(x_val)
                y.append(y_val)

    return (x,y,grid_layer,delc,delr)
    

def write_grid_file(path_file,grid_layer,x,y,delc,delr):
    
    '''
    Description
    -----------

    This function writes text file with the same structure than parameter file in grid form
   
    Parameters
    ----------
    path_file : file path of the wrritting file
    grid : data of a layer structured in rectangular grid
    x (list of lists) : Each element is a list with x coordinates of a layer 
    y (list of lists) : Each element is a list with y coordinates of a layer 
    grid_layer(list)  : Each element is a numpy.ndarray with parameter values  
    delc : widths of the columns
    delr : heights of the lines

    Example
    -----------
    write_grid_file(path_file,grid_layer,x,y,delc,delr)
        
    '''
    grid_pp = open(path_file , "a")
    ncol = np.arange(0,149,1)
    nrow = np.arange(1,129,1)
    x[0].insert(0,0)
    x[0].insert(0,0)
    #del delc[0:1]
    delc = list(np.int_(delc))
    delr = list(np.int_(delr)) 
    i = 0
    
    for grid in grid_layer:
        i = i + 1
        grid = grid.transpose()
        
        perm = zip(*grid)
        grid_pp.write('Marthe_Grid Version=9.0 \n')
        grid_pp.write('Title=Travail                                                        '+path_file+'            '+str(i)+'\n')
        grid_pp.write('[Infos]\n')
        grid_pp.write('Field=\n')
        grid_pp.write('Type=\n')
        grid_pp.write('Elem_Number=0\n')
        grid_pp.write('Name=\n')
        grid_pp.write('Time_Step=-9999\n')
        grid_pp.write('Time=0\n')
        grid_pp.write('Layer=0\n')
        grid_pp.write('Max_Layer=0\n')
        grid_pp.write('Nest_grid=0\n')
        grid_pp.write('Max_NestG=0\n')
        grid_pp.write('[Structure]\n')
        grid_pp.write('X_Left_Corner=284.5\n')
        grid_pp.write('Y_Lower_Corner=156\n')
        grid_pp.write('Ncolumn=148\n')
        grid_pp.write('Nrows=128\n')
        grid_pp.write('[Data]\n')
        grid_pp.write('0 \t')
        [grid_pp.write(str(i)+'\t') for i in ncol]
        grid_pp.write('\n')
        [grid_pp.write(str(i)+'\t') for i in x[0]]
        grid_pp.write('\n')
        for row, cols, perm_line, col_size in zip(nrow, y[0],grid, delr) :
            grid_pp.write(str(row)+'\t'+str(cols)+'\t')
            [grid_pp.write(str(i)+'\t') for i in perm_line]
            grid_pp.write(str(col_size) +'\t \n')
        [grid_pp.write(str(j)+'\t') for j in delc]
        grid_pp.write('\n')
        grid_pp.write('[End_Grid]\n')

    return ()