
import sys
import numpy as np
import scipy as sp
import scipy.sparse.linalg as lg
from scipy.sparse import lil_matrix
import random 
import statistics as stat
import matplotlib.pyplot as plt
import statistics
from scipy import stats

gridsize=4  # arena size
matrixdim=(gridsize+1)*(gridsize+1)*6+1
Arena=lil_matrix((matrixdim,matrixdim))

# build probability transition matrix

PIF=0.49 # inside go front
PIR=0.01 # inside go backwards
PIST=0.24 # inside small turn
PIBT=0.01 #inside large turn


PEPF=0.45 # move parallel with edge front
PEPR=0.04 # move parallel with edge go backwards
PEPST=0.45 #probability edge parallel small turn (PEPST)
PEPBT=0.06 # probability edge big turn (PEPBT)


# move toward the edge PEC
PECR= 0.6   #probability edge collide reflect
PECB=0.15     #probability edge collide back
PECST=0.15     #probability edge collide small turn
PECBT=0.1      #probability edge collide big turn

# sharp corner
PSCR=0.5    #probability sharp corner rear
PSCBT=0.5     #probability sharp corner big turn

# obtuse corner parallel
POCPB=.25  #probabiliy obtuse corner parallel back
POCPNE=.25 #probabiliy obtuse corner parallel next edge
POCPM=.5 #probabiliy obtuse corner parallel middle

# obtuse corner middle
POCMB=0.34 #probabiliy obtuse corner middle back
POCMS=0.33 #probabiliy obtuse corner middle side

#left corner sharp angle
Arena[2,6]=PSCBT #120A->0B
Arena[3,6]=PSCR
colnum=6*(gridsize+1)+5
Arena[2,colnum]=PSCR
Arena[3,colnum]=PSCBT

# fill the hypothetical nodes
Arena[0,2]=1
Arena[1,2]=1
Arena[4,2]=1
Arena[5,2]=1

numpoints=gridsize-1; #number of points on the upper boundary
for i in range(numpoints):     #each iteration is a boundary point (top boundary)
    rowindex=6+i*6 #have not included the index due to direction
    colindex=3+6*i #starting col index for each grid e.g. A180
    Arena[rowindex,colindex]=PEPR
    Arena[rowindex+1, colindex] = PECBT
    Arena[rowindex+2, colindex] = PECST
    Arena[rowindex+3, colindex] = PEPF

    Arena[rowindex,colindex+9]=PEPF
    Arena[rowindex+1, colindex+9] = PECST
    Arena[rowindex+2, colindex+9] = PECBT
    Arena[rowindex+3, colindex+9] = PEPR

    colindexlong=(gridsize+1)*6+4+i*6
    Arena[rowindex,colindexlong]=PEPBT
    Arena[rowindex+1, colindexlong] =PECB
    Arena[rowindex+2, colindexlong] = PECR
    Arena[rowindex+3, colindexlong] = PEPST

    Arena[rowindex,colindexlong+7]=PEPST
    Arena[rowindex+1, colindexlong+7] = PECR
    Arena[rowindex+2, colindexlong+7] = PECB
    Arena[rowindex+3, colindexlong+7] = PEPBT

# fill in the hypothetial nodes
    Arena[rowindex+4,rowindex]=1
    Arena[rowindex + 5,rowindex] = 1

# obtuse corner parallel
#POCPB=.25  #probabiliy obtuse corner parallel back
#POCPNE=.25 #probabiliy obtuse corner parallel next edge
#POCPM=.5 #probabiliy obtuse corner parallel middle

# obtuse corner middle
#POCMB=1/3 #probabiliy obtuse corner middle back
#POCMS=1/3 #probabiliy obtuse corner middle side

#the obtuse corner
startrowindex=6*gridsize
colnum1=6*(gridsize-1)+3
Arena[startrowindex,   colnum1   ]=POCPB
Arena[startrowindex+1,  colnum1 ]=POCMS
Arena[startrowindex+2,  colnum1  ]=POCPNE

colnum2=((gridsize+1)+(gridsize-1))*6+4
Arena[startrowindex,   colnum2   ]=POCPM
Arena[startrowindex+1,  colnum2 ]=POCMB
Arena[startrowindex+2,  colnum2  ]=POCPM

colnum3=((gridsize+1)+gridsize)*6+5
Arena[startrowindex,   colnum3   ]=POCPNE
Arena[startrowindex+1,  colnum3 ]=POCMS
Arena[startrowindex+2,  colnum3  ]=POCPB

# fill in the hypothetical nodes
Arena[startrowindex+3,startrowindex]=1
Arena[startrowindex+4,startrowindex]=1
Arena[startrowindex+5,startrowindex]=1

# start the construct each row of arean in the middle
for i in range (gridsize-1):
    startrowindex=(gridsize+1)*6+i*(gridsize+1)*6  # row index of the first entry
    col1=startrowindex-(gridsize+1)*6+2
    col2=startrowindex-(gridsize+1)*6+6+1
    col3=startrowindex+6
    col4=startrowindex+(gridsize+1)*6+5
    Arena[startrowindex+2, col1 ]=PEPF
    Arena[startrowindex + 3,col1]=PECST
    Arena[startrowindex + 4,col1]=PECBT
    Arena[startrowindex + 5,col1]=PEPR

    Arena[startrowindex+2, col2 ]=PEPST
    Arena[startrowindex + 3,col2]=PECR
    Arena[startrowindex + 4,col2]=PECB
    Arena[startrowindex + 5,col2]=PEPBT

    Arena[startrowindex+2, col3 ]=PEPBT
    Arena[startrowindex + 3,col3]=PECB
    Arena[startrowindex + 4,col3]=PECR
    Arena[startrowindex + 5,col3]=PEPST

    Arena[startrowindex+2, col4 ]=PEPR
    Arena[startrowindex + 3,col4]=PECBT
    Arena[startrowindex + 4,col4]=PECST
    Arena[startrowindex + 5,col4]=PEPF
    #fill in the hypothetical nodes
    Arena[startrowindex,startrowindex+2 ]=1
    Arena[startrowindex + 1,startrowindex+2]=1

    #now work on the interior nodes
    for j in range (gridsize-1):
        startrowindex=startrowindex+6
        col1 = startrowindex - (gridsize + 1) * 6+2
        col2=startrowindex - (gridsize + 1) * 6+6+1
        col3=startrowindex-6+3
        col4=startrowindex+6
        col5=startrowindex + gridsize * 6+4
        col6=startrowindex + gridsize * 6+6+5

  
        Arena[startrowindex,col1]=PIBT
        Arena[startrowindex+1,col1]=PIST
        Arena[startrowindex+2,col1]=PIF
        Arena[startrowindex+3,col1]=PIST
        Arena[startrowindex+4,col1]=PIBT
        Arena[startrowindex+5,col1]=PIR


        Arena[startrowindex,col2]=PIST
        Arena[startrowindex+1,col2]=PIF
        Arena[startrowindex+2,col2]=PIST
        Arena[startrowindex+3,col2]=PIBT
        Arena[startrowindex+4,col2]=PIR
        Arena[startrowindex+5,col2]=PIBT


        Arena[startrowindex,col3]=PIR
        Arena[startrowindex+1,col3]=PIBT
        Arena[startrowindex+2,col3]=PIST
        Arena[startrowindex+3,col3]=PIF
        Arena[startrowindex+4,col3]=PIST
        Arena[startrowindex+5,col3]=PIBT


        Arena[startrowindex,col4]=PIF
        Arena[startrowindex+1,col4]=PIST
        Arena[startrowindex+2,col4]=PIBT
        Arena[startrowindex+3,col4]=PIR
        Arena[startrowindex+4,col4]=PIBT
        Arena[startrowindex+5,col4]=PIST


        Arena[startrowindex,col5]=PIBT
        Arena[startrowindex+1,col5]=PIR
        Arena[startrowindex+2,col5]=PIBT
        Arena[startrowindex+3,col5]=PIST
        Arena[startrowindex+4,col5]=PIF
        Arena[startrowindex+5,col5]=PIST


        Arena[startrowindex,col6]=PIST
        Arena[startrowindex+1,col6]=PIBT
        Arena[startrowindex+2,col6]=PIR
        Arena[startrowindex+3,col6]=PIBT
        Arena[startrowindex+4,col6]=PIST
        Arena[startrowindex+5,col6]=PIF
    #the inner for loop ends here
    #construct the right edge
    startrowindex=startrowindex+6
    col1=startrowindex-(gridsize + 1) * 6+2
    col2=startrowindex-6+3
    col3=startrowindex+(gridsize) * 6+4
    col4=startrowindex+(gridsize+1) * 6+5


    Arena[startrowindex, col1] =PECBT
    Arena[startrowindex+1, col1] =PECST
    Arena[startrowindex+2, col1] =PEPF
    Arena[startrowindex+5, col1] =PEPR

    Arena[startrowindex, col2] =PECB
    Arena[startrowindex+1, col2] =PECR
    Arena[startrowindex+2,col2 ] =PEPST
    Arena[startrowindex+5,col2 ] =PEPBT

    Arena[startrowindex, col3] =PECR
    Arena[startrowindex+1, col3] =PECB
    Arena[startrowindex+2,col3 ] =PEPBT
    Arena[startrowindex+5,col3 ] =PEPST

    Arena[startrowindex, col4] =PECST
    Arena[startrowindex+1,col4 ] =PECBT
    Arena[startrowindex+2, col4] =PEPR
    Arena[startrowindex+5,col4 ] =PEPF
    #fill in the hypothetical edge
    Arena[startrowindex+3,startrowindex ] = 1
    Arena[startrowindex+4, startrowindex] =1


startrowindex=(gridsize+1)*6*gridsize
col1=startrowindex-(gridsize+1)*6+2
col2=startrowindex-(gridsize+1)*6+6+1
col3=startrowindex+6


Arena[startrowindex+3,col1]=POCPNE
Arena[startrowindex+4,col1]=POCMS
Arena[startrowindex+5,col1]=POCPB

Arena[startrowindex+3,col2]=POCPM
Arena[startrowindex+4,col2]=POCMB
Arena[startrowindex+5,col2]=POCPM

Arena[startrowindex+3,col3]=POCPB
Arena[startrowindex+4,col3]=POCMS
Arena[startrowindex+5,col3]=POCPNE

#fill in hypothetical edge
Arena[startrowindex,startrowindex+3]=1
Arena[startrowindex+1,startrowindex+3]=1
Arena[startrowindex+2,startrowindex+3]=1


startrowindex=(gridsize+1)*6*gridsize
for i in range (gridsize-1):
    startrowindex=startrowindex+6

    col1=startrowindex-(gridsize+1)*6+2
    col2=startrowindex-(gridsize+1)*6+6+1
    col3=startrowindex-6+3
    col4=startrowindex+6

    Arena[startrowindex, col1]= PEPBT
    Arena[startrowindex+3,col1] =PEPST
    Arena[startrowindex+4,col1] =PECR
    Arena[startrowindex+5,col1] =PECB

    Arena[startrowindex, col2]=PEPST
    Arena[startrowindex+3,col2] =PEPBT
    Arena[startrowindex+4,col2] =PECB
    Arena[startrowindex+5,col2] =PECR


    Arena[startrowindex, col3]=PEPR
    Arena[startrowindex+3,col3] =PEPF
    Arena[startrowindex+4,col3] =PECST
    Arena[startrowindex+5,col3] =PECBT


    Arena[startrowindex, col4]=PEPF
    Arena[startrowindex+3,col4] =PEPR
    Arena[startrowindex+4,col4] =PECBT
    Arena[startrowindex+5,col4] =PECST

    #fill in the hypothetical edge
    Arena[startrowindex + 1,startrowindex ] = 1
    Arena[startrowindex + 2,startrowindex ] = 1

startrowindex=6*((gridsize+1)*(gridsize)+gridsize)

col1=startrowindex-6*(gridsize+1)+2
col2=startrowindex-6+3

Arena[startrowindex, col1] = PSCBT
Arena[startrowindex + 5, col1] = PSCR

Arena[startrowindex, col2] = PSCR
Arena[startrowindex + 5, col2] = PSCBT

#fill in the hypothetical edge
Arena[startrowindex+1,startrowindex ]=1
Arena[startrowindex+2,startrowindex ]=1
Arena[startrowindex+3,startrowindex ]=1
Arena[startrowindex+4, startrowindex]=1

# adjsut the goal node
rowindex=(gridsize+1)*(gridsize+1)*6
Arena[rowindex, rowindex]=1 # this is the absorbing node
# to modify the matrix goal node at the center, the number of grids in each dimension much be even
rowindex=6*((gridsize+1)*(gridsize+1)//2)
colindex = (gridsize + 1) * (gridsize + 1) * 6
for i in range (6):
    for j in range(matrixdim):
        Arena[rowindex+i, j] =0
    Arena[rowindex + i, colindex] = 1

#now change the 1 hop nodes (i.e.sensing range is 1)
rowindex=6*((gridsize+1)*(gridsize+1)//2)-(gridsize+1)*6
midindex=6*((gridsize+1)*(gridsize+1)//2)
for i in range (6):
    for j in range(matrixdim):
        Arena[rowindex+i, j] =0
    Arena[rowindex+i,matrixdim-1] =1

rowindex=rowindex+6
for i in range (6):
    for j in range(matrixdim):
        Arena[rowindex+i, j] =0
    Arena[rowindex+i,matrixdim-1] =1

rowindex=midindex-6
for i in range (6):
    for j in range(matrixdim):
        Arena[rowindex+i, j] =0
    Arena[rowindex+i,matrixdim-1] =1

rowindex=midindex+6
for i in range (6):
    for j in range(matrixdim):
        Arena[rowindex+i, j] =0
    Arena[rowindex+i,matrixdim-1] =1

rowindex=midindex+6*gridsize
for i in range (6):
    for j in range(matrixdim):
        Arena[rowindex+i, j] =0
    Arena[rowindex+i,matrixdim-1] =1

rowindex=midindex+6*gridsize+6
for i in range (6):
    for j in range(matrixdim):
        Arena[rowindex+i, j] =0
    Arena[rowindex+i,matrixdim-1] =1


print(Arena[2])

#extract the matrix

# Q=Arena[0:matrixdim-1,0:matrixdim-1] # transition probability matrix for nonabsorbing states
# print(Q)
# I=sp.sparse.identity(matrixdim-1)
# I_Q=I-Q 
# RHS=np.ones([matrixdim-1,1])
# n=I_Q.shape[0]
# M_x = lambda x: lg.spsolve(I_Q,x)
# M = lg.LinearOperator((n, n), M_x)

# print('GMRES start')
# solution=lg.gmres(I_Q,RHS,restart=20,M=M)
# gmres_flag=solution[1]
# print(f"GMRES end: {gmres_flag}")

# solution=solution[0] #solving the equation for mean
# solutionsq=np.ones([matrixdim-1,1])
# solution2=np.ones([matrixdim-1,1])

# for i in range (matrixdim-1):
#   solutionsq[i,0]=solution[i]*solution[i]
#   solution2[i,0]=solution[i]*2

# solutionarr=np.asarray(solution)
# solutionnew=np.zeros((matrixdim-1,1))

# for i in range(matrixdim-1):
#   solutionnew[i,0]=solutionarr[i]

# RHS=solution2-I_Q.dot(solutionnew)-I_Q.dot(solutionsq)
# var=lg.gmres(I_Q,RHS,restart=20,M=M)
# #solving for the variance