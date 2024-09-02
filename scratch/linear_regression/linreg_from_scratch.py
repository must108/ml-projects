import pandas as pd
import numpy as np

teams = pd.read_csv("./teams.csv")

X = teams[["athletes", "prev_medals"]].copy()
Y = teams[["medals"]].copy()

# matrix multiplication:

# rows of first matrix and cols of second matrix is dimensions of the
# resulting matrix

# to transpose a matrix is to essentially: make all rows into columns
# mutliplying a matrix by its transpose leaves you with a square matrix
# you can invert a square matrix!

# an inverted matrix has the same size as the square matrix
# if you multiply a matrix by its inverse, you end up with the
# "identity matrix"

# if you multiply an identity matrix by its original matrix,
# you get the same original matrix as a result
# like multiplying by 1

# to solve for linear regression, put vals in a matrix (b0, b1, e)
# do this to solve for our coefficients (b0, b1, or B)
# each value can be represented by a matrix
# matrix of y vals = (matrix of x) * (matrix of B) + (matrix of e)

# this can be simplified to y = XB + e
# e can be dropped, as we're trying to minimize error

# y = XB
# we want to cancel out X by multiplying by its inverse
# but since X isnt a square matrix, we multiply by its transpose

# (XT = transpose)
# XTy = XTX B

# now, XTX = square matrix.

# (XTX)-1 XTy = (XTX)-1 XTX B
# multiply by the inverse of XTX, as XTX is square

# identity matrix is 1, so XTX is cancelled out

# (XTX)-1 XTy = IB (where I is the identity matrix)
# (XTX)-1 XTy = B (as the identity matrix can be treated as 1)

# B = (XTX)-1 XTy
# our final formula, solves for lin reg coefficients
