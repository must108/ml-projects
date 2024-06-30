# training a neural network heavily involves solving an optimization problem.

# a higher learning rate leads to poor model performance (model can find desired values efficiently)
# a lower learning rate leads to long training times.


# null momentum can lead to an optimizer being stuck in 
# a local minumum

# a non-null momentum can help find the min of the function
# typical learning rate values range between 10^-2 and 10^-4
# typical momentum values range between 0.85 and 0.99
# best practice is to start with an LR of 0.001 and
# a momentum of 0.95