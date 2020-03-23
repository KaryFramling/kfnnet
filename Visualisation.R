# Requires:
# - Functions.R

# Create a 3D plot where:
# - function.approximator: an Object of class FunctionApproximator. Has to have at least method "eval" that gives matrix of outputs.
# - show.indices: should always be c(1,2) for 3D plot, presumably.
# - show.mins: vector of minimum x,y values
# - show.maxs: vector of maximum x,y values
# - show.steps: vector of step values for x,y values
# - outindex: the column to use in output matrix as Z axis values
# - default.inputs: input value to use in input matrix for all other than x and y columns. 
#                   This parameter determines the total number of inputs if greater than 2. 
# 
plot3d.output <- function(function.approximator, show.indices, show.mins, show.maxs, show.steps, outindex=1, default.inputs, ...) {

  # Get Z values. We do this one by one in case the function approximator
  # is not capable of handling an entire data set of x/y values
  m <- create.input.matrix(show.indices, show.mins, show.maxs, show.steps, default.inputs)
  outs <- as.matrix(apply(m, 1, function.approximator$eval)) # Added "as.matrix" for dealing with vector, hope it doesn't break anything...
#  z <- outs[outindex,] # This is how it was initially, let's see what happens when changing...
  z <- outs[, outindex]

  if ( identical(min(z), max(z) ) )
    zlim <- c(0,1)
  else
    zlim <- range(z, na.rm = TRUE)

  # Now redimension for displaying
  xvals <- seq(show.mins[1], show.maxs[1], show.steps[1])
  yvals <- seq(show.mins[2], show.maxs[2], show.steps[2])
  zm <- matrix(data=z, nrow=length(xvals), ncol=length(yvals), byrow=TRUE)
  persp(xvals, yvals, zm, zlim=zlim, ...)
}

test.3Dapproximator.new <- function() {

  m <- list (
             get.inputs = function() { NULL },
             get.outputs = function() { NULL },
             eval = function(invals) {
               return(invals[1]*invals[2])
             }
             )
  class(m) <- c("FunctionApproximator")
  return(m)
}

test.3Dapproximator <- function() {
  fa <- test.3Dapproximator.new()
  plot3d.output(fa, c(1,2), c(-1,-1), c(1,1), c(0.1,0.1), 1, 0, theta=115, phi=30, shade=0.3)  
}
