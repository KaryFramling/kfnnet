# "R" utilities for dealing with mnist data.
# Partially copied from https://charleshsliao.wordpress.com/2017/02/25/two-ways-of-visualization-of-mnist-with-r/ 
#
# Kary Fr?mling, created September 2019
#

load_image_file <- function(filename) {
	ret = list()
	f = file(filename,'rb')
	readBin(f,'integer',n=1,size=4,endian='big')
	ret$n = readBin(f,'integer',n=1,size=4,endian='big')
	nrow = readBin(f,'integer',n=1,size=4,endian='big')
	ncol = readBin(f,'integer',n=1,size=4,endian='big')
	x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
	ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
	close(f)
	ret
}

load_label_file <- function(filename) {
	f = file(filename,'rb')
	readBin(f,'integer',n=1,size=4,endian='big')
	n = readBin(f,'integer',n=1,size=4,endian='big')
	y = readBin(f,'integer',n=n,size=1,signed=F)
	close(f)
	y
}

mnist.demo <- function() {
  Training <- load_label_file("train-labels-idx1-ubyte")
  Test <- load_label_file("t10k-labels-idx1-ubyte")
  
  # build histograms to see the distribution of digits (Something wrong with this one!)
  resTable <- table(as.numeric(Training))
  par(mfrow = c(1, 1),
      yaxt = "s",
      xaxt = "s")
  par(mar = c(5, 4, 4, 2) + 0.1) # increase y-axis margin.
  plot <-
    plot(resTable,
         main = "Total Number of Digits (Training Set)",
         ylim = c(0, 9500),
         ylab = "Examples Number")
  text(
    x = plot,
    y = resTable + 50,
    labels = resTable,
    cex = 0.75
  )
  
  # Build hist to show numbers of Test set
  testTable <- table(as.numeric(Test))
  x11() # New window
  plot <- plot(testTable,
               main = "Total Number of Digits (Test Set)",
               ylim = c(0, 1500),
               ylab = "Examples Number")
  text(
    x = plot,
    y = testTable + 50,
    labels = testTable,
    cex = 0.75
  )
  
  #show handwritten digit with show_digit(matriximage$x[n,]),n is any number below 60000.
  matriximage <- load_image_file("train-images-idx3-ubyte")
  matrixtest <- load_image_file("t10k-images-idx3-ubyte")
  show_digit <- function(arr784, col = gray(12:1 / 12), ...) {
    image(matrix(arr784, nrow = 28)[, 28:1], col = col, ...)
  }
  
  #Prepare for prediction visualization (WHY TWO DIFFERENT FUNCTIONS, BOTH ARE IDENTICAL (show_digit and show_number)????)
  # show_number <- function(arr784, col=gray(12:1/12), ...) {
  # 	image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
  # }
  x11()
  show_digit(matriximage$x[2017, ])
  x11()
  show_digit(matrixtest$x[9999, ])
  
  # Visualize certain number of digits to have a big picture of the whole situation.
  x11()
  par(mfrow = c(3, 3))
  par(mar = c(0, 0, 0, 0))
  #to see digits with specific number for(i in 1:n)
  for (i in 1:9) {
    m = matrix(matriximage$x[i, ], 28, 28)
    image(m[, 28:1])
  }
}

