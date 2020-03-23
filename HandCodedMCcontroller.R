handcoded.mc.controller.new <- function(mcmodel) {
  mcm <- mcmodel

  doControl <- function() {
    x <- mcm$get.x()
    v <- mcm$get.v()
    if ( v >= 0.0 )
      mcm$set.thrust(1.0)
    else
      mcm$set.thrust(-1.0)
  }
  
  list (
        run = function() {
          while ( !mcm$at.goal() ) {
            doControl();
            mcm$step();
          }
        }
        )
}

test <- function() {
  source("MountainCar.R")
  mc <- mountaincar.new()
  controller <- handcoded.mc.controller.new(mc)
  controller$run()
  print(paste(c(mc$get.x(), mc$get.y(), mc$get.v())))
}
