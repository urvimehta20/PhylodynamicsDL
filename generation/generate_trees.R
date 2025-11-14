library(phylopomp)
library(pbapply)
library(parallel)


pbapply::pboptions(
    type = "timer", # or "timer" if you like the elapsed/remaining time
    style = 1, 
    min_time = 0 # show immediately even for short jobs
)


set.seed(22 / 7)

n_trees <- 20000
samples <- matrix(runif(0.1, 10, n = 2 * n_trees), ncol = 2)


params <- lapply(1:nrow(samples), function(i) {
    samples[i, ]
})
cl <- makeForkCluster(8)

trees <- unlist(pbapply::pblapply(params, function(param) {
    tree <- simulate("Moran", mu = param[1], psi = param[2], time = 10)
    newick(tree)
}, cl = cl))

write.table(trees, file = "./output.newick", col.names = FALSE, row.names = FALSE, quote = FALSE)
