setwd('/home/qianqian/drug_simulation_new/simulate_data')

lab <- read.csv(file='source_domain_binary_labels.csv',
         stringsAsFactors=F,check.names=T,row.names=1)
exp <- read.csv(file='source_domain_expression.csv',
                stringsAsFactors=F,check.names=T,row.names=1)

var.a <- apply(exp,2,var)
ps <- which(var.a > mean(var.a))    
sub.exp <- exp[,ps]

source('sim_utils.R'); library(tidyselect)

drug.names <- colnames(lab)

noise.var = 1
N <- length(drug.names)
prop <- 0.3

dirs0='simulated_prop_0.3/'
dir.create(dirs0)

for ( id in 1:N){
    d.name <- paste0('drug-',id)
    dir0 = paste0(dirs0,d.name)    
    dir.create(dir0)
    dr.name =drug.names[id]
#' ---------- target data and label ----------

    al.lab <- lab[,colnames(lab)==dr.name,drop=FALSE]
    al.exp <- sub.exp
    print (dim(sub.exp))
    fo1 <- which(al.lab[,1]=='resistant')
    fo2 <- which(al.lab[,1]=='sensitive')

    set.seed(123); fos1 <- sample(fo1,round(length(fo1)*prop))
    set.seed(123); fos2 <- sample(fo2,round(length(fo2)*prop))
    fot1 <- setdiff(fo1,fos1)
    fot2 <- setdiff(fo2,fos2)

    t.exp <- al.exp[c(fot1,fot2),]
    t.lab <- al.lab[c(fot1,fot2),,drop=FALSE]
    colnames(t.lab) <- 'label'
    identical(rownames(t.exp),rownames(t.lab))
    
    #' ----------------------------------------------------
    sim.1 <- sim_data(count_mtrx = t(t.exp),
                      count_meta = t.lab,
                      drug = 'label',
                      response ='resistant')
    num <- nrow(sim.1[[1]]) * ncol(sim.1[[1]])
    set.seed(123); ar.noise <- rnorm(num, mean = 0, sd = noise.var)
    mt.noise = matrix(ar.noise,nrow=nrow(sim.1[[1]]),ncol=ncol(sim.1[[1]]))
    sims.1 = sim.1[[1]] + mt.noise

    sim.2 <- sim_data(count_mtrx = t(t.exp),
                      count_meta = t.lab,
                      drug = 'label',
                      response ='sensitive')

    num <- nrow(sim.2[[1]]) * ncol(sim.2[[1]])
    set.seed(321); ar.noise <- rnorm(num, mean = 0, sd = noise.var)
    mt.noise = matrix(ar.noise,nrow=nrow(sim.2[[1]]),ncol=ncol(sim.2[[1]]))
    sims.2 = sim.2[[1]] + mt.noise
    
    sims.data <- as.matrix(cbind(sims.1,sims.2))
    colnames(sims.data) <- paste0('tar_cell_',1:ncol(sims.data))
    sims.lab <- data.frame(
        label=c(rep('resistant',1000),rep('sensitive',1000)))
    rownames(sims.lab) <- colnames(sims.data)

    e.f1 <- paste0(dir0,'/target_data.csv')
    l.f1 <- paste0(dir0,'/target_label.csv')

    sims.data <- t(sims.data)
    write.csv(sims.data,file =e.f1,quote=F)
    write.csv(sims.lab,file =l.f1,quote=F)

    tar.adj <- cbind(rownames(sims.data),rownames(sims.data))
    colnames(tar.adj) <- c('cell1','cell2')
    write.csv(tar.adj,file=paste0(dir0,'/target_adj.csv'),quote=F)
    #' ----------------------------------------------------

#' ---------- source data and label ----------
    s.exp <- al.exp[c(fos1,fos2),]
    s.lab <- al.lab[c(fos1,fos2),,drop=FALSE]
    colnames(s.lab) <- 'label'
    identical(rownames(s.exp),rownames(s.lab))

    e.f2 <- paste0(dir0,'/source_data.csv')
    l.f2 <- paste0(dir0,'/source_label.csv')

    write.csv(s.exp,file =e.f2,quote=F)
    write.csv(s.lab,file =l.f2,quote=F)

    src.adj <- cbind(rownames(s.lab),rownames(s.lab))
    colnames(src.adj) <- c('cell1','cell2')
    write.csv(src.adj,
              file=paste0(dir0,'/source_adj.csv'),quote=F)
}



