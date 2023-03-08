
#' This functions provides the way to generate source graph
#' @param src.dat: rows are cells and columns are genes

#' @return 
#' @export

mnn_graph <- function(src.dat, k1, export_file){
    suppressPackageStartupMessages(library('Seurat'))
    
    object1 <- CreateSeuratObject(counts=t(src.dat),
                                  project="1",assay = "Data1",
                                  min.cells=0,min.features = 0,
                                  names.field = 1,
                                  names.delim = "_")
    objects <- list(object1,object1)
    
    objects1 <- lapply(objects,function(obj){
        obj <- NormalizeData(obj,verbose=F)
        obj <- FindVariableFeatures(obj,
                                    selection.method = "vst",
                                    nfeatures = 2000,verbose=F)
        obj <- ScaleData(obj,features=rownames(obj),
                         verbose=FALSE)
        obj <- RunPCA(obj, features=rownames(obj),
                      verbose = FALSE)
        return(obj)})

    d2.nn <- FindIntegrationAnchors(object.list = objects1,
                                    k.anchor=k1, verbose=F)
    
    d2.arc=d2.nn@anchors
    d2.arc1=cbind(d2.arc[d2.arc[,4]==1,1],d2.arc[d2.arc[,4]==1,2],d2.arc[d2.arc[,4]==1,3])
    adj=d2.arc1[d2.arc1[,3]>0,1:2]
    adj[,1] <- rownames(src.dat)[adj[,1]]
    adj[,2] <- rownames(src.dat)[as.numeric(adj[,2])]
    colnames(adj) <- c('cell1','cell2')
    write.csv(adj,file= export_file,quote=F)
    print ('done')
}

#' This functions provides the way to generate target graph
#' @param tar.dat: rows are genes and columns are cells

#' @return 
#' @export

#' generate adjacent graph    
spatial_graph <- function(tar.meta, k2, export_dir){
    st.meta <- tar.meta
    library(RANN); df = st.meta[,c('x','y')]
    closest <- RANN::nn2(data = df, k = k2)[[1]]

    N1 = nrow(closest)*k2
    
    adj <- matrix(0,nrow=N1,ncol=2)
    for (i in 1:nrow(closest)){
        id = (i-1)*k2 + 1
        adj[id:(id+k2-1),1] = closest[i,1]
        adj[id:(id+k2-1),2] = closest[i,1:k2]
    }

    adj[,1] <- rownames(tar.meta)[adj[,1]]
    adj[,2] <- rownames(tar.meta)[as.numeric(adj[,2])]
    colnames(adj) <- c('cell1','cell2')
    write.csv(adj,file=paste0(export_dir,'/target_adj.csv'),quote=F)
    print ('done')
}

SpaRx_graph <- function(src.exp, tar.exp, tar.meta, style, K, dir){
    if (style == 'simulation'){        
        mnn_graph(src.dat = src.exp, k1 = K,
                  export_file =paste0(dir,'/source_adj.csv'))
        mnn_graph(src.dat = tar.exp, k1 = K,
                  export_file =paste0(dir,'/target_adj.csv'))
    }
    if (style == 'real'){        
        mnn_graph(src.dat = src.exp, k1 = K,
                  export_file =paste0(dir,'/source_adj.csv'))
        spatial_graph(tar.meta = tar.meta, k2 = K,
                      export_dir = dir)
    }
}
