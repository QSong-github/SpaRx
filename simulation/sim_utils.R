
sim_data = function(count_mtrx,
                    count_meta,
                    drug,
                    response,
                    n = 1000, verbose = TRUE){
    suppressMessages(require(DropletUtils))
    suppressMessages(require(purrr))
    suppressMessages(require(dplyr))
    suppressMessages(require(tidyr))
    
    count_meta$row <- 'new'
    ds_spots <- lapply(seq_len(n), function(i) {
        ab <- which(count_meta[,drug] ==response)
        new.tmp <- colnames(count_mtrx)[ab]
        cell_pool <- sample(new.tmp, sample(x = 2:10, size = 1))
        
        pos <- which(colnames(count_mtrx) %in% cell_pool)
        tmp_ds <- count_meta[pos, ] %>% mutate(weight = 1)
    
        name_simp <- paste("spot_", i, sep = "")
    
        spot_ds <- tmp_ds %>% dplyr::select(all_of(drug), 
                                        weight) %>% dplyr::group_by(!!sym(drug)) %>% 
        dplyr::summarise(sum_weights = sum(weight)) %>% dplyr::ungroup() %>% 
        tidyr::pivot_wider(names_from = all_of(drug), 
                           values_from = sum_weights) %>% dplyr::mutate(name = name_simp)
    
        syn_spot <- rowSums(as.matrix(count_mtrx[, cell_pool]))

        if (sum(syn_spot) > 25000) {
            syn_spot_sparse <- DropletUtils::downsampleMatrix(Matrix::Matrix(syn_spot, sparse = T), prop = 20000/sum(syn_spot))
        }
        else {
            syn_spot_sparse <- Matrix::Matrix(syn_spot, sparse = T)
        }
        names_genes = rownames(count_mtrx)
        rownames(syn_spot_sparse) <- names_genes
        colnames(syn_spot_sparse) <- name_simp
        return(list(syn_spot_sparse, spot_ds))
    })

    ds_syn_spots <- purrr::map(ds_spots, 1) %>%
    base::Reduce(function(m1, 
                          m2) cbind(unlist(m1), unlist(m2)), .)
    ds_spots_metadata <- purrr::map(ds_spots, 2) %>% dplyr::bind_rows() %>% 
        data.frame()

    ds_spots_metadata$type <- response
    ds_spots_metadata$num <- ds_spots_metadata[,1]
    ds_spots_metadata <- ds_spots_metadata[,-1]
    return(list(topic_profiles = ds_syn_spots, cell_composition = ds_spots_metadata))
}

sim_data2 = function(count_mtrx,
                     count_meta,
                     response,
                     n = 1000, verbose = TRUE){
    suppressMessages(require(DropletUtils))
    suppressMessages(require(purrr))
    suppressMessages(require(dplyr))
    suppressMessages(require(tidyr))
    
    count_meta$row <- 'new'
    ds_spots <- lapply(seq_len(n), function(i) {
        ab <- which(count_meta$drug ==response)
        new.tmp <- count_meta$Cell[ab]
        cell_pool <- sample(new.tmp, sample(x = 2:10, size = 1))
        
        pos <- which(colnames(count_mtrx) %in% cell_pool)
        tmp_ds <- count_meta[pos, ] %>% mutate(weight = 1)
    
        name_simp <- paste("spot_", i, sep = "")
    
        syn_spot <- rowSums(as.matrix(count_mtrx[, pos]))

        if (sum(syn_spot) > 25000) {
            syn_spot_sparse <- DropletUtils::downsampleMatrix(Matrix::Matrix(syn_spot, sparse = T), prop = 20000/sum(syn_spot))
        }
        else {
            syn_spot_sparse <- Matrix::Matrix(syn_spot, sparse = T)
        }
        names_genes = rownames(count_mtrx)
        rownames(syn_spot_sparse) <- names_genes
        colnames(syn_spot_sparse) <- name_simp
        return(syn_spot_sparse)
    })

    ds_syn_spots <- as.matrix(do.call(cbind,ds_spots))
    return(ds_syn_spots)
}

