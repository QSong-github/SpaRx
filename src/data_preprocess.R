args <- commandArgs(trailingOnly = TRUE)
print(args)

sc.file = args[[1]]
st.file = args[[2]]
meta.file = args[[3]]  #' NULL or file
style0 = args[[4]] #' simulation or real
k0 = as.integer(args[[5]])
dir0 = args[[6]]

print ('meta')
print (meta.file)

#' --- example ---
sc.exp <- read.csv(file=sc.file,stringsAsFactors=F,check.names=F,row.names=1)
st.exp <- read.csv(file=st.file,stringsAsFactors=F,check.names=F,row.names=1)

if (style0=='simulation'){ st.meta = NULL } else { st.meta <- read.csv(file=meta.file,stringsAsFactors=F,check.names=F,row.names=1) }
    

source('./src/preprocess_utils.R')

SpaRx_graph(src.exp = sc.exp,
            tar.exp = st.exp,
            tar.meta = st.meta,
            style = style0,
            K=k0,
            dir=dir0)
