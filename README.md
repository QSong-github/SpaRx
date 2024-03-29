# SpaRx: Elucidate spatial heterogeneity of drug sensitivity for personalized treatment

[![DOI](https://zenodo.org/badge/611370637.svg)](https://zenodo.org/badge/latestdoi/611370637)

Spatial cellular heterogeneity contributes to the disease complexity and therapeutic resistance, which commonly involve the interplay between different cell types within the microenvironment. Recent advances of spatial technologies enable the elucidation of single cell heterogeneity with spatial locations that offer remarkable opportunities to understand the cellular interplays and molecular processes involved in therapeutic resistance. In this work, we proposed a novel domain adaptation method, i.e., SpaRx, to reveal the heterogeneity of spatial cellular responses to drugs, via leveraging pharmacogenomics knowledgebase and single-cell spatial profiles. Further application of SpaRx to the state-of-art single-cell spatial transcriptomics data reveals that tumor cells in different locations of tumor lesion present heterogenous levels of sensitive or resistance to drugs. Moreover, resistant tumor cells interact with themselves and the surrounding constituents to form ecosystem capable of drug resistance. Collectively, SpaRx enables the prioritization and interpretation of spatially organized cells in complex tissue, to unveil the molecular mechanisms underpinning drug resistance, which will empower precision medicine by identifying personalized drug targets, effective drug combinations, and drug repositioning.


## Highlights
* SpaRx bridges the gap between drug screening knowledgebase and single-cell spatial transcriptomcis data.
* SpaRx uncovers that tumor cells in different locations of tumor lesion present heterogenous levels of sensitivity or resistance to drugs.
* SpaRx reveals that tumor cells interact with themselves and the microenvironment to form ecosystem capable of drug resistance.


## Data folder structure
```
├── requirement.txt
├── datasets
│      ├── source_adj.csv
│      ├── source_data.csv
│      ├── source_label.csv
│      ├── target_adj.csv
│      ├── target_data.csv
│      ├── target_label.csv (optional)
```
Example datasets for running SpaRx are provided in https://www.dropbox.com/scl/fo/vzai7az362af74au8xqb8/h?rlkey=e7oht0ir80rf5xeqvefufbgsx&dl=0. This folder also contains the organized cell line (GDSC and CCLE) gene expression data (source_domain_expression.csv) and binarized drug response labels (source_domain_binary_labels.csv).

```
ssh run_SpaxRx.sh
```

## Train your own data
you can change the data direction and other hyperparameters in the configure_default.yml

## FAQ
* __How can I install SpaRx?__       
You can download SpaRx from our github link:
  ```
  git clone https://github.com/QSong-github/SpaRx.git
  ```
  SpaRx is built based on pytorch, tested in Ubuntu 18.04, CUDA environment(cuda 11.3)
  the requirement packages includes:
  ```
  torchvision==0.12.0+cu113
  torch==1.11.0
  tqdm==4.63.0
  numpy==1.21.6
  pandas==1.3.5
  PyYAML==6.0
  torch_geometric==2.1.0
  torch_geometric==2.1.0
  torch_scatter==2.0.9
  torch_sparse==0.6.15
  torch_cluster==1.6.0
  ```
  or you can also use the following scripts:
  ```
  pip install -r requirements.txt
  ```
* __I want to try the toy demo, can I run SpaRx in one command line?__    
  You can use the following commands:
  ```
  python  configuration.py 
  python  main_func.py
  ```
  or 
  ```
  ssh run_SpaRx.sh
  ```
* __How can I apply SpaRx in my own dataset? And how to generate the desired format for SpaRx?__         
    Please prepare your data following the format we provided in the example data.

* __How can I tune SpaRx model for best performance?__         
     You can use the following commands:
    ```
    ssh SpaRx_tuning.sh
    ```

* __Do I need a GPU for running SpaRx?__    
    SpaRx is able to run on a standard laptop with or w/o GPU. For computational efficiency, we recommend running SpaRx with GPU.

* __Can I generate my own configuraton file using command line?__    
    You can use the following commands:
    ```
    python configuration.py 
    ```


