rm(list=ls())

library('gridExtra')
library('ggplot2')
library('reshape2')

system("cd /home/jack/Dropbox/source/cse6230/proj1");

###### DO DATA STUFF #############################
summa_data = read.table("Results/mm_singlenode.csv",sep=",");
localmm_data = read.table("Results/summa_singlenode.csv",sep=",");

colnames(summa_data) <- c("nodes","ppn","m","n","k","px","py","panel_size","time","trials","timepertrial","type","threads");
colnames(localmm_data) <- c("nodes","ppn","m","n","k","time","trials","timepertrial","threads");

summa_castdata = ddply(summa_data, c("nodes","procs","m","n","k","px","py","panel_size","threads","type"), function(df) data.frame(meantime=mean(df$timepertrial)));
local_mmdata = ddply(localmm_data, c("m","n","k","time","trials","timepertrial"), function(df) data.frame(meantime=mean(df$timepertrial)));

m_s=summa_castdata$m; m_l = localmm_castdata$m;
n_s=summa_castdata$n; n_l = localmm_castdata$n;
k_s=summa_castdata$k; k_l = localmm_castdata$k;
px_s=summa_castdata$px; px_l = localmm_castdata$px;
py_s=summa_castdata$py; py_l = localmm_castdata$py;
pb_s=summa_castdata$panel_size; pb_l = localmm_castdata$panel_size;
threads_s=summa_castdata$threads; threads_l = localmm_castdata$threads;
type_s=summa_castdata$type; type_l = localmm_castdata$type;
tpt_s=summa_castdata$timepertrial; tpt_l = localmm_castdata$timepertrial;

###### PLOTS #######################################################
# 1. Compare local_mm implementations on single node and multiple nodes
#     - For single node, vary different sizes over different threads for 
#       OpenMP and MKL. Compare this with Naive implementation over processes
#       instead of threads.
#     - For multi node, 
# 2. Compare summa on single node and multiple nodes
#     - For single node, vary different sizes, different panels, and 
#       different threads 
#     - For multi node, vary different nodes, sizes, panels
#
#

