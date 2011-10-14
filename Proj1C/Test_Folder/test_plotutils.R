###################################################
# STEP 1: GET ENVIRONMENT READY

rm(list=ls())

# install these packages if you haven't yet
#install.packages('gridExtra')
#install.packages('ggplot2')
#install.packages('reshape2')

library('gridExtra')
library('ggplot2')
library('reshape2')

#setwd("/home/jack/Dropbox/source/cse6230/project1");
setwd("/nethome/jchua3/project1/");
results_folder="Results/proj1b/Weak_Scaling/";
#results_folder="Results/proj1b/Strong_Scaling/";

source("test_folder/Rutil/plotutils.R");


##################################################
# STEP 2: GET DATA AND CLEAN
mm_sn_data = read.table(paste(results_folder,"mm_single_node.csv",sep=""),sep=",");
mn_data = read.table(paste(results_folder,"multiple_nodes.csv",sep=""),sep=",");

mn_data = clean_data(mn_data,"SUMMA");
mm_sn_data = clean_data(mm_sn_data,"MM");

##################################################
# STEP 3: COMPUTE MIN AND MAX TIME PER IMPLEMENTATION
# In this case, we're looking for the minimum time per k per implementation
# and the associated node/ppn/tpp counts
mn_stats = find_stats(mn_data,c("k","imp"),min,c("nodes","ppn","tpp"));

##################################################
# STEP 4: PLOT STATISTICS
# Use ggplot
plotname=paste(results_folder,"mn_stats_plot.png",sep="");
xaxislabel="nodes";
yaxislabel="lg(time per trial)";
plottitle="Test title";

p <- ggplot(mn_stats,aes(x=nodes,y=tpt,group=imp));
p + geom_line(aes(colour=imp)) + opts(title=plottitle) + scale_x_continuous(xaxislabel) + scale_y_continuous(yaxislabel);
ggsave(plotname,scale=1);


##################################################
# STEP 4: OUTPUT TO CSV FILE
filename=paste(results_folder,"mn_stats_testoutput.csv",sep="");
dfsToCSV(mn_data,mn_stats,filename=filename);
