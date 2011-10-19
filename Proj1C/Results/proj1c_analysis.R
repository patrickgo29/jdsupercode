rm(list=ls())

library('gridExtra')
library('ggplot2')
library('reshape2')
library('plyr')

workingDir="/nethome/jchua3/jdsupercode/Proj1C/";
dataDir=paste(workingDir,"Results/Strong_Scaling/",sep="");
setwd(workingDir)

source("Test_Folder/plotutils.R")

##### DATA #######
ss_data = read.table(paste(dataDir,"results.csv",sep=""),sep=",");
colnames(ss_data) <- c("type","cores","cbl","cop","bm","bn","bk","m","n","k","tpt");
ss_data <- transform(ss_data, cores=as.numeric(cores),
                    cbl=as.numeric(cbl), cop=as.numeric(cop), 
                    bm=as.numeric(bm), bn=as.numeric(bn),
                    bk=as.numeric(bk), m=as.numeric(m),
                    n=as.numeric(n),k=as.numeric(k),
                    tpt=as.numeric(tpt));
ss_data$tpt = log10(ss_data$tpt*1000)

###### STATS #####
stats = find_stats(ss_data,c("type","cbl","cop","cores","m","n","k"),min,c());

###### EXPORT STATS
dfsToCSV(stats,filename="Results/Strong_Scaling/SS_stats.csv");

##### PLOTS ######

# first plot each implementation and group by size
data_p1 = stats[stats$type=="mkl",];
data_p2 = stats[(stats$type=="openmp" & stats$cop==0 & stats$cbl==0),];
data_p3 = stats[(stats$type=="openmp" & stats$cop==0 & stats$cbl==1),];
data_p4 = stats[(stats$type=="openmp" & stats$cop==1 & stats$cbl==1),];


plottitle="Strong Scaling MKL";
xaxislabel="Threads";
yaxislabel="Log(Time Per Trial)";
p1 <- ggplot(data_p1,aes(x=cores,y=tpt,group=m));
p1 + geom_line(aes(colour=m)) + opts(title=plottitle) + scale_x_continuous(xaxislabel) + scale_y_continuous(yaxislabel);
ggsave("Results/Strong_Scaling/SS_mkl.png",scale=1);

plottitle="Strong Scaling OpenMP";
xaxislabel="Threads";
yaxislabel="Log(Time Per Trial)";
p1 <- ggplot(data_p2,aes(x=cores,y=tpt,group=m));
p1 + geom_line(aes(colour=m)) + opts(title=plottitle) + scale_x_continuous(xaxislabel) + scale_y_continuous(yaxislabel);
ggsave("Results/Strong_Scaling/SS_openmp.png",scale=1);

plottitle="Strong Scaling OpenMP with Cache Blocking";
xaxislabel="Threads";
yaxislabel="Log(Time Per Trial)";
p1 <- ggplot(data_p3,aes(x=cores,y=tpt,group=m));
p1 + geom_line(aes(colour=m)) + opts(title=plottitle) + scale_x_continuous(xaxislabel) + scale_y_continuous(yaxislabel);
ggsave("Results/Strong_Scaling/SS_openmp_cbl.png",scale=1);

plottitle="Strong Scaling OpenMP with Cache Blocking and Copy Optimization";
xaxislabel="Threads";
yaxislabel="Log(Time Per Trial)";
p1 <- ggplot(data_p4,aes(x=cores,y=tpt,group=m));
p1 + geom_line(aes(colour=m)) + opts(title=plottitle) + scale_x_continuous(xaxislabel) + scale_y_continuous(yaxislabel);
ggsave("Results/Strong_Scaling/SS_openmp_cbl_cop.png",scale=1);

