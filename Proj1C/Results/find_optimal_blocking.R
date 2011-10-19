rm(list=ls())

#install.packages('gridExtra')
#install.packages('ggplot2')
#install.packages('reshape2')
#install.packages('plyr')

library('gridExtra')
library('ggplot2')
library('reshape2')
library('plyr')

#workingDir="/home/owner/Dropbox/source/cse6230/proj1/Proj1C/Results/";
workingDir="/nethome/jchua3/jdsupercode/Proj1C/Results/Optimal_Blocking/";
setwd(workingDir)

# GET DATA
cache_data = read.table(paste(workingDir,"total.csv",sep=""),sep=",");
colnames(cache_data) <- c("type","cbl","cop","dunno","bm","bn","bk","m","n","k","time");
cache_data <- transform(cache_data, cbl=as.numeric(cbl), cop=as.numeric(cop),
                        dunno=as.numeric(dunno), bm=as.numeric(bm),
                        bn=as.numeric(bn), bk=as.numeric(bk), m=as.numeric(m),
                        n=as.numeric(n), k=as.numeric(k),
                        time=as.numeric(time));


# TUNE FOR FIRST LEVEL OF CACHE
#first_cache_sizes = unique(cache_data$bm);
first_level_mean = ddply(cache_data,c("bm"),function(df)mean(df$time));
second_level_mean = ddply(cache_data,c("bn"),function(df)mean(df$time));
third_level_mean = ddply(cache_data,c("bk"),function(df)mean(df$time));

first_level_median = ddply(cache_data,c("bm"),function(df)median(df$time));
second_level_median = ddply(cache_data,c("bn"),function(df)median(df$time));
third_level_median = ddply(cache_data,c("bk"),function(df)median(df$time));

first_level_min = ddply(cache_data,c("bm"),function(df)min(df$time));
second_level_min = ddply(cache_data,c("bn"),function(df)min(df$time));
third_level_min = ddply(cache_data,c("bk"),function(df)min(df$time));

# global minimum
cache_data[cache_data$time == min(cache_data$time),]

# plot results
p1 <- ggplot(first_level_mean, aes(x=bm,y=V1));
p1 + geom_line() + opts(title="Preliminary Tuning for bm") + scale_x_continuous("bm") + scale_y_continuous("Time per trial");
ggsave("bm_prelim_tune.png",scale=1);





