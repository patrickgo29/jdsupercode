rm(list=ls())

library('gridExtra')
library('ggplot2')
library('reshape2')

#setwd("/home/jack/Dropbox/source/cse6230/project1");
setwd("/nethome/jchua3/project1/");
#results_folder="Results/Weak_Scaling/";
#results_folder="Results/Strong_Scaling/";
results_folder="Results/Odd_Sizes/";

###### DO DATA STUFF #############################
#
# TIME_MM: mm, nodes, ppn, tpp, imp, m, n, k, tet, trials, tpt
# SUMMA: summa, nodes, ppn, tpp, imp, m, n, k, x, y, b, tet, trials, tpt

ssn_data = read.table(paste(results_folder,"summa_single_node.csv",sep=""),sep=",");
mm_sn_data = read.table(paste(results_folder,"mm_single_node.csv",sep=""),sep=",");

colnames(ssn_data) <- c("alg","nodes","ppn","tpp","imp","m","n","k","x","y","b","tet","trials","tpt");
colnames(mm_sn_data) <- c("alg","nodes","ppn","tpp","imp","m","n","k","tet","trials","tpt");

# Refactor data
ssn_data <- transform(ssn_data, nodes=as.numeric(nodes), ppn=as.numeric(ppn),
                      tpp=as.numeric(tpp),m=as.numeric(m),n=as.numeric(n),
                      k=as.numeric(k),x=as.numeric(x),y=as.numeric(y),
                      b=as.factor(b),tet=as.numeric(tet),trials=as.numeric(trials),
                      tpt=as.numeric(tpt), alg=as.character(alg), 
                      imp=as.character(imp));
mm_sn_data <- transform(mm_sn_data, nodes=as.numeric(nodes), ppn=as.numeric(ppn),
                      tpp=as.numeric(tpp),m=as.numeric(m),n=as.numeric(n),
                      k=as.numeric(k),tet=as.numeric(tet),trials=as.numeric(trials),
                      tpt=as.numeric(tpt), alg=as.character(alg), 
                      imp=as.character(imp));

# Transform time to milliseconds
ssn_data$tpt_mil = 1000*ssn_data$tpt;
mm_sn_data$tpt_mil = 1000*mm_sn_data$tpt;

# Transform time to log scale
ssn_data$logtpt = log2(ssn_data$tpt_mil);
mm_sn_data$logtpt = log2(mm_sn_data$tpt_mil);

# Get rid of observations that are invalid (block size too large)
ssn_data_adj=NULL; mm_sn_data_adj=NULL; mn_data_adj=NULL;
for (i in 1:length(mn_data$alg)) {
    rowtemp = mn_data[i,];
    m = rowtemp$m; n=rowtemp$n; k=rowtemp$k; x=rowtemp$x;
    y = rowtemp$y; pb=as.numeric(as.character(rowtemp$b));
    A_x = k/x; A_y = m/y; B_x = n/x; B_y = k/y;
    if ((pb>A_x | pb>A_y | pb>B_x | pb>B_y)) {
        next
    } 
    if (A_x%%pb!=0 | A_y%%pb!=0 | B_x%%pb!=0 | B_y%%pb!=0) {
        next
    }
    if (m%%y!=0 | n%%x!=0 | k%%x!=0 | k%%y!=0) {
        next
    }
    mn_data_adj = rbind(mn_data_adj,rowtemp);
}
ssn_data = ssn_data_adj;


###### PLOTS #######################################################
# Note: by default, ggplot always clears the screen and uses the entire device
# Note: names specify implementation, number of nodes, and panel block size

# 1. plots for single node matrix multiply (MM_SN) ***********************

# find best naive implementation from above for each panel size
sizes = unique(ssn_data$m);
ix = 0;
b=c(); m=c(); t=c(); mean=c(); median=c();
x=c(); y=c();
for (i in sizes) {
    ix=ix+1;
    tempdata = ssn_data[ssn_data$imp=='naive' & ssn_data$m==i,];
    means    = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
    colnames(means) = c("panels","times");
    panelTimes = data.frame(unique(as.numeric(as.character(tempdata$b))),means$times);
    colnames(panelTimes) = c("panels","times");
    b[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$panels;
    m[ix] = i;
    t[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$times;
    mean[ix] = mean(means$times);
    median[ix] = median(means$times);
}
minPanels = data.frame(b,m,t,mean,median);
table_min_naive_ssn_panels = minPanels;

sizes = unique(ssn_data$m);
ix = 0;
b=c(); m=c(); t=c(); mean=c(); median=c();
for (i in sizes) {
    ix=ix+1;
    tempdata = ssn_data[ssn_data$imp=='naive' & ssn_data$m==i,];
    means    = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
    colnames(means) = c("panels","times");
    panelTimes = data.frame(unique(as.numeric(as.character(tempdata$b))),means$times);
    colnames(panelTimes) = c("panels","times");
    b[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$panels;
    m[ix] = i;
    t[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$times;
    mean[ix] = mean(means$times);
    median[ix] = median(means$times);
}
maxPanels = data.frame(b,m,t,mean,median);
table_max_naive_ssn_panels = maxPanels;

table_min_openmp_mm_sn_panels = NULL;
table_min_mkl_mm_sn_panels = NULL;
table_max_openmp_mm_sn_panels = NULL;
table_max_mkl_mm_sn_panels = NULL;


for (i in unique(mm_sn_data$m)) {          # for each unique size,
    for (j in unique(mm_sn_data$nodes)) {    # and node count
         # get the subset of data we want
         tempdata = mm_sn_data[mm_sn_data$m==i & mm_sn_data$nodes==j,];

         # find corresponding naive implementation and append to our data
         panel = minPanels[minPanels$m==i,]$b;
         naivedata = ssn_data[ssn_data$imp=='naive' & ssn_data$m==i &
                              ssn_data$b==panel,];
         naivedata = naivedata[c("alg","nodes","ppn","imp","tpp","m","n","k","tet",
                                 "trials","tpt","tpt_mil","logtpt")];
         naivedata[c("tpp")] = naivedata[c("ppn")];  #quick hack for plotting          
         tempdata = rbind(tempdata,naivedata);

        # create print out tables for openmp and mkl
         min_openmp = tempdata[tempdata$imp=='openmp',];
         min_openmp = min_openmp[min_openmp$tpt==min(min_openmp$tpt),];
         max_openmp = tempdata[tempdata$imp=='openmp',];
         max_openmp = max_openmp[max_openmp$tpt==max(max_openmp$tpt),];
         min_mkl    = tempdata[tempdata$imp=='mkl',];
         min_mkl    = min_mkl[min_mkl$tpt==min(min_mkl$tpt),];
         max_mkl = tempdata[tempdata$imp=='mkl',];
         max_mkl = max_mkl[max_mkl$tpt==max(max_mkl$tpt),];
         table_min_openmp_mm_sn_panels =rbind(table_min_openmp_mm_sn_panels,min_openmp);
         table_min_mkl_mm_sn_panels    =rbind(table_min_mkl_mm_sn_panels,min_mkl);
         table_max_openmp_mm_sn_panels =rbind(table_max_openmp_mm_sn_panels,max_openmp);
         table_max_mkl_mm_sn_panels    =rbind(table_max_mkl_mm_sn_panels,max_mkl);

         # naming
         tempname = paste("mm_sn-",as.character(i),"-",as.character(j),
                             "nodes",sep="");
            
         # plot and save to the specified savefolder and filename
         p <- ggplot(tempdata[tempdata$tpp==1,], aes(x=k, y=logtpt, group=imp));
         p + geom_line(aes(colour=imp)) + opts(title=tempname) + scale_x_continuous('k') + scale_y_continuous('lg(time per trial)');
         ggsave("1threadOddSize.png",scale=1);

         p <- ggplot(tempdata[tempdata$tpp==8,], aes(x=k, y=logtpt, group=imp));
         p + geom_line(aes(colour=imp)) + opts(title=tempname) + scale_x_continuous('k') + scale_y_continuous('lg(time per trial)');
         ggsave("8threadsOddSize.png",scale=1);

    }
}

# adjust the format of tables a bit, add means and medians
table_min_openmp_mm_sn_panels$mean =aggregate(mm_sn_data$tpt[mm_sn_data$imp=='openmp'],
                       by=list(mm_sn_data$m[mm_sn_data$imp=='openmp']),FUN=mean)[,2];
table_max_openmp_mm_sn_panels$mean =aggregate(mm_sn_data$tpt[mm_sn_data$imp=='openmp'],
                       by=list(mm_sn_data$m[mm_sn_data$imp=='openmp']),FUN=mean)[,2];
table_min_mkl_mm_sn_panels$mean = aggregate(mm_sn_data$tpt[mm_sn_data$imp=='mkl'],
                       by=list(mm_sn_data$m[mm_sn_data$imp=='mkl']),FUN=mean)[,2];
table_max_mkl_mm_sn_panels$mean = aggregate(mm_sn_data$tpt[mm_sn_data$imp=='mkl'],
                       by=list(mm_sn_data$m[mm_sn_data$imp=='mkl']),FUN=mean)[,2];

table_min_openmp_mm_sn_panels$median=aggregate(mm_sn_data$tpt[mm_sn_data$imp=='openmp'],
                       by=list(mm_sn_data$m[mm_sn_data$imp=='openmp']),FUN=median)[,2];
table_max_openmp_mm_sn_panels$median=aggregate(mm_sn_data$tpt[mm_sn_data$imp=='openmp'],
                       by=list(mm_sn_data$m[mm_sn_data$imp=='openmp']),FUN=median)[,2];
table_min_mkl_mm_sn_panels$median=aggregate(mm_sn_data$tpt[mm_sn_data$imp=='mkl'],
                       by=list(mm_sn_data$m[mm_sn_data$imp=='mkl']),FUN=median)[,2];
table_max_mkl_mm_sn_panels$median=aggregate(mm_sn_data$tpt[mm_sn_data$imp=='mkl'],
                       by=list(mm_sn_data$m[mm_sn_data$imp=='mkl']),FUN=median)[,2];

table_min_openmp_mm_sn_panels = table_min_openmp_mm_sn_panels[c("m","tpt",
                                                             "mean","median")];
table_min_mkl_mm_sn_panels = table_min_mkl_mm_sn_panels[c("m","tpt","mean","median")];
table_max_openmp_mm_sn_panels = table_max_openmp_mm_sn_panels[c("m","tpt",
                                                            "mean","median")];
table_max_mkl_mm_sn_panels = table_max_mkl_mm_sn_panels[c("m","tpt","mean","median")];

###### ANALYSIS #####################################################
print(table_min_naive_ssn_panels)
print(table_max_naive_ssn_panels)
print(table_min_openmp_mm_sn_panels)
print(table_max_openmp_mm_sn_panels)
print(table_min_mkl_mm_sn_panels)
print(table_max_mkl_mm_sn_panels)
 
