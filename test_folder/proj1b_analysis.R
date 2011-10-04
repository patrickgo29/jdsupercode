rm(list=ls())

library('gridExtra')
library('ggplot2')
library('reshape2')

setwd("/home/jack/Dropbox/source/cse6230/project1");
results_folder="Results2/";

###### DO DATA STUFF #############################
#
# TIME_MM: mm, nodes, ppn, tpp, imp, m, n, k, tet, trials, tpt
# SUMMA: summa, nodes, ppn, tpp, imp, m, n, k, x, y, b, tet, trials, tpt

ssn_data = read.table(paste(results_folder,"summa_single_node.csv",sep=""),sep=",");
mm_sn_data = read.table(paste(results_folder,"mm_single_node.csv",sep=""),sep=",");
mn_data = read.table(paste(results_folder,"multiple_nodes.csv",sep=""),sep=",");

colnames(ssn_data) <- c("alg","nodes","ppn","tpp","imp","m","n","k","x","y","b","tet","trials","tpt");
colnames(mm_sn_data) <- c("alg","nodes","ppn","tpp","imp","m","n","k","tet","trials","tpt");
colnames(mn_data) <- c("alg","nodes","ppn","tpp","imp","m","n","k","x","y","b","tet","trials","tpt");

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
mn_data <- transform(mn_data, nodes=as.numeric(nodes), ppn=as.numeric(ppn),
                      tpp=as.numeric(tpp),m=as.numeric(m),n=as.numeric(n),
                      k=as.numeric(k),x=as.numeric(x),y=as.numeric(y),
                      b=as.factor(b),tet=as.numeric(tet),trials=as.numeric(trials),
                      tpt=as.numeric(tpt), alg=as.character(alg), 
                      imp=as.character(imp));

# Transform time to milliseconds
ssn_data$tpt_mil = 1000*ssn_data$tpt;
mm_sn_data$tpt_mil = 1000*mm_sn_data$tpt;
mn_data$tpt_mil = 1000*mn_data$tpt;

# Transform time to log scale
ssn_data$logtpt = log10(ssn_data$tpt_mil);
mm_sn_data$logtpt = log10(mm_sn_data$tpt_mil);
mn_data$logtpt = log10(mn_data$tpt_mil);

###### PLOTS #######################################################
# 1. Plots for single node matrix multiplication. 
# 2. Plots for single node SUMMA multiplication.
# 3. Plots for multi node SUMMA multiplication.
#
# Note: by default, ggplot always clears the screen and uses the entire device
# 
# Note: names specify implementation, number of nodes, and panel block size

# 1. plots for single node matrix multiply (MM_SN)
for (i in unique(mm_sn_data$m)) {          # for each unique implementation,
    for (j in unique(mm_sn_data$nodes)) {    # and node count
         # get the subset of data we want
         tempdata = mm_sn_data[mm_sn_data$m==i & mm_sn_data$nodes==j,];

         # naming
         tempname = paste("mm_sn-",as.character(i),"-",as.character(j),
                             "nodes",sep="");
            
         # plot and save to the specified savefolder and filename
         p <- ggplot(tempdata, aes(x=tpp, y=logtpt, group=imp));
         p + geom_line(aes(colour=imp)) + opts(title=tempname);
         ggsave(paste(results_folder,tempname,".pdf",sep=""),scale=1);
    }
}

# 2. plot single nodes for 'naive' case. we're doing this because
# we are using threads instead of processes. only iterate over panel block sizes
for (i in unique(mm_sn_data$m)) {
        tempdata = ssn_data[ssn_data$imp=='naive' &
                            ssn_data$m==i,];
        tempname = paste("ssn-naive-",as.character(i),sep="");
        p <- ggplot(tempdata, aes(x=ppn, y=logtpt, group=b, colour=b));
        p + geom_line() + opts(title=tempname); 
        ggsave(paste(results_folder,tempname,".pdf",sep=""),scale=1);
}

# 3. plots for multiple nodes (MN)
# max 8 threads per node
# only openMP/MKL uses threads

# find best naive implementation from above for each panel size
sizes = unique(ssn_data$m);
ix = 0;
minPanel = c();
for (i in sizes) {
    ix=ix+1;
    tempdata = ssn_data[ssn_data$imp=='naive' & ssn_data$m==i,];
    means    = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
    colnames(means) = c("panels","times");
    panelTimes = data.frame(unique(as.numeric(as.character(tempdata$b))),means$times);
    colnames(panelTimes) = c("panels","times");
    minPanel[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$panels;
}
minPanels = data.frame(sizes,minPanel);


for (i in unique(mn_data$m)) {         # for each non-naive implementation  
    for (j in unique(mn_data$nodes)) {        # node count,
        for (k in unique(mn_data$b)) {        # panel size,
            tempdata = mn_data[mn_data$m==i &
                               mn_data$nodes==j &
                               mn_data$b==k,];
            tempname = paste("mn-",as.character(i),"-",as.character(j),"nodes",
                             "-",as.character(k),"panel",sep="");

            p <- ggplot(tempdata, aes(x=ppn, y=logtpt, group=imp));
            p + geom_line(aes(colour=imp)) + opts(title=tempname);
            ggsave(paste(results_folder,tempname,".pdf",sep=""),scale=1);
        }
    }
}

# 3(cont). plot multiple nodes in 'naive' case. doing this because 
# naive case does not vary in threads like the others do. iterate over
# panel block size and nodes
for (k in unique(mn_data$b)) {
        tempdata = mn_data[mn_data$imp=='naive' & mn_data$b==k,];
        tempname = paste("mn-naive-",as.character(k),"panel",sep="");
        p <- ggplot(tempdata, aes(x=nodes,y=logtpt,group=m));
        p + geom_line(aes(colour=m)) + opts(title=tempname);
        ggsave(paste(results_folder,tempname,".pdf",sep=""),scale=1);
}

# clean up
rm(results_folder,tempdata,tempname);

###### ANALYSIS ######################################
# 1. Print configuration with lowest tpt for each size matrix

for (i in unique(ssn_data$m)) {
    tempdata = ssn_data[ssn_data$m==i,];
    print(tempdata[tempdata$tpt==min(tempdata$tpt),])
}
for (i in unique(mm_sn_data$m)) {
        tempdata = mm_sn_data[mm_sn_data$m==i,];
        print(tempdata[tempdata$tpt==min(tempdata$tpt),])
}  
for (i in unique(mn_data$m)) {
        tempdata = mn_data[mn_data$m==i,];
        print(tempdata[tempdata$tpt==min(tempdata$tpt),])
}  

# 2. 
