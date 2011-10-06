rm(list=ls())

library('gridExtra')
library('ggplot2')
library('reshape2')

#setwd("/home/jack/Dropbox/source/cse6230/project1");
setwd("/nethome/jchua3/project1/");
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
ssn_data$logtpt = log2(ssn_data$tpt_mil);
mm_sn_data$logtpt = log2(mm_sn_data$tpt_mil);
mn_data$logtpt = log2(mn_data$tpt_mil);

###### PLOTS #######################################################
# Note: by default, ggplot always clears the screen and uses the entire device
# Note: names specify implementation, number of nodes, and panel block size

# 1. plots for single node matrix multiply (MM_SN) ***********************

# find best naive implementation from above for each panel size
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
         p <- ggplot(tempdata, aes(x=tpp, y=logtpt, group=imp));
         p + geom_line(aes(colour=imp)) + opts(title=tempname);
         ggsave(paste(results_folder,tempname,".png",sep=""),scale=1);
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


# 3. plots for multiple nodes (MN) ****************************************
# max 8 threads per node
# only openMP/MKL uses threads

# get unique values from size and node columns
sizes = unique(mn_data$m);
nodes = unique(mn_data$nodes);

# figure out best blocking for naive implementations for each size and node count
ix = 0;
naive_b = c(); naive_m = c(); naive_n = c(); naive_t = c();
naive_mean = c(); naive_median = c();
naive_ppn = c(); naive_tpp = c(); naive_k=c();
for (i in sizes) {
   for (j in nodes) {
       ix=ix+1;
       tempdata = mn_data[mn_data$imp=='naive' & mn_data$m==i & mn_data$nodes==j,];
       #means = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
       mins = aggregate(tempdata$tpt,by=list(tempdata$b,tempdata$ppn,tempdata$tpp),FUN=min);
       colnames(mins) = c("panels","ppn","tpp","times");
      panelTimes = data.frame(as.numeric(as.character(mins$panels)),
                              mins$ppn,    
                              mins$tpp,
                              mins$times);
       colnames(panelTimes) = c("panels","ppn","tpp","times");
       naive_b[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$panels[1];
       naive_m[ix] = i;
       naive_n[ix] = j; #slight abuse of notation, here n is the node count =O
       naive_t[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$times[1];
       naive_mean[ix] = mean(panelTimes$times);
       naive_median[ix] = median(panelTimes$times);
       naive_k[ix] = tempdata[1,]$k;
       naive_tpp[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$tpp[1];
       naive_ppn[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$ppn[1];
   }
}
naive_minPanels = data.frame(naive_n,naive_ppn,naive_tpp,naive_m,naive_m,naive_k,naive_b,naive_t,naive_mean,naive_median);
table_min_naive_mn_panels = naive_minPanels;


# figure out best blocking for openmp implementation 
# for each size and node count
ix = 0;
openmp_b = c(); openmp_m = c(); openmp_n = c(); openmp_t = c();
openmp_mean = c(); openmp_median = c();
openmp_ppn = c(); openmp_tpp = c(); openmp_k=c();
for (i in sizes) {
   for (j in nodes) {
       ix=ix+1;
       tempdata = mn_data[mn_data$imp=='openmp' & mn_data$m==i & mn_data$nodes==j,];
       #means = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
       mins = aggregate(tempdata$tpt,by=list(tempdata$b,tempdata$ppn,tempdata$tpp),FUN=min);
       colnames(mins) = c("panels","ppn","tpp","times");
      panelTimes = data.frame(as.numeric(as.character(mins$panels)),
                              mins$ppn,  
                              mins$tpp,
                              mins$times);
       colnames(panelTimes) = c("panels","ppn","tpp","times");
       openmp_b[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$panels[1];
       openmp_m[ix] = i;
       openmp_n[ix] = j; #slight abuse of notation, here n is the node count =O
       openmp_t[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$times[1];
       openmp_mean[ix] = mean(panelTimes$times);
       openmp_median[ix] = median(panelTimes$times);
       openmp_k[ix] = tempdata[1,]$k;
       openmp_tpp[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$tpp[1];
       openmp_ppn[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$ppn[1];
   }
}
openmp_minPanels = data.frame(openmp_n,openmp_ppn,openmp_tpp,openmp_m,openmp_m,openmp_k,openmp_b,openmp_t,openmp_mean,openmp_median);
table_min_openmp_mn_panels = openmp_minPanels;

# figure out best blocking for mkl implementations 
# for each size and node count
ix = 0;
mkl_b = c(); mkl_m = c(); mkl_n = c(); mkl_t = c();
mkl_mean = c(); mkl_median = c();
mkl_ppn = c(); mkl_tpp = c(); mkl_k=c();
for (i in sizes) {
   for (j in nodes) {
       ix=ix+1;
       tempdata = mn_data[mn_data$imp=='mkl' & mn_data$m==i & mn_data$nodes==j,];
       #means = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
       mins = aggregate(tempdata$tpt,by=list(tempdata$b,tempdata$ppn,tempdata$tpp),FUN=min);
       colnames(mins) = c("panels","ppn","tpp","times");
      panelTimes = data.frame(as.numeric(as.character(mins$panels)),
                              mins$ppn,    
                              mins$tpp,
                              mins$times);
       colnames(panelTimes) = c("panels","ppn","tpp","times");
       mkl_b[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$panels[1];
       mkl_m[ix] = i;
       mkl_n[ix] = j; #slight abuse of notation, here n is the node count =O
       mkl_t[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$times[1];
       mkl_mean[ix] = mean(panelTimes$times);
       mkl_median[ix] = median(panelTimes$times);
       mkl_k[ix] = tempdata[1,]$k;
       mkl_tpp[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$tpp[1];
       mkl_ppn[ix] = panelTimes[panelTimes$times == min(panelTimes$times),]$ppn[1];
   }
}
mkl_minPanels = data.frame(mkl_n,mkl_ppn,mkl_tpp,mkl_m,mkl_m,mkl_k,mkl_b,mkl_t,mkl_mean,mkl_median);
table_min_mkl_mn_panels = mkl_minPanels;


# DO THE SAME FOR MAX
# figure out best blocking for naive implementation 
# for each size and node count
ix = 0;
naive_b = c(); naive_m = c(); naive_n = c(); naive_t = c();
naive_mean = c(); naive_median = c();
naive_ppn = c(); naive_tpp = c(); naive_k=c();
for (i in sizes) {
   for (j in nodes) {
       ix=ix+1;
       tempdata = mn_data[mn_data$imp=='naive' & mn_data$m==i & mn_data$nodes==j,];
       #means = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
       maxes = aggregate(tempdata$tpt,by=list(tempdata$b,tempdata$ppn,tempdata$tpp),FUN=max);
       colnames(maxes) = c("panels","ppn","tpp","times");
      panelTimes = data.frame(as.numeric(as.character(maxes$panels)),
                              maxes$ppn,    
                              maxes$tpp,
                              maxes$times);
       colnames(panelTimes) = c("panels","ppn","tpp","times");
       naive_b[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$panels[1];
       naive_m[ix] = i;
       naive_n[ix] = j; #slight abuse of notation, here n is the node count =O
       naive_t[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$times[1];
       naive_mean[ix] = mean(panelTimes$times);
       naive_median[ix] = median(panelTimes$times);
       naive_k[ix] = tempdata[1,]$k;
       naive_tpp[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$tpp[1];
       naive_ppn[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$ppn[1];
   }
}
naive_maxPanels = data.frame(naive_n,naive_ppn,naive_tpp,naive_m,naive_m,naive_k,naive_b,naive_t,naive_mean,naive_median);
table_max_naive_mn_panels = naive_maxPanels;

ix = 0;
openmp_b = c(); openmp_m = c(); openmp_n = c(); openmp_t = c();
openmp_mean = c(); openmp_median = c();
openmp_ppn = c(); openmp_tpp = c(); openmp_k=c();
for (i in sizes) {
   for (j in nodes) {
       ix=ix+1;
       tempdata = mn_data[mn_data$imp=='openmp' & mn_data$m==i & mn_data$nodes==j,];
       #means = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
       maxes = aggregate(tempdata$tpt,by=list(tempdata$b,tempdata$ppn,tempdata$tpp),FUN=max);
       colnames(maxes) = c("panels","ppn","tpp","times");
      panelTimes = data.frame(as.numeric(as.character(maxes$panels)),
                              maxes$ppn,    
                              maxes$tpp,
                              maxes$times);
       colnames(panelTimes) = c("panels","ppn","tpp","times");
       openmp_b[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$panels[1];
       openmp_m[ix] = i;
       openmp_n[ix] = j; #slight abuse of notation, here n is the node count =O
       openmp_t[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$times[1];
       openmp_mean[ix] = mean(panelTimes$times);
       openmp_median[ix] = median(panelTimes$times);
       openmp_k[ix] = tempdata[1,]$k;
       openmp_tpp[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$tpp[1];
       openmp_ppn[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$ppn[1];
   }
}
openmp_maxPanels = data.frame(openmp_n,openmp_ppn,openmp_tpp,openmp_m,openmp_m,openmp_k,openmp_b,openmp_t,openmp_mean,openmp_median);
table_max_openmp_mn_panels = openmp_maxPanels;

ix = 0;
mkl_b = c(); mkl_m = c(); mkl_n = c(); mkl_t = c();
mkl_mean = c(); mkl_median = c();
mkl_ppn = c(); mkl_tpp = c(); mkl_k=c();
for (i in sizes) {
   for (j in nodes) {
       ix=ix+1;
       tempdata = mn_data[mn_data$imp=='mkl' & mn_data$m==i & mn_data$nodes==j,];
       #means = aggregate(tempdata$tpt,by=list(tempdata$b),FUN=mean);
       maxes = aggregate(tempdata$tpt,by=list(tempdata$b,tempdata$ppn,tempdata$tpp),FUN=max);
       colnames(maxes) = c("panels","ppn","tpp","times");
      panelTimes = data.frame(as.numeric(as.character(maxes$panels)),
                              maxes$ppn,    
                              maxes$tpp,
                              maxes$times);
       colnames(panelTimes) = c("panels","ppn","tpp","times");
       mkl_b[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$panels[1];
       mkl_m[ix] = i;
       mkl_n[ix] = j; #slight abuse of notation, here n is the node count =O
       mkl_t[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$times[1];
       mkl_mean[ix] = mean(panelTimes$times);
       mkl_median[ix] = median(panelTimes$times);
       mkl_k[ix] = tempdata[1,]$k;
       mkl_tpp[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$tpp[1];
       mkl_ppn[ix] = panelTimes[panelTimes$times == max(panelTimes$times),]$ppn[1];
   }
}
mkl_maxPanels = data.frame(mkl_n,mkl_ppn,mkl_tpp,mkl_m,mkl_m,mkl_k,mkl_b,mkl_t,mkl_mean,mkl_median);
table_max_mkl_mn_panels = mkl_maxPanels;

# plotting
for (i in unique(mn_data$m)) {                # for each implementation
#    tempdata = mn_data[mn_data$m==i,];
    tempdata <- NULL;
    tempname = paste("mn-",as.character(i),sep="");
            
    # find corresponding naive implementation for each node
    for (j in unique(mn_data$nodes)) {
     naive_panel = naive_minPanels[naive_minPanels$naive_m==i & 
                                   naive_minPanels$naive_n==j,]$naive_b;
     openmp_panel = openmp_minPanels[openmp_minPanels$openmp_m==i & 
                                     openmp_minPanels$openmp_n==j,]$openmp_b;
     mkl_panel = mkl_minPanels[mkl_minPanels$mkl_m==i & 
                               mkl_minPanels$mkl_n==j,]$mkl_b;
    
    # get subset of data that all nodes work with
     naivedata = mn_data[mn_data$imp=='naive' & mn_data$m==i & mn_data$b==naive_panel,];
     openmpdata = mn_data[mn_data$imp=='openmp' & mn_data$m==i & mn_data$b==openmp_panel,];
     mkldata = mn_data[mn_data$imp=='mkl' & mn_data$m==i & mn_data$b==mkl_panel,];

    # within each data subset, try to find the minimum time for each node and 
    # append it to tempdata
     naive_sub = naivedata[naivedata$nodes==j,];
     naive_min_node_j = naive_sub[naive_sub$tpt==min(naive_sub$tpt),];

     openmp_sub = openmpdata[openmpdata$nodes==j,];
     openmp_min_node_j = openmp_sub[openmp_sub$tpt==min(openmp_sub$tpt),];

     mkl_sub = mkldata[mkldata$nodes==j,];
     mkl_min_node_j = mkl_sub[mkl_sub$tpt==min(mkl_sub$tpt),];
                    
     tempdata = rbind(tempdata,naive_min_node_j,openmp_min_node_j,mkl_min_node_j);
    }

    # do the plotting
    p <- ggplot(tempdata, aes(x=nodes, y=logtpt, group=imp));
    p + geom_line(aes(colour=imp)) + opts(title=tempname);
    ggsave(paste(results_folder,tempname,".png",sep=""),scale=1);
}

# clean up
rm(results_folder,tempdata,tempname);

###### ANALYSIS #####################################################
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

# 2. Print tables
print(table_min_naive_ssn_panels)
print(table_max_naive_ssn_panels)
print(table_min_openmp_mm_sn_panels)
print(table_max_openmp_mm_sn_panels)
print(table_min_mkl_mm_sn_panels)
print(table_max_mkl_mm_sn_panels)
print(table_min_naive_mn_panels)
print(table_min_openmp_mn_panels)
print(table_min_mkl_mn_panels)
print(table_max_naive_mn_panels)
print(table_max_openmp_mn_panels)
print(table_max_mkl_mn_panels)
 
