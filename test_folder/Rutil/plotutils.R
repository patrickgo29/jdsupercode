#install.packages("session")
library('session')

# clean_data(data,type)
#
# Assumes data is in the following form:
# 
# MM: alg, nodes, ppn, tpp, imp, m, n, k, tet, trials, tpt
# SUMMA: alg, nodes, ppn, tpp, imp, m, n, k, x, y, b, tet, trials, tpt 
#
# INPUT: data - any data frame with the above variables
#        type - "MM" - data file contains variables from local_mm algorithm
#               "SUMMA" - contains variables from SUMMA algorithm
# OUTPUT: resultant data frame
# USAGE: data <- clean_data(data,"SUMMA")
clean_data <- function(data,type) {
    
if (type == "SUMMA") 
{
    # Refactor data
    colnames(data) <- c("alg","nodes","ppn","tpp","imp","m","n","k",
                        "x","y","b","tet","trials","tpt");
    data <- transform(data, nodes=as.numeric(nodes), ppn=as.numeric(ppn),
                      tpp=as.numeric(tpp), m=as.numeric(m), n=as.numeric(n),
                      k=as.numeric(k), x=as.numeric(x), y=as.numeric(y),
                      b=as.numeric(b),tet=as.numeric(tet),
                      trials=as.numeric(trials), tpt=as.numeric(tpt));
}

if (type == "MM")
{
    # Refactor data
    colnames(data) <- c("alg","nodes","ppn","tpp","imp","m","n","k",
                        "tet","trials","tpt");
    data <- transform(data, nodes=as.numeric(nodes), ppn=as.numeric(ppn),
                      tpp=as.numeric(tpp), m=as.numeric(m), n=as.numeric(n),
                      k=as.numeric(k), tet=as.numeric(tet),
                      trials=as.numeric(trials), tpt=as.numeric(tpt));
}

# Transform time to milliseconds log scale
data$logtpt = log2(data$tpt*1000);

# Get rid of observations that are invalid
data_adj = NULL;
if (type == "SUMMA")
{
    for (i in 1:length(data[,1])) {
    
        # Initialize temp vars
        rowtemp = data[i,];
        m=rowtemp$m; n=rowtemp$n; k=rowtemp$k; x=rowtemp$x;
        y=rowtemp$y; pb=rowtemp$b;
        A_x=k/x; A_y=m/y; B_x=n/x; B_y=k/y;
    
        if ((pb>A_x | pb>A_y | pb>B_x | pb>B_y)) {
            next
        }
        if (A_x%%pb!=0 | A_y%%pb!=0 | B_x%%pb!=0 | B_y%%pb!=0) {
            next
        }
        if (m%%y!=0 | n%%x!=0 | k%%x!=0 | k%%y!=0) {
            next
        }

        # append row to data if conditionals don't pass
        data_adj = rbind(data_adj, rowtemp);
    }
    data = data_adj;
}

# Free vars and return
rm(m,n,k,x,y,pb,A_x,A_y,B_x,B_y)
return(data);

}

# find_stat(target,grouplist,fun)
#
# Given particular dataset, find a particular statistic of time per trial
# using grouping variables.
# 
# INPUT: data - data frame
#        group - vector of variable strings to uniquely identify observations with
#        fun - function used (max / min) 
#        other - vector of names for additional vars to add
# RETURN: resultant data frame with group stats. will also contain a column with
# indices from the target variable where the (min,max) was found, or the closest obs
# to the (mean,med)
#
# USAGE: impBMin <- find_stat(data,c("imp","b"),min,c("ppn","nodes"))
#
# NOTE: strings in target and group must correspond to actual variable names in data
find_stats <- function(data,group,fun,other) {
   
    # find tpt values within each group
    val = ddply(data,group,function(df)fun(df$tpt))
    colnames(val) = c(group,"tpt");

    # append other data about the max and min to the resultant dataset
    if ("alg" %in% other) {
        alg = ddply(data,group,function(df)df$alg[df$tpt == fun(df$tpt)]);
        val = merge(val,alg,by=group);
    }
    if ("nodes" %in% other) {
        nodes = ddply(data,group,function(df)df$nodes[df$tpt == fun(df$tpt)]);
        val = merge(val,nodes,by=group);
    }
    if ("ppn" %in% other) {
        ppn = ddply(data,group,function(df)df$ppn[df$tpt == fun(df$tpt)]);
        val = merge(val,ppn,by=group);
    }
    if ("tpp" %in% other) {
        tpp = ddply(data,group,function(df)df$tpp[df$tpt == fun(df$tpt)]);
        val = merge(val,tpp,by=group);
    }
    if ("imp" %in% other) {
        imp = ddply(data,group,function(df)df$imp[df$tpt == fun(df$tpt)]);
        val = merge(val,imp,by=group);
    }
    if ("m" %in% other) {
        m = ddply(data,group,function(df)df$m[df$tpt == fun(df$tpt)]);
        val = merge(val,m,by=group);
    }
    if ("n" %in% other) {
        n = ddply(data,group,function(df)df$n[df$tpt == fun(df$tpt)]);
        val = merge(val,n,by=group);
    }
    if ("k" %in% other) {
        k = ddply(data,group,function(df)df$k[df$tpt == fun(df$tpt)]);
    }
    if ("x" %in% other) {
        x = ddply(data,group,function(df)df$x[df$tpt == fun(df$tpt)]);
        val = merge(val,x,by=group);
    }
    if ("y" %in% other) {
        y = ddply(data,group,function(df)df$y[df$tpt == fun(df$tpt)]);
        val = merge(val,y,by=group);
    }
    if ("b" %in% other) {
        b = ddply(data,group,function(df)df$b[df$tpt == fun(df$tpt)]);
        val = merge(val,b,by=group);
    }
    if ("mean" %in% other) {
        meantpt = ddply(data,group,function(df)mean(df$tpt));
        val = merge(val,meantpt,by=group);
    }
    if ("median" %in% other) {
        mediantpt = ddply(data,group,function(df)median(df$tpt));
        val = merge(val,mediantpt,by=group);
    }
    # sort dataframe
    val = val[with(val, order(group)),]

    # rename cols, return
    colnames(val) = c(group,"tpt",other);
    return(val)
}

# dfsToCSV(df1,df2,...,filename)
#
# simple function to write multiple dataframes to a CSV file
#
# INPUTS: df1, ..., dfn - n dataframes you want to write to a file
#         filename - name of file to write to
dfsToCSV <- function(...,filename) {
    dfs <- list(...)

    # first clear the file
    command = paste("rm ",filename,sep="");
    system(command);

    # write them all to the same file
    for (i in 1:length(dfs)) {
        write.table(dfs[[i]], file=filename, append=TRUE, sep=",");
    }
}

# plotgroup(df)
#
# function to create multiple plots of a dataframe based on a grouping
# variable. we use the same naming convention as last time
#
# INPUTS: df - dataframe
#         group - vector of var strings to group plots by
#         xaxis - var string to plot on x axis
#         yaxis - var string to plot on y axis
plotgroup <- function(df,group,xaxis,yaxis) {
    
}
