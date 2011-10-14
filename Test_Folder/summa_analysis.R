rm(list=ls())

#install.packages('gridExtra')
#install.packages('ggplot2')
#install.packages('reshape2')
library('gridExtra')
library('ggplot2')
library('reshape2')

system("cd /home/jack/Dropbox/source/cse6230/proj1");

###### DO DATA STUFF ##############################
data = read.table("output/summa_output.csv",sep=";");
colnames(data) <- c("m","n","k","px","py","panel_size","var","time");
castdata = ddply(data, c("m","n","k","px","py","panel_size","var"), function(df)
     data.frame(meantime=mean(df$time)));
m=castdata$m;
n=castdata$n;
k=castdata$k;
px=castdata$px;
py=castdata$py;
panel_size=castdata$panel_size;
var=castdata$var;
meantime=castdata$meantime;

###### PLOTS ################

ttimedata = castdata[var == "total_time",];
Atimedata = castdata[var == "bcast_time_A",];
Btimedata = castdata[var == "bcast_time_B",];
mtimedata = castdata[var == "local_mm_time",];

# increasing panel block size
p1 <- qplot(x=panel_size, y=meantime, data=ttimedata, main="Total time"); #increasing panel size
p2 <- qplot(x=panel_size, y=meantime, data=Atimedata, main="Bcast A");
p3 <- qplot(x=panel_size, y=meantime, data=Btimedata, main="Bcast B");
p4 <- qplot(x=panel_size, y=meantime, data=mtimedata, main="local_mm");
grid.arrange(p1,p2,p3,p4);

# diff grid configurations
X11();
p5 <- qplot(x=px, y=meantime, data=ttimedata, main="Total time"); #grid configurations
p6 <- qplot(x=px, y=meantime, data=Atimedata, main="Bcast A");
p7 <- qplot(x=px, y=meantime, data=Btimedata, main="Bcast B");
p8 <- qplot(x=px, y=meantime, data=mtimedata, main="local_mm");
grid.arrange(p5,p6,p7,p8);

###### SOME ANALYSIS ###########
ttimedata[ttimedata$meantime == min(ttimedata$meantime),]
Atimedata[Atimedata$meantime == min(Atimedata$meantime),]
Btimedata[Btimedata$meantime == min(Btimedata$meantime),]
mtimedata[mtimedata$meantime == min(mtimedata$meantime),]
