fid = fopen('output/summa_output.csv');
temp = textscan(fid,'%f %f %f %f %f %f %s %f','Delimiter',',','CollectOutput',1);
fclose(fid);

% create data cols %%%%%%%%%%%%%%%%%%%%%%%%
M = length(temp.data);
data = zeros(M,8);

data(:,1:7) = temp.textdata;
data(:,8)   = temp.data;
