clc
clear
load('data.mat')
i =1;
for intRow = 1:7:length(demo)
    disp(['detector: '+demo{intRow,3}+' descriptor: '+demo{intRow,7}])
    stringVec(i)=['detector: '+demo{intRow,3}+' descriptor: '+demo{intRow,7}];
    cameraTTCmean(i)=demo{intRow+5,4};
    cameraTTCmin(i)=demo{intRow+5,6};
    cameraTTCmax(i)=demo{intRow+5,8};
    lidarTTCmean(i)=demo{intRow+6,4};
    lidarTTCmin(i)=demo{intRow+6,6};
    lidarTTCmax(i)=demo{intRow+5,8};
    i=i+1;
end
yneg = cameraTTCmean-cameraTTCmin;
ypos = cameraTTCmax-cameraTTCmean;

x = 1:1:9;
errorbar(x,cameraTTCmean,yneg,ypos)
xlim([0 10])
title('Mean of TTC Camera')
set(gca, 'XTickLabel',stringVec, 'XTick',1:length(stringVec))
xtickangle(60)

figure()
yneg = lidarTTCmean-lidarTTCmin;
ypos = lidarTTCmax-lidarTTCmean;
x = 1:1:9;
errorbar(x,lidarTTCmean,yneg,ypos)
xlim([0 10])
title('Mean of TTC Lidar')
set(gca, 'XTickLabel',stringVec, 'XTick',1:length(stringVec))
xtickangle(60)