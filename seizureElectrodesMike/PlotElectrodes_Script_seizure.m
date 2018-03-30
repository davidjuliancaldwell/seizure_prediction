% load montage
sid = input('what is the subject ID ? \n','s');
SUB_DIR = fullfile(myGetenv('subject_dir'));
load(fullfile(strcat(sid,'_electrodes.mat')));

%
% scatter plot of electrode locations - colored by number in order 

colors = distinguishable_colors(2);

% take labeling from plot dots direct
% plot cortex too
figure
PlotCortex(sid,'b')
hold on
ictal = scatter3(ictalExtractLocs(:,1),ictalExtractLocs(:,2),ictalExtractLocs(:,3),[100],colors(2,:),'filled');
soz = scatter3(sozExtractLocs(:,1),sozExtractLocs(:,2),sozExtractLocs(:,3),[100],colors(1,:),'filled');
legend([soz,ictal],{'ictal','seizure'})

