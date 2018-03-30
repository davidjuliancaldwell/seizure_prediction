% a07793
% SOZ: LTP2/3, LAT 2/3, LAF 3,13
% Interictal: LAT 2/3, LPT 2-4, LAF 6, 13, 10-12 G: 42, 50, 51, 21, 53, 58, 59, 60
%  
% 702d24
% SOZ: LMT 2-4 (subclinical epileptic events first week then grid changed)
% LLT 9, LTP 1,2, LOT 1,2, then LAT1 (not on initial grid, probably don’t exist)	 
% Interictal: LMT 2-4, LTP 1, LAT 1-4 (during first week, on grid)
%  
% 9ab7ab
% SOZ: LMT1, LAT1
% Interictal: LAT 1-2, LMT 1-2, LTP 1-2, LPT 7
%  
% 69da36	 
% SOZ: G 34, 35, 50, 51
% Interictal: LTP1, LMT1, G 34, 35, 44, 45, 54, 55, 63, 64

%% DJC script to map the clinical electrodes to the seizure ones
DATA_DIR = 'C:\Users\djcald.CSENETID\SharedCode\seizurePrediction\seizureElectrodesMike';
SIDS = {'a07793','702d24','9ab7ab','69da36'};
%SIDS = {'a07793','9ab7ab','69da36'};

%sid = 'a07793';
for sid = SIDS
    sid = sid{:};
    switch sid
        case 'a07793'
            sozNames = {'LTT','LAT','LAF'}; % LTP = LTT?
            sozElecs = {[2,3],[2,3],[3,13]};
            
            ictalNames = {'LAT','LPT','LAF','Grid'};
            ictalElecs = {[2,3],[2:4],[6,10,11,12,13],[21 42 50 51 53 58 59 60]};
        case '702d24'
            sozNames = {'MST'}; % LMT = MST ??
            sozElecs = {[2:4]};
            
            ictalNames = {'MST','TP','AST'}; % LTP = TP, LAT = AST ?
            ictalElecs = {[2:4],[1],[1:4]};
        case '9ab7ab'
            sozNames = {'MST','AST'}; % LMT = MST? LAT = AST?
            sozElecs = {[1],[1]};
            
            ictalNames = {'AST','MST','TP','PST'}; % LTP = TP?, PST = LPT ?
            ictalElecs = {[1,2],[1,2],[1,2],[7]};
        case '69da36'
            sozNames = {'Grid'};
            sozElecs = {[34 35 50 51]};
            
            ictalNames = {'Grid','TP','AMT'}; % LTP is TP? LMT is AMT ?
            ictalElecs = {[34 35 44 45 54 55 63 64],[1],[1]};
    end
    
    [sozExtractInds,sozExtractLocs] = splitMontage_seizure(DATA_DIR,sid,sozNames,sozElecs);
    [ictalExtractInds,ictalExtractLocs] = splitMontage_seizure(DATA_DIR,sid,ictalNames,ictalElecs);
    
    %
    saveIt = 1;
    plotIt = 1;
    %
    if saveIt
        saveName = strcat(sid,'_electrodes.mat');
        save(saveName,'sozExtractInds','sozExtractLocs','ictalExtractInds',...
            'ictalExtractLocs','sozNames','sozElecs','ictalNames','ictalElecs')
    end
end
%%
if plotIt
    
    figure
    n_elems = max(max(identifier));
    j = 0;
    colors = distinguishable_colors(n_elems);
    figure
    [h1,h2] = PlotCortex(sid,'b',[],0.5);
    hold on
    
    for i = 1:n_elems
        tempInd = identifier==i;
        tempInd = tempInd(:,1);
        scatter3(locs(tempInd,1),locs(tempInd,2),locs(tempInd,3),[100],colors(i,:),'filled')
    end
    
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    legend(interestNames)
    %legend({'','',interestNames{:}})
    
    
    %% compare to all electrodes
    
    n_elems = length(Montage.Montage);
    j = 0;
    colors = distinguishable_colors(n_elems);
    figure
    hold on
    
    %[h1,h2] = PlotCortex(sid,'r',[],0.5);
    
    
    for i = 1:n_elems
        
        combined_info = split(Montage.MontageTokenized{i},["(",")"]);
        name = combined_info{1};
        elecs = str2num(combined_info{2});
        total = length(elecs);
        
        sub_sel = Montage.MontageTrodes((elecs+j),:);
        locsS = sub_sel;
        j = j + total;
        
        h = scatter3(locsS(:,1),locsS(:,2),locsS(:,3),[100],colors(i,:),'filled');
        
        trodeLabels = [1:total];
        for chan = 1:total
            txt = num2str(trodeLabels(chan));
            t = text(locsS(chan,1),locsS(chan,2),locsS(chan,3),txt,'FontSize',10,'HorizontalAlignment','center','VerticalAlignment','middle');
            set(t,'clipping','on');
        end
    end
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    
    legend(Montage.MontageTokenized)
    %legend({'','',Montage.MontageTokenized{:}})
end

