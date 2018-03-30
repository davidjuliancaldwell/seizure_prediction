
function [extractInds,extractLocs] = splitMontage_seizure(DATA_DIR,sid,interestNames,interestElecs)

load(fullfile(DATA_DIR,sid,'trodes.mat'))

nElems = length(interestNames);
j = 1;
extractLocs = [];
extractInds = [];

for i = 1:nElems
    
    typeElec = interestNames{i};
    elecNums = interestElecs{i};
    
    for elec = elecNums
 
        tempName = eval(typeElec);
        [row,col] = find(AllTrodes == tempName(elec,:));
        rowCommon = mode(row);
        
        if j == 1
            extractLocs = AllTrodes(rowCommon,:);
            extractInds = rowCommon;
            j = 0;
            
        else
            extractLocs = [extractLocs; AllTrodes(rowCommon,:)];
            extractInds = [extractInds; rowCommon];
            
        end
    end
    
    
end



end