function  [Offspring, idealpoint, parametersNetwork, ObjValsLow, mapObjValsHigh, ObjValsPredict] = seqmo(Offspring, ...
    Problem,neighborB,idealpoint,parametersNetwork,itrNumber,config)

IndivsLowAll = Offspring.decs';  
ObjValsLowAll =  Offspring.objs';  

[FrontNoOff,~,~] = NSGA2Front(ObjValsLowAll',Problem.N);
[~,arrangedPopIndexLowAll]=sort(FrontNoOff);  
arrangedPopIndexLowAll = arrangedPopIndexLowAll';

ll = ceil(length(arrangedPopIndexLowAll)./2);
indexLow = arrangedPopIndexLowAll(ll+1 : end);
IndivsLow = IndivsLowAll(:,indexLow);
ObjValsLow = ObjValsLowAll(:,indexLow);

indexHigh = arrangedPopIndexLowAll(1 : ll);
IndivsHigh = IndivsLowAll(:,indexHigh);
ObjValsHigh = ObjValsLowAll(:,indexHigh);


%% Mapping 

if config.mappingOne2one == 0
    % tic;
    tempIndexHigh = indexHigh;
    a_empty = [];
    a_not_empty = [];
    j = 1;
    for i = 1:length(indexLow)
        a = indexLow(i);
        aneib = neighborB(a,:)';
        [C, ia, ib] = intersect(aneib, tempIndexHigh);
        if isempty(C)
            a_empty = [a_empty;a];
        else
            s = randi(length(ib));
            mapIndexHigh(j,1) = tempIndexHigh(ib(s));

            a_not_empty = [a_not_empty;a];
            j = j + 1;
        end
    end

    fixIndexHigh = indexHigh; 
    wd = config.nich; 
    t = 1;
    a2_empty = [];
    while ~isempty(a_empty) && t < ceil(size(neighborB,1)/wd)
        if length(a_empty) <= 2
            randomIndices = randperm(length(fixIndexHigh), length(fixIndexHigh)-j+1);
            mapIndexHigh(j:length(fixIndexHigh),1) = fixIndexHigh(randomIndices);
            a_not_empty = [a_not_empty; a_empty];

            break; 
        end
        for k = 1:length(a_empty)
            a2 = a_empty(k);

            mappingFlag = 0;

            if (a2-(t+1)*wd < 1) && (mappingFlag ~= 1)
                a2nei = [1 : a2-t*wd]';
                mappingFlag = 1; 
                turnPointT = t;
                turnPointa2 = a2 + t*wd;

            elseif a2+(t+1)*wd > size(neighborB,1) && (mappingFlag ~= 2 )
                a2neib = [a2+t*wd : size(neighborB,1)]';
                mappingFlag = 2; 
                turnPointT = t;
                turnPointa2 = a2 - t*wd;

            else

                a2neib = [neighborB(a2 - t*wd,:) neighborB(a2 + t*wd,:)]';
            end

            if mappingFlag == 1

                ponitClose = turnPointa2 + (2*(t-turnPointT)-1)*wd;
                ponitFar = ponitClose + wd;
                ponitFarNext = turnPointa2 + (2*(t+1-turnPointT)-1)*wd + wd;

                if ponitFarNext > size(neighborB,1)
                    a2neib = [ponitFar : size(neighborB,1)]';
                else
                    a2neib = [neighborB(ponitClose, :) neighborB(ponitFar, :)]';
                end
            end

            if mappingFlag == 2

                ponitClose = turnPointa2 - (2*(t-turnPointT)-1)*wd;
                ponitFar = ponitClose - wd;
                ponitFarNext = turnPointa2 - (2*(t+1-turnPointT)-1)*wd- wd;

                if ponitFarNext < 1
                    a2neib = [1: ponitFar]';
                else
                    a2neib = [ neighborB(ponitFar, :) neighborB(ponitClose, :)]';
                end
            end

            [C2, ia2, ib2] = intersect(a2neib, fixIndexHigh);

            if isempty(C2)
                a2_empty = [a2_empty;a2];
            else
                s2 = randi(length(ib2));
                mapIndexHigh(j,1) = fixIndexHigh(ib2(s2));
                a_not_empty = [a_not_empty;a2];

                j = j + 1;
                a_empty(k) = -1;
            end
        end
        a_empty(find(a_empty==-1))=[];
        t = t + 1;
    end

end    

if config.mappingOne2one == 1
    tic;

    remainIndexHigh = indexHigh; 

    a_empty = [];  
    a_not_empty = [];
   
    j = 1;
    for i = 1:length(indexLow)
        a = indexLow(i);
        aneib = neighborB(a,:)';
        [C, ia, ib] = intersect(aneib, remainIndexHigh);
        if isempty(C)
            a_empty = [a_empty;a];
            
        else 
            s = randi(length(ib));
            mapIndexHigh(j,1) = remainIndexHigh(ib(s));
            remainIndexHigh(ib(s)) = []; 

            a_not_empty = [a_not_empty;a];
            j = j + 1;
        end
    end

    fprintf('a_not_empty length: %d\n',length(a_not_empty));


    fixIndexHigh = indexHigh; 
    wd = config.nich; 
    t = 1;
    a2_empty = [];
    while ~isempty(a_empty) && t < ceil(size(neighborB,1)/wd)
        if length(a_empty) <= wd
            mapIndexHigh(j:length(fixIndexHigh),1) = remainIndexHigh;
            a_not_empty = [a_not_empty; a_empty];

            break; 
        end
        for k = 1:length(a_empty)
            a2 = a_empty(k);

            mappingFlag = 0;

            if (a2-(t+1)*wd < 1) && (mappingFlag ~= 1)
                a2nei = [1 : a2-t*wd]';
                mappingFlag = 1; 
                turnPointT = t;
                turnPointa2 = a2 + t*wd;

            elseif a2+(t+1)*wd > size(neighborB,1) && (mappingFlag ~= 2 )
                a2neib = [a2+t*wd : size(neighborB,1)]';
                mappingFlag = 2;
                turnPointT = t;
                turnPointa2 = a2 - t*wd;

            else

                a2neib = [neighborB(a2 - t*wd,:) neighborB(a2 + t*wd,:)]';
            end

            if mappingFlag == 1

                ponitClose = turnPointa2 + (2*(t-turnPointT)-1)*wd;
                ponitFar = ponitClose + wd;
                ponitFarNext = turnPointa2 + (2*(t+1-turnPointT)-1)*wd + wd;

                if ponitFarNext > size(neighborB,1)
                    a2neib = [ponitFar : size(neighborB,1)]';
                else
                    a2neib = [neighborB(ponitClose, :) neighborB(ponitFar, :)]';
                end
            end

            if mappingFlag == 2

                ponitClose = turnPointa2 - (2*(t-turnPointT)-1)*wd;
                ponitFar = ponitClose - wd;
                ponitFarNext = turnPointa2 - (2*(t+1-turnPointT)-1)*wd- wd;

                if ponitFarNext < 1
                    a2neib = [1: ponitFar]';
                else
                    a2neib = [ neighborB(ponitFar, :) neighborB(ponitClose, :)]';
                end
            end

            [C2, ia2, ib2] = intersect(a2neib, remainIndexHigh);

            if isempty(C2)
                a2_empty = [a2_empty;a2];
            else
                s2 = randi(length(ib2));
                mapIndexHigh(j,1) = remainIndexHigh(ib2(s2));
                remainIndexHigh(ib2(s2)) = [];

                a_not_empty = [a_not_empty;a2];

                j = j + 1;
                a_empty(k) = -1;
            end
            if(length(find(a_empty~=-1))~=length(remainIndexHigh))
                fprintf('error! length of a_empty and remainIndexHigh is not equal.')
            end
        end
        a_empty(find(a_empty==-1))=[];
        t = t + 1;
    end
    if ~isempty(a_empty)
        fprintf('indLow remains values! Length of a_empty is: %d\n',length(a_empty))
    end
    toc;
end

aragIndexLow = a_not_empty;

mapIndivsHigh = IndivsLowAll(:, mapIndexHigh);
mapObjValsHigh = ObjValsLowAll(:, mapIndexHigh);

IndivsLow = IndivsLowAll(:,aragIndexLow);
ObjValsLow = ObjValsLowAll(:,aragIndexLow);

indexLow = aragIndexLow;

%% Training

trainPopSource = IndivsLow';
trainPopTarget = mapIndivsHigh';

parametersNetwork = permtrain(trainPopSource,trainPopTarget,config,parametersNetwork,itrNumber);   

%% Testing
testPopSource = IndivsLow';
testPopTarget = mapIndivsHigh';

[ptrOut] = permtest(parametersNetwork,testPopSource,testPopTarget,config); 
tempIndivsPredict= ptrOut';
IndivsPredict = double(gather(extractdata(tempIndivsPredict)));

%% update Offspring 

SolutionPredict = Problem.Evaluation(IndivsPredict');

Offspring(indexLow) = SolutionPredict;

ObjValsPredict= SolutionPredict.objs';


end