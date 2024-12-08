function [permsOutput] = permtest(parameters,testPopSource,testPopTarget,config)

miniBatchSize = config.miniBatchSize;
numEpochs = 1;

numData = size(testPopSource,1);
numDimVariable = size(testPopSource,2);

sequencesSource = cell(numData, 1);

for k = 1:numData
    sequencesSource{k} = testPopSource(k,:);
end

sequencesDecoderInput = cell(numData, 1);  

for k = 1:numData
    sequencesDecoderInput{k} = (numDimVariable+1)*ones(1, numDimVariable+1);;

end


sequencesTarget = cell(numData, 1);
for k = 1:numData

    sequencesTarget{k} = [numDimVariable+1 testPopTarget(k,:)];
end


sequencesSourceDs = arrayDatastore(sequencesSource,OutputType="same");
sequencesTargetDs = arrayDatastore(sequencesTarget,OutputType="same");
sequencesDecoderInputDs = arrayDatastore(sequencesDecoderInput,OutputType="same");
sequencesDs = combine(sequencesSourceDs,sequencesDecoderInputDs,sequencesTargetDs);

inputSize = numDimVariable + 1;


numMiniBatchOutputs = 4; 

mbq = minibatchqueue(sequencesDs,numMiniBatchOutputs,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@(x,idc,testt) preprocessMiniBatchTestAddT(x,idc,testt,inputSize));


trailingAvg = [];
trailingAvgSq = [];
numObservationsTrain = numel(sequencesSource);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

epoch = 0;
iteration = 0;

sequenceLengthTarget = numDimVariable + 1;
codedPtrOutput = [];

% Loop over epochs.
while epoch < numEpochs

    epoch = epoch + 1;

    reset(mbq);

    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        [X,initDecoderInput,testT,sequenceLengthsSource] = next(mbq);
  
        [Z,hiddenState] = modelEncoder(parameters.encoder,X,sequenceLengthsSource);

        doTeacherForcing = config.testDoTeacherForcing;

        dropout = 0;

        originInput = X;

        [totalAttentionScores,miniPtrOutput] = decoderPredictions(parameters.decoder,originInput,Z,testT,hiddenState,dropout, ...
            doTeacherForcing,sequenceLengthTarget);

        codedPtrOutput = [codedPtrOutput; miniPtrOutput];
    end

end

permsOutput = codedPtrOutput;
end


function [X,initDecoderInput,testT,sequenceLengthsSource] = preprocessMiniBatchTestAddT(sequencesSource,sequencesDecoderInput,sequencesTarget,inputSize)

sequenceLengthsSource = cellfun(@(x) size(x,2),sequencesSource);

X = padsequences(sequencesSource,2,PaddingValue=inputSize);
X = permute(X,[1 3 2]);

initDecoderInput = padsequences(sequencesDecoderInput,2,PaddingValue=inputSize);
initDecoderInput = permute(initDecoderInput,[1 3 2]);

testT = padsequences(sequencesTarget,2,PaddingValue=inputSize);
testT = permute(testT,[1 3 2]);
end



function [X,initDecoderInput,sequenceLengthsSource] = preprocessMiniBatchTest(sequencesSource,sequencesDecoderInput,inputSize)

sequenceLengthsSource = cellfun(@(x) size(x,2),sequencesSource);

X = padsequences(sequencesSource,2,PaddingValue=inputSize);
X = permute(X,[1 3 2]);

initDecoderInput = padsequences(sequencesDecoderInput,2,PaddingValue=inputSize);
initDecoderInput = permute(initDecoderInput,[1 3 2]);

end



