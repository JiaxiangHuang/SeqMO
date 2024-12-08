function parameters = permtrain(trainPopSource,trainPopTarget,config,parameters,itrNumber) 


miniBatchSize = config.miniBatchSize;
numEpochs = config.numEpochs;  
learnRate = config.learnRate;
gradientDecayFactor = config.gradientDecayFactor;
squaredGradientDecayFactor = config.squaredGradientDecayFactor;
embeddingDimension = config.embeddingDimension;  
numHiddenUnits = config.numHiddenUnits;     
dropout = config.dropout;

numData = size(trainPopSource,1);
numDimVariable = size(trainPopSource,2);

sequencesSource = cell(numData, 1);
sequencesTarget = cell(numData, 1);

for k = 1:numData
    sequencesSource{k} = [trainPopSource(k,:)];

end

for k = 1:numData 
    sequencesTarget{k} = [numDimVariable+1 trainPopTarget(k,:)];
end

%% Preprocess Data
sequencesSourceDs = arrayDatastore(sequencesSource,OutputType="same");
sequencesTargetDs = arrayDatastore(sequencesTarget,OutputType="same");
sequencesDs = combine(sequencesSourceDs,sequencesTargetDs);

clear trainPopSource trainPopTarget sequencesTarget sequencesSourceDs sequencesTargetDs

if itrNumber == 1
%% Initialize Model Parameters

inputSize = numDimVariable + 1; 
sz = [embeddingDimension inputSize];
mu = 0;
sigma = 0.01;
parameters.encoder.emb.Weights = initializeGaussian(sz,mu,sigma);

% Initialize the learnable parameters for the encoder LSTM operation.
sz = [4*numHiddenUnits embeddingDimension];
numOut = 4*numHiddenUnits;
numIn = embeddingDimension;
parameters.encoder.lstm.InputWeights = initializeGlorot(sz,numOut,numIn);
parameters.encoder.lstm.RecurrentWeights = initializeOrthogonal([4*numHiddenUnits numHiddenUnits]);
parameters.encoder.lstm.Bias = initializeUnitForgetGate(numHiddenUnits);

% Initialize Decoder Model Parameters
outputSize = numDimVariable + 1;
sz = [embeddingDimension outputSize];
mu = 0;
sigma = 0.01;
parameters.decoder.emb.Weights = initializeGaussian(sz,mu,sigma);

% Initialize the weights of the attention mechanism using the Glorot 
% initializer using the initializeGlorot function.
sz = [numHiddenUnits numHiddenUnits];
numOut = numHiddenUnits;
numIn = numHiddenUnits;
parameters.decoder.attention.Weights = initializeGlorot(sz,numOut,numIn);


% Initialize the learnable parameters for the decoder LSTM operation:
sz = [4*numHiddenUnits embeddingDimension+numHiddenUnits];
numOut = 4*numHiddenUnits;
numIn = embeddingDimension + numHiddenUnits; 
                           
parameters.decoder.lstm.InputWeights = initializeGlorot(sz,numOut,numIn);
parameters.decoder.lstm.RecurrentWeights = initializeOrthogonal([4*numHiddenUnits numHiddenUnits]);
parameters.decoder.lstm.Bias = initializeUnitForgetGate(numHiddenUnits);

end

inputSize = numDimVariable + 1; 
outputSize = numDimVariable + 1;

%% Define Model Functions
% Define Model Loss Function
numMiniBatchOutputs = 4; 

mbq = minibatchqueue(sequencesDs,numMiniBatchOutputs,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@(x,t) preprocessMiniBatch(x,t,inputSize,outputSize));

% Initialize the values for the adamupdate function.
trailingAvg = [];
trailingAvgSq = [];
% Calculate the total number of iterations for the training progress monitor
numObservationsTrain = numel(sequencesSource);
clear sequencesSource
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

epoch = 0;
iteration = 0;

codedPtrOutput = []; 

% Loop over epochs.
while epoch < numEpochs 
    epoch = epoch + 1;

    reset(mbq);

    % Loop over mini-batches.
    while hasdata(mbq) 
        iteration = iteration + 1;

        [X,T,sequenceLengthsSource,maskSequenceTarget] = next(mbq);

        % Compute loss and gradients.
        [loss, gradients, miniPtrOutput] = dlfeval(@modelLoss, parameters, X, T, ...
            sequenceLengthsSource, maskSequenceTarget, dropout, config.probDoTeacherForcing);  % Change: Use dlfeval to call the function

        % Update parameters using adamupdate.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients,trailingAvg,trailingAvgSq,...
            iteration,learnRate,gradientDecayFactor,squaredGradientDecayFactor);
 
        % Save the pointer output at the last epoch
        if epoch == numEpochs - 1  
            codedPtrOutput = [codedPtrOutput; miniPtrOutput];
        end

    end

    if rem(epoch, 10) == 0
    fprintf('Train Epoch: %d finished, Loss: %f\n', epoch, loss);  
    end

end
end

%% initialize

function weights = initializeGaussian(sz,mu,sigma)
weights = randn(sz,'single')*sigma + mu;
weights = dlarray(weights);
end

function weights = initializeGlorot(sz,numOut,numIn)
Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));
weights = bound * Z;
weights = dlarray(weights);
end

function parameter = initializeOrthogonal(sz)
Z = randn(sz,'single');
[Q,R] = qr(Z,0);
D = diag(R);
Q = Q * diag(D ./ abs(D));
parameter = dlarray(Q);
end

function bias = initializeUnitForgetGate(numHiddenUnits)
bias = zeros(4*numHiddenUnits,1,'single');
idx = numHiddenUnits+1:2*numHiddenUnits;
bias(idx) = 1;
bias = dlarray(bias);
end

%% preprocessMiniBatch

function [X,T,sequenceLengthsSource,maskTarget] = preprocessMiniBatch(sequencesSource,sequencesTarget,inputSize,outputSize)
sequenceLengthsSource = cellfun(@(x) size(x,2),sequencesSource);
X = padsequences(sequencesSource,2,PaddingValue=inputSize);
X = permute(X,[1 3 2]);
[T,maskTarget] = padsequences(sequencesTarget,2,PaddingValue=outputSize);
T = permute(T,[1 3 2]);
maskTarget = permute(maskTarget,[1 3 2]);
end

%% modelLoss

function [loss,gradients,ptrOutput] = modelLoss(parameters,X,T,...
    sequenceLengthsSource,maskTarget,dropout,probDoTeacherForcing)
[Z,hiddenState] = modelEncoder(parameters.encoder,X,sequenceLengthsSource);

doTeacherForcing = rand < probDoTeacherForcing;

sequenceLengthTarget = size(T,3);

[totalAttentionScores,ptrOutput] = decoderPredictions(parameters.decoder,X,Z,T,hiddenState,dropout,...
    doTeacherForcing,sequenceLengthTarget);
clear Z hiddenState

T = extractdata(gather(T(:,:,2:end)));   

batchSize =  size(T,2);
sequenceLengthCut = size(T,3);

tempT = squeeze(T);
tempX = extractdata(gather(squeeze(X)));
[numPermutations, ~] = size(tempX);
clear T X

ind_X_T = zeros(size(tempT));

% Use indexing for each permutation
for i = 1:numPermutations
    % Create a map from the value to the index for the current permutation
    [~, idx] = sort(tempX(i, :));
    ind_X_T(i, :) = idx(tempT(i, :));
end

[B, I] = ndgrid(1:batchSize, 1:sequenceLengthCut);


target_index = ind_X_T; 

% Extract the required probabilities using linear indexing
linearIndices = sub2ind(size(totalAttentionScores), target_index, B, I);
target_probs = totalAttentionScores(linearIndices); %32*81

clear totalAttentionScores B I

% Computing the cross-entropy loss in a vectorized manner
loss1 = -mean(log(target_probs(:)));  
loss = dlarray(gpuArray(single(loss1)));
gradients = dlgradient(loss, parameters);

end


