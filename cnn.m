emb = fastTextWordEmbedding;
filenameTrain = "TrkceTwit_1.csv";
textName = "Tweets";
labelName = "Duygu";
ttdsTrain = tabularTextDatastore(filenameTrain,'SelectedVariableNames',[textName labelName]);
ttdsTrain.ReadSize = 10;
preview(ttdsTrain)

labels = readLabels(ttdsTrain,labelName);
classNames = unique(labels);
numObservations = numel(labels);


sequenceLength = 14;
tdsTrain = transform(ttdsTrain, @(data) transformTextData(data,sequenceLength,emb,classNames))

preview(tdsTrain)




numFeatures = emb.Dimension;
inputSize = [1 sequenceLength numFeatures];
numFilters = 200;

ngramLengths = [2 3 4 5];
numBlocks = numel(ngramLengths);

numClasses = numel(classNames);


layer = imageInputLayer(inputSize,'Normalization','none','Name','input');
lgraph = layerGraph(layer);


for j = 1:numBlocks
    N = ngramLengths(j);
    
    block = [
        convolution2dLayer([1 N],numFilters,'Name',"conv"+N,'Padding','same')
        batchNormalizationLayer('Name',"bn"+N)
        reluLayer('Name',"relu"+N)
        dropoutLayer(0.2,'Name',"drop"+N)
        maxPooling2dLayer([1 sequenceLength],'Name',"max"+N)];
    
    lgraph = addLayers(lgraph,block);
    lgraph = connectLayers(lgraph,'input',"conv"+N);
end


figure
plot(lgraph)
title("Network Architecture")

layers = [
    depthConcatenationLayer(numBlocks,'Name','depth')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','soft')
    classificationLayer('Name','classification')];

lgraph = addLayers(lgraph,layers);

figure
plot(lgraph)
title("Network Architecture")


for j = 1:numBlocks
    N = ngramLengths(j);
    lgraph = connectLayers(lgraph,"max"+N,"depth/in"+j);
end

figure
plot(lgraph)
title("Network Architecture")

%  TRAIN

miniBatchSize = 128;
numIterationsPerEpoch = floor(numObservations/miniBatchSize);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(tdsTrain,lgraph,options);
analyzeNetwork(net);
predictedY = resubPredict(net);
[XTest,YTest] = digitTest4DArrayData;
YPredicted = classify(net,XTest);
plotconfusion(XTest,YTest)