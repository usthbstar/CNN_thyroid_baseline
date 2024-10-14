%%%%%%%%%%%%%%%%%% CNN algorithm
%%%%%%%%%%%%%%%%%% For Thyroid cancer

clc
clear all
close all

%Load benigns and maligns data 
datapath='C:\Users\PC2321\Desktop\originaldata'
imds = imageDatastore(datapath,'IncludeSubfolders',true,'LabelSource','foldernames');

%Split data
[imdsTrain, imdsValidation]=splitEachLabel(imds,0.8, 'randomize'); 

%Convolutional neural network architecture CNN
layers=[
       imageInputLayer([224 224 3])
    convolution2dLayer(3, 16)
    tanhLayer()
    
    convolution2dLayer(1, 32)
    tanhLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32)
    tanhLayer()
    
    convolution2dLayer(1, 64)
    tanhLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64)
    tanhLayer()
    
    convolution2dLayer(1, 128)
    tanhLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 128)
    tanhLayer()
    
    convolution2dLayer(1, 256)
    batchNormalizationLayer
    tanhLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 256, 'Padding', 'same')
    tanhLayer()
    
    convolution2dLayer(1, 512)
    batchNormalizationLayer
    tanhLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 512, 'Padding', 'same')
    tanhLayer()
    
    convolution2dLayer(1, 1024)
    batchNormalizationLayer
    tanhLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 1024, 'Padding', 'same')
    tanhLayer()
    
    fullyConnectedLayer(4096)
    batchNormalizationLayer
    tanhLayer
    fullyConnectedLayer(512)
    batchNormalizationLayer
    tanhLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
          
%adam or sgdm
Options=trainingOptions('sgdm',...   
    'MiniBatchSize',32, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',1, ...
    'Verbose',true, ...
    'Plots','training-progress');
tic
%Train netWorks
[net netinfo]=trainNetwork(imdsTrain,layers,Options);


analyzeNetwork(net)            %AnalyzeNetwork
lgraph = layerGraph(net);      %Convert the Network into a Layer Graph
plot(lgraph) %plot(lgraph)
saveas(gcf, 'networkPlot.png') %Customize and Save the Plot

%Classification 
[YPred, probs]=classify(net,imdsValidation);

toc

%Confusion matrix
[cmat,classNames] = confusionmat(imdsValidation.Labels, YPred)             %gives matrix values
cm = confusionchart(cmat,classNames);                                      %for plot confusion matrix

%Parameters
% acc, sn, sp metrics
numClasses = numel(categories(imdsTrain.Labels))
[ACC, SEN, SEP, F1, PR]= ConfusionMatMultiClass1(cmat,numClasses)

%Plot ROC curve 
targets=grp2idx(imdsValidation.Labels);
[X,Y,Threshold,AUCpr] = perfcurve(targets, probs(:,1), 1, 'xCrit', 'fpr', 'yCrit', 'tpr');
figure(1),plot(X,Y)
xlabel('Specificity'); ylabel('Sensitivity');
grid on

%Plot Loss function 
figure(2),
plot(netinfo.TrainingLoss,'b-')
hold on
x = 1 : length(netinfo.ValidationLoss);
y = netinfo.ValidationLoss;
idx = ~any(isnan(y),1);
plot(x(idx),y(idx),'--k','Marker','.','MarkerSize' ,12);
xlabel("Iteration")
ylabel("Loss")
grid on

%Plot Accuracy
figure(3),
plot(netinfo.TrainingAccuracy,'b-')
hold on
x = 1 : length(netinfo.ValidationAccuracy);
y = netinfo.ValidationAccuracy;
idx = ~any(isnan(y),1);
plot(x(idx),y(idx),'--k','Marker','.','MarkerSize' ,12);
xlabel("Iteration")
ylabel("Accuracy")
grid on

trainingLoss = netinfo.TrainingLoss
validationLoss = netinfo.ValidationLoss
trainingAccuracy = netinfo.TrainingAccuracy
validationAccuracy = netinfo.ValidationAccuracy


