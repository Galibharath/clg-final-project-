
clc;
clear;
close all;
close all hidden;
warning off;

%% Choice Whether Image or Video

msg = "Choose Image or Video";
opts = ["Image" "video"];
choice = menu(msg,opts);

if choice == 1

% Training 
matlabroot = cd;    % Dataset path
datasetpath = fullfile(matlabroot,'Image_Dataset');
imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');
    
[imdsTrain, imdsValidation] = splitEachLabel(imds,0.7);
    
augimdsTrain = augmentedImageDatastore([300 300],imdsTrain);
augimdsValidation = augmentedImageDatastore([300 300],imdsValidation);
    
layers = [
        imageInputLayer([300 300 3])
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(9)
        softmaxLayer
        classificationLayer];
    
options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.0001, ...
        'MaxEpochs',20, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',10, ...
        'Verbose',false, ...
        'Plots','training-progress');
 
% [imageNet, imageTraininfo] = trainNetwork(augimdsTrain,layers,options);
   
 load imageNet
load imageTraininfo

% Testing
[filename,pathname] = uigetfile('*.*','Select the image'); %Open file selection dialog box
image=imread([pathname,filename]); %Read image from graphics file
image = imresize(image,[300 300]);
figure
imshow(image)
title('Input Image');

YPred = classify(imageNet,image);
msgbox(char(YPred));

% performance metrices 
predictedLabels = classify(imageNet, augimdsTrain);
testLabels = imdsTrain.Labels;
figure
C = confusionchart(testLabels, predictedLabels);

% Calculate accuracy
accuracy = sum(testLabels == predictedLabels) / numel(testLabels);
fprintf('The classified Accuracy is: %f\n',accuracy*100)

else

% Use uigetfile to open a file dialog in the current folder
[filename, filepath] = uigetfile({'*.mp4';'*.avi';'*.mov';'*.mkv';'*.wmv';'*.flv';'*.gif'}, 'Select a video file');
% For example, you can read the video using VideoReader:
videoObj = VideoReader(fullfile(filepath, filename));  
% Create a VideoPlayer object to display the video
videoPlayer = vision.VideoPlayer;
while hasFrame(videoObj)
    frame = readFrame(videoObj);
    step(videoPlayer, frame);
end
% Get the number of frames in the video
numFrames = videoObj.NumFrames;

% Read and save the last frame as an image
lastFrame = read(videoObj, numFrames);
lastFrame = imresize(lastFrame,[400 400]);
imshow(lastFrame);
title('Last Frame');

matlabroot = cd;    % Dataset path
datasetpath = fullfile(matlabroot,'Frameextracted_Dataset');
imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds,0.7);

augimdsTrain = augmentedImageDatastore([400 400 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([400 400 3],imdsValidation);
    
layers = [
        imageInputLayer([400 400 3])
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(9)
        softmaxLayer
        classificationLayer];
    
options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.0001, ...
        'MaxEpochs',20, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',10, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
% [videoNet, videoTraininfo] = trainNetwork(augimdsTrain,layers,options);

load videoNet
load videoTraininfo

[YPred,score1] = classify(videoNet,lastFrame);
msgbox(char(YPred))
accuracy = mean(videoTraininfo.TrainingAccuracy);
fprintf('Accuracy of classified Model is: %0.4f\n',accuracy);

end