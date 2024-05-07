function accuracy = runSvmEvaluation(matFilePath)
    % Load the data from the .mat file
    load(matFilePath);

    % Prepare data
    [X, Y] = prepareSVMTrainingData(trial);

    % Randomly permute the data
    numSamples = size(X, 1);
    indices = randperm(numSamples);

    % Split indices for training and testing (e.g., 80-20 split)
    numTrain = floor(0.8 * numSamples);
    trainIndices = indices(1:numTrain);
    testIndices = indices(numTrain + 1:end);

    % Create training and testing datasets
    X_train = X(trainIndices, :);
    Y_train = Y(trainIndices);
    X_test = X(testIndices, :);
    Y_test = Y(testIndices);

    % Train the SVM models
    modelParameters = positionEstimatorTraining(trial);  % You might adjust the function to accept X_train, Y_train

    % Calculate SVM accuracy on the test data
    accuracy = calculateSvmAccuracy(modelParameters, X_test, Y_test);
    fprintf('The SVM classification accuracy is: %.2f%%\n', accuracy);
end

function accuracy = calculateSvmAccuracy(modelParameters, X_test, Y_test)
    % Initialize count of correct predictions
    correct_predictions = 0;
    
    % Loop through each test example
    for i = 1:size(X_test, 1)
        % Compute SVM confidences for each class
        svm_confidences = zeros(1, length(modelParameters.svmModel));
        for numSvm = 1:length(modelParameters.svmModel)
            svm_confidences(numSvm) = SVMPred(modelParameters.svmModel{numSvm}, X_test(i, :));
        end
        
        % Determine the predicted class based on the highest confidence
        [~, predicted_label] = max(svm_confidences);
        
        % Increment the correct_predictions count if the prediction is correct
        if predicted_label == Y_test(i)
            correct_predictions = correct_predictions + 1;
        end
    end
    
    % Calculate the accuracy as the ratio of correct predictions to the total number of test examples
    accuracy = (correct_predictions / size(X_test, 1)) * 100;
end

function confidence = SVMPred(model_param, X)
    % Compute the kernel between X and each support vector
    K = zeros(size(X, 1), size(model_param.X, 1));
    for i = 1:size(X, 1)
        for j = 1:size(model_param.X, 1)
            distanceSquared = sum((X(i, :) - model_param.X(j, :)).^2);
            K(i, j) = exp(-model_param.gamma * distanceSquared);
        end
    end
    % The confidence score is the decision function value
    confidence = (K * (model_param.alpha .* model_param.y)) + model_param.b;
end

function [X, Y] = prepareSVMTrainingData(training_data)
    % This function prepares the input feature matrix X and target vector Y for SVM training
    % Spike counts are averaged over a specified initial time window for each trial
    initialTimeWindow = 320; 
    numTrials = size(training_data, 1);
    numDirections = size(training_data, 2);
    
    % Pre-allocate
    X = []; % Feature matrix
    Y = []; % Target vector
    
    % Loop over trials and directions to extract features and labels
    for trial = 1:numTrials
        for direction = 1:numDirections
            spikes = training_data(trial, direction).spikes;
            avgSpikes = mean(spikes(:, 1:initialTimeWindow), 2)'; % Row vector of average spike counts
            X = [X; avgSpikes]; % Append to feature matrix
            Y = [Y; direction]; % Append direction as label
        end
    end

end