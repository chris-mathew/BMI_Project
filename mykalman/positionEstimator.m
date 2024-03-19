function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    newModelParameters = modelParameters;
    
    % SVM Prediction of Direction
    t_length = 320;
    X_test = mean(test_data.spikes(:, 1:t_length), 2)';
    
    % Get the confidence scores from each SVM
    svm_confidences = zeros(1, length(modelParameters.svmModel));
    for numSvm = 1:length(modelParameters.svmModel)
        svm_confidences(numSvm) = SVMPred(modelParameters.svmModel{numSvm}, X_test);
    end

    % Use the confidence scores to predict the class
    y_pred = determineClass(svm_confidences);
    newModelParameters.direction = y_pred;
    
    % Kalman Filtering for (x,y) Prediction
    predicted_angle = y_pred;
    kfParams = modelParameters.kalmanParams(predicted_angle);

    % Ensure currentState and covariance are initialized correctly
    if isfield(modelParameters, 'decodedHandPos') && ~isempty(modelParameters.decodedHandPos)
        currentState = modelParameters.decodedHandPos;
        currentCovariance = modelParameters.currentCovariance;
    else
        % Initialize state and covariance
        if size(test_data.startHandPos,1) >= 2
            currentState = [test_data.startHandPos(1:2,1); 0; 0];
            currentCovariance = eye(4); % Initial covariance, can be adjusted based on model confidence
        else
            error('Insufficient startHandPos data to initialize currentState');
        end
    end

    % Predict next state and covariance
    [predictedState, predictedCovariance] = kalmanPredict(currentState, currentCovariance, kfParams);

    % Update step with observation (if new spike data is available)
    if size(test_data.spikes, 2) > 320
        % Assuming spikes from selected_neurons and converting to firing rate or any other observable form
        observedSpikes = test_data.spikes(:, end-19:end); % Last 20 ms
        observation = mean(observedSpikes, 2); % Example observation model, can be more complex

        % Update the prediction based on the new observation
        [updatedState, updatedCovariance] = kalmanUpdate(predictedState, predictedCovariance, observation, kfParams);
    else
        updatedState = predictedState;
        updatedCovariance = predictedCovariance;
    end

    % Update positions and model parameters for next iteration
    x = updatedState(1);
    y = updatedState(2);
    newModelParameters.decodedHandPos = updatedState;
    newModelParameters.currentCovariance = updatedCovariance;
end

function [updatedState, updatedCovariance] = kalmanUpdate(predictedState, predictedCovariance, observation, kfParams)
    H = kfParams.H; % Observation matrix
    R = kfParams.R; % Observation noise covariance matrix
    
%     size(predictedCovariance)
%     size(H')
%     size(R)
%     size(H)

    K = predictedCovariance * H' / (H * predictedCovariance * H' + R); % Kalman gain
%     size(H)
%     size(predictedState)
%     size(observation)
%     size(K)
%     size(predictedState)

    updatedState = predictedState + K * (observation - H * predictedState);
    updatedCovariance = (eye(size(K, 1)) - K * H) * predictedCovariance;

    % Return the updated state and covariance
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

function [predictedState, predictedCovariance] = kalmanPredict(currentState, currentCovariance, kfParams)
    A = kfParams.A; % State transition matrix
    Q = kfParams.Q; % Process noise covariance matrix

    predictedState = A * currentState;
    predictedCovariance = A * currentCovariance * A' + Q;

    % Return the predicted state and covariance
end


function y_pred = determineClass(svm_confidences)
    % Find the class with the highest confidence
    [maxConfidence, y_pred] = max(svm_confidences);

    % Handle the case where multiple classes have the same highest confidence
    if sum(svm_confidences == maxConfidence) > 1
        candidates = find(svm_confidences == maxConfidence);
        y_pred = candidates(randi(length(candidates)));
    end
end
