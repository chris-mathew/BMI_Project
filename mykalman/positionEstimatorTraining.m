function modelParameters = positionEstimatorTraining(training_data)
     % Extract features for SVM training
    [X, Y] = prepareSVMTrainingData(training_data);
    kernelType = 'rbfKernel';
    numClasses = size(training_data, 2); % Assuming 8 directions/classes

    for numSvm = 1:numClasses
        % The current class is positive, and all other classes are negative
        Y_train = double(Y == numSvm);
        Y_train(Y_train == 0) = -1;

        % Train the SVM model for the current class
        modelParameters.svmModel{numSvm} = trainSVM(X, Y_train, kernelType);
    end

    
    % Initialize structure for Kalman filter parameters with consistent fields
    dummyKalmanParam = struct('A', [], 'H', [], 'Q', [], 'R', [], 'initialState', [], 'initialCovariance', []);
    modelParameters.kalmanParams = repmat(dummyKalmanParam, 1, 8); 

    % Train Kalman filter parameters for each direction
    for direction = 1:8 
        modelParameters.kalmanParams(direction) = trainKalmanParams(training_data, direction);
    end
end


function [X, Y] = prepareSVMTrainingData(training_data)
    % This function prepares the input feature matrix X and target vector Y for SVM training
    % Spike counts are averaged over a specified initial time window for each trial
    initialTimeWindow = 320; % ms, example time window to average spikes
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


function kalmanParams = trainKalmanParams(training_data, direction)
    % Initialize variables to accumulate data
    handPosAll = [];
    handVelAll = [];
    spikesAll = [];
    
    for trial = 1:size(training_data, 1)
        % Extract hand position and compute velocity
        handPos = training_data(trial, direction).handPos(1:2, :); % Considering only X, Y positions
        handVel = diff(handPos, 1, 2); % Simple difference to compute velocity
        
        % Padding handVel to make it the same length as handPos
        handVel = [handVel, handVel(:, end)];
        
        % Extract neural data
        spikes = training_data(trial, direction).spikes;
        
        % Accumulate data
        handPosAll = [handPosAll, handPos];
        handVelAll = [handVelAll, handVel];
        spikesAll = [spikesAll, spikes];
    end
    
    % Average spikes over a 20ms window to reduce dimensionality
    binSize = 20; % 20ms binning
    spikesBinned = binData(spikesAll, binSize);
    
    % Create state matrix (position and velocity) and observation matrix (spikes)
    stateMatrix = [handPosAll; handVelAll];
    observationMatrix = spikesBinned;
    stateMatrixBinned = binData(stateMatrix, binSize);
    
    % Linear regression to estimate H
    H = observationMatrix * pinv(stateMatrixBinned);
    
    % Transition matrix A based on simple physics (constant velocity model)
    A = stateMatrixBinned(:, 2:end) * pinv(stateMatrixBinned(:, 1:end-1));
    
    % Process noise covariance Q and observation noise covariance R estimation
    % Ideally, these are estimated from residuals of predictions, but for simplicity, we can start with identity matrices
    Q = eye(4); % Adjust based on variance in state transitions
    R = eye(size(spikesBinned, 1)); % Adjust based on variance in observation noise
    
    % Construct kalmanParams structure
    kalmanParams.A = A;
    kalmanParams.H = H;
    kalmanParams.Q = Q;
    kalmanParams.R = R;
    kalmanParams.initialState = mean(stateMatrix, 2); % Initial state estimate
    kalmanParams.initialCovariance = cov(stateMatrix'); % Initial covariance estimate
end

function binnedData = binData(data, binSize)
    % Simple function to bin data in columns
    dataSize = size(data, 2);
    numBins = floor(dataSize / binSize);
    binnedData = zeros(size(data, 1), numBins);
    
    for bin = 1:numBins
        startIdx = (bin - 1) * binSize + 1;
        endIdx = startIdx + binSize - 1;
        binnedData(:, bin) = mean(data(:, startIdx:endIdx), 2);
    end
end

function model = trainSVM(X, Y, kernelType)
    % Initialize parameters
    m = size(X, 1); % Number of training examples
    n = size(X, 2); % Number of features
    Y(Y == 0) = -1; % Convert 0 labels to -1
    alpha = zeros(m, 1);
    b = 0;
    C = 25; % Regularization parameter
    tol = 0.01; % Tolerance for stopping criterion
    max_passes = 50; % Max number of times to iterate over alpha's without changing
    passes = 0;

    % Choose or compute gamma for RBF kernel
    if strcmp(kernelType, 'rbfKernel')
        gamma = 1 / (size(X, 2) * var(X(:)));
    else
        gamma = []; % Not used for linear kernel
    end
    
    % Precompute the kernel matrix
    K = zeros(m, m);
    for i = 1:m
        for j = 1:m
            if strcmp(kernelType, 'linearKernel')
                K(i, j) = X(i, :) * X(j, :)';
            elseif strcmp(kernelType, 'rbfKernel')
                distanceSquared = sum((X(i, :) - X(j, :)) .^ 2);
                K(i, j) = exp(-gamma * distanceSquared);
            end
        end
    end
    

    % Train
    while passes < max_passes
        num_changed_alphas = 0;
        for i = 1:m
            Ei = b + sum(alpha .* Y .* K(:,i)) - Y(i);
            if (Y(i) * Ei < -tol && alpha(i) < C) || (Y(i) * Ei > tol && alpha(i) > 0)
                % Select random j not equal to i
                j = i;
                while j == i
                    j = randi(m);
                end
                Ej = b + sum(alpha .* Y .* K(:,j)) - Y(j);

                % Save old alphas
                alpha_i_old = alpha(i);
                alpha_j_old = alpha(j);

                % Compute L and H
                if Y(i) == Y(j)
                    L = max(0, alpha(j) + alpha(i) - C);
                    H = min(C, alpha(j) + alpha(i));
                else
                    L = max(0, alpha(j) - alpha(i));
                    H = min(C, C + alpha(j) - alpha(i));
                end
                if L == H
                    continue;
                end

                % Compute eta
                eta = 2 * K(i,j) - K(i,i) - K(j,j);
                if eta >= 0
                    continue;
                end

                % Update alpha(j)
                alpha(j) = alpha(j) - (Y(j) * (Ei - Ej)) / eta;

                % Clip alpha(j)
                alpha(j) = min(max(alpha(j), L), H);
                
                if abs(alpha(j) - alpha_j_old) < tol
                    continue;
                end

                % Update alpha(i)
                alpha(i) = alpha(i) + Y(i) * Y(j) * (alpha_j_old - alpha(j));

                % Compute b
                b1 = b - Ei - Y(i) * (alpha(i) - alpha_i_old) * K(i,i) - Y(j) * (alpha(j) - alpha_j_old) * K(i,j);
                b2 = b - Ej - Y(i) * (alpha(i) - alpha_i_old) * K(i,j) - Y(j) * (alpha(j) - alpha_j_old) * K(j,j);
                if 0 < alpha(i) && alpha(i) < C
                    b = b1;
                elseif 0 < alpha(j) && alpha(j) < C
                    b = b2;
                else
                    b = (b1 + b2) / 2;
                end

                num_changed_alphas = num_changed_alphas + 1;
            end
        end
        if num_changed_alphas == 0
            passes = passes + 1;
        else
            passes = 0;
        end
    end

    % Save model parameters
    model.kernel = kernelType;
    model.gamma = gamma; % Save gamma in the model

    % Find support vectors
    idx = alpha > 0;
    model.X = X(idx,:);
    model.y = Y(idx);
    model.alpha = alpha(idx);
    model.b = b;
    model.w = ((alpha .* Y)' * X)'; % Only for linear SVM
end

