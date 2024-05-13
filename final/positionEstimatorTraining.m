function [modelParameters] = positionEstimatorTraining(training_data)

    % initialise parameters used 
    [num_trials, num_direc] = size(training_data);
    min_trial_duration = findMinMaxTrialDuration(training_data);
    bin_size = 20;
    modelParameters = struct;
    
    % train SVM for hand direction classification and prediction  
    [X, Y] = prepareSVMTrainingData(training_data); % X = (400x98), Y = (400x1)
    numClasses = size(training_data, 2); 
    
    for numSvm = 1:numClasses
        % current class is positive,  all other classes are negative
        Y_train = double(Y == numSvm);
        Y_train(Y_train == 0) = -1;
    
        % train SVM model for the current class
        modelParameters.svmModel{numSvm} = trainSVM(X, Y_train);
    end

    % resturcture data into 2D as a better input for PCA
    binned_data = get_binned_firing_rates(training_data, bin_size, min_trial_duration);
    [num_neurons, num_time_bins] = size(binned_data(1,1).binned_firing_rates);
    
    data_2D = zeros(num_neurons * num_time_bins, num_trials * num_direc);
    
        for j = 1:num_direc
            for i = 1:num_trials
                column_index = (j-1) * num_trials + i;
    
                for k = 1:num_time_bins
                    row_start = (k-1) * num_neurons + 1;
                    row_end = k * num_neurons;
    
                    firing_rates = binned_data(i,j).binned_firing_rates(:,k);
    
                    data_2D(row_start:row_end, column_index) = firing_rates;
                end
            end
        end
    
    % start estimation of PCR model of data
    modelParameters.pcr = struct;
    modelParameters.max_timebin_index = floor(min_trial_duration/bin_size);
    
    % reshaping training_data into 3D matrix of format (num_trials x num_time_bins x num_direc)
    [x_binned, y_binned, x_mean, y_mean] = get_3D_data(training_data, bin_size, min_trial_duration);
    
    start_time = 320;
    end_time = num_time_bins*bin_size;
    time_binned_index = (start_time:bin_size:end_time)/bin_size;
    start_time_index = start_time/bin_size;
    
    for j = 1: num_direc
        x_current_direc = x_binned(:,:,j);
        y_current_direc = y_binned(:,:,j);
    
        for i = 1:length(time_binned_index)
            spikes_current_direc = data_2D(1:start_time_index*num_neurons+(i-1)*num_neurons,(j-1)*(num_trials)+1:(j)*num_trials);
            centered_x = x_current_direc(:,i) - mean(x_current_direc(:,i));
            centered_y = y_current_direc(:,i) - mean(y_current_direc(:,i));
    
            [whole_feat_space, ~, ~, k] = getPCA(spikes_current_direc, 0.8);
            spikes_current_direc = spikes_current_direc - mean(spikes_current_direc,1);
            
            V = whole_feat_space(:,1:k);
            Z = V'*(spikes_current_direc); % project principal components up to top k comp to data
            Z = Z';
    
            % PCR formula from: https://rpubs.com/esobolewska/pcr-step-by-step
            Bx = (V*inv(Z'*Z)*Z')*centered_x;
            By = (V*inv(Z'*Z)*Z')*centered_y;
            
            modelParameters.pcr(j,i).bx = Bx;
            modelParameters.pcr(j,i).by = By;
            modelParameters.pcr(j,i).mean_firing = mean(spikes_current_direc,1);
            modelParameters.pcr(j,i).ex = x_mean;
            modelParameters.pcr(j,i).ey = y_mean;
        end
    end
end



% ========================= defined functions =========================
function [X, Y] = prepareSVMTrainingData(training_data)
    % This function prepares the input feature matrix X and target vector Y for SVM training
    % Spike counts are averaged over a specified initial time window for each trial
    initialTimeWindow = 320; 
    numTrials = size(training_data, 1);
    numDirections = size(training_data, 2);
    
    X = []; % feature matrix
    Y = []; % target vector
    
    % extract features and labels
    for single_trial = 1:numTrials
        for direction = 1:numDirections
            spikes = training_data(single_trial, direction).spikes;
            avgSpikes = mean(spikes(:, 1:initialTimeWindow), 2)'; 
            X = [X; avgSpikes];
            Y = [Y; direction];
        end
    end

end


function model = trainSVM(X, Y)
% referenced and modified from: https://github.com/everpeace/ml-class-assignments/tree/master/ex6.Support_Vector_Machines
    m = size(X, 1); % num of training examples = 400
    n = size(X, 2); % num of features = 98
    Y(Y == 0) = -1; 
    alpha = zeros(m, 1);
    b = 0;
    C = 25; % regularisation parameter
    tol = 0.01; % tolerance for stopping criterion
    max_passes = 50; % max number of times to iterate over alpha
    passes = 0;
    gamma = 1 / (size(X, 2) * var(X(:)));

    
    % compute the kernel matrix
    K = zeros(m, m);
    for i = 1:m
        for j = 1:m
                distanceSquared = sum((X(i, :) - X(j, :)) .^ 2);
                K(i, j) = exp(-gamma * distanceSquared);
        end
    end
    

    % train
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

    idx = alpha > 0;
    model.gamma = gamma; 
    model.X = X(idx,:);
    model.y = Y(idx);
    model.alpha = alpha(idx);
    model.b = b;
end


function min_trial_duration = findMinMaxTrialDuration(trial)
[num_trial, num_direc] = size(trial);

min_trial_duration = inf;
% find min trial duration
for j = 1:num_direc
    for i = 1:num_trial
        current_trial_duration = size(trial(i,j).spikes,2);
        if current_trial_duration < min_trial_duration
            min_trial_duration = current_trial_duration; % Update minTrialDuration if current trial duration is shorter
        end
    end
end
end


function binned_data = get_binned_firing_rates(trial, bin_size, min_trial_duration)
binned_data = struct;
[num_trial, num_direc] = size(trial);


for j = 1:num_direc
    for i = 1:num_trial
        all_neuro_spikes_data = trial(i,j).spikes(:,1:min_trial_duration); % all_neuro_spikes is of (98 x spike_duration)
        num_neurons = size(trial(i,j).spikes,1);
%         each_spike_duration = size(all_neuro_spikes_data,2);
        binned_time = 1:bin_size:min_trial_duration + bin_size;
        binned_spikes = zeros(num_neurons,length(binned_time)-1);
        for n = 1:num_neurons
            spike_index = find(all_neuro_spikes_data(n, :) == 1);
            binned_spikes(n, :) = histcounts(spike_index, binned_time);
        end
        binned_firing_rates = binned_spikes*(1000/bin_size);

%         tempx = trial(i, j).handPos(1, 1:min_trial_duration);
%         tempy = trial(i, j).handPos(2, 1:min_trial_duration);
%         binned_handPos_x = tempx(1:bin_size:end);
%         binned_handPos_y = tempy(1:bin_size:end);

        binned_data(i, j).binned_spikes = binned_spikes;
        binned_data(i, j).binned_firing_rates = binned_firing_rates;
%         binned_data(i, j).binned_handPos = [binned_handPos_x; binned_handPos_y];
    end
end 
end


function [x_binned, y_binned, x_mean, y_mean] = get_3D_data(trial, bin_size, min_trial_duration)

    [num_trial, num_direc] = size(trial);
    num_time_bins = length(1:bin_size:min_trial_duration);
    start_time = 320;
    end_time = num_time_bins * bin_size;
    time_binned_index = (start_time:bin_size:end_time) / bin_size;
    start_time_index = start_time / bin_size;

    x_hand_pos = zeros(num_trial, min_trial_duration, num_direc);
    y_hand_pos = zeros(num_trial, min_trial_duration, num_direc);
    x_binned = zeros(num_trial, num_time_bins, num_direc);
    y_binned = zeros(num_trial, num_time_bins, num_direc);

    for j = 1:num_direc
        for i = 1:num_trial
            tempx = trial(i, j).handPos(1, 1:min_trial_duration);
            tempy = trial(i, j).handPos(2, 1:min_trial_duration);

            x_hand_pos(i,:,j) = tempx;
            y_hand_pos(i,:,j) = tempy;

            binned_handPos_x = tempx(1:bin_size:end);
            binned_handPos_y = tempy(1:bin_size:end);

            x_binned(i,:,j) = binned_handPos_x;
            y_binned(i,:,j) = binned_handPos_y;
        end
    end

    % get mean position for x and y hand position 
    x_mean = squeeze(mean(x_hand_pos, 1));
    y_mean = squeeze(mean(y_hand_pos, 1));

    x_binned = x_binned(:, time_binned_index(1):time_binned_index(end), :);
    y_binned = y_binned(:, time_binned_index(1):time_binned_index(end), :);
end




function [whole_feat_space, sorted_eigenvalues, sorted_eigenvectors, k] = getPCA(binned_rates, threshold)
if nargin < 2
        threshold = 0.8;
end
% according to steps in: https://towardsdatascience.com/a-step-by-step-introduction-to-pca-c0d78e26a0dd
% Step 1: Standardize the data
% binned_rates = binned_data.binned_firing_rates; % binned_firing_rates is (num_neurons x binned_rates)

% eps = 1e-25;
standard_binned_data = (binned_rates - mean(binned_rates,2)); %./(std(binned_rates,0,2) + eps);

% Step 2: Compute the covariance matrix
cov_matrix = (standard_binned_data'*standard_binned_data)/(size(standard_binned_data,2));

% Step 3: Perform eigendecompositon on the covariance matrix
[eigenvectors, eigenvalues] = eig(cov_matrix); % https://uk.mathworks.com/help/matlab/ref/eig.html#d126e420773
% diag(eigenvalues)

% Step 4: Sort the eigenvectors in decreasing order based on corresponding eigenvalues
[sorted_eigenvalues, index] = sort(diag(eigenvalues), 'descend'); % https://uk.mathworks.com/help/matlab/ref/sort.html#d126e1507433
sorted_eigenvectors = eigenvectors(:, index);

% Step 5: Determine k, the number of top principal components to select
% sorted_eigenvalues_test = [1.567 0.345 0.556]
% cumsum(sorted_eigenvalues_test)
% sum(sorted_eigenvalues_test)
explained_variance = cumsum(sorted_eigenvalues)/sum(sorted_eigenvalues);
% Find the number of components that explain between 80% and 90% of the variance
k = find(explained_variance >= threshold , 1, 'first');

% Step 6: Construct the projection matrix from the chosen number of top principal components
% top_k_eigenvectors = sorted_eigenvectors(:, 1:k);

% Step 7: Compute the new k-dimensional feature space
whole_feat_space = standard_binned_data * sorted_eigenvectors;
whole_feat_space = whole_feat_space./sqrt(sum(whole_feat_space.^2));
% new_feat_space = standard_binned_data * top_k_eigenvectors;

end

