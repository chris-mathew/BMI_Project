function [modelParameters] = positionEstimatorTraining(training_data)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.

%   training_data = load('monkeydata_training.mat');

% all_trial_data = training_data.trial;
all_trial_data = training_data;
[num_trials, num_direc] = size(all_trial_data);

[min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(all_trial_data);

bin_size = 20;
binned_data = get_binned_firing_rates(all_trial_data, bin_size);
[training_data, training_labels] = get_data_and_labels(binned_data);
[num_neurons, num_time_bins] = size(binned_data(1,1).binned_firing_rates);

modelParameters = struct;

% for j = 1: num_direc
%     for i = 1: num_time_bins
%         [whole_feat_space, ~, ~, ~, ~, sorted_eigenvectors, k] = getPCA(binned_data(i,j).binned_firing_rates, 0.8);
%         x_hand_pos = binned_data(i,j).binned_handPos(1,:);
%         y_hand_pos = binned_data(i,j).binned_handPos(2,:);
%         
%         Z = whole_feat_space(:,1:k)
%         
%         Bx = sorted_eigenvectors(:,1:k)*(inv(Z'*Z)*Z')*x_hand_pos;
%         By = sorted_eigenvectors(:,1:k)*(inv(Z'*Z')*Z')*y_hand_pos;
%     end
% end

% Flatten the data
% flattenedData = [];
% for j = 1: num_direc
%     for i = 1: num_trial
%         flattenedData = [flattenedData, reshape(binned_data(i,j).binned_firing_rates, num_neurons*num_time_bins, 1)];
%     end
% end

flattened_firing_data = zeros(num_neurons * num_time_bins, num_trials * num_direc);

    for direc = 1:num_direc
        for trial = 1:num_trials
            % Index for the current column in firingData
            column_index = (direc-1) * num_trials + trial;

            for time_bin = 1:num_time_bins
                % Row index range for the current time bin
                row_start = (time_bin-1) * num_neurons + 1;
                row_end = time_bin * num_neurons;

                % Extract firing rates for this time bin and trial
                firing_rates = binned_data(trial, direc).binned_firing_rates(:, time_bin);

                % Place the firing rates in the correct location in firingData
                flattened_firing_data(row_start:row_end, column_index) = firing_rates;
            end
        end
    end


modelParameters.pcr = struct;

modelParameters.max_timebin_index = floor(min_trial_duration/bin_size);


% x_hand_pos = zeros(num_trials,num_time_bins,num_direc);
% y_hand_pos = zeros(num_trials,num_time_bins,num_direc);
% for j = 1: num_direc
%     for i = 1: num_trials
%         x_hand_pos(i,:,j) = binned_data(i,j).binned_handPos(1,:);
%         y_hand_pos(i,:,j) = binned_data(i,j).binned_handPos(2,:);
%     end
% end

[x_hand_pos_binned, y_hand_pos_binned, x_hand_pos, y_hand_pos] = get_hand_pos_data (all_trial_data, bin_size);
start_time = 320;
end_time = num_time_bins*bin_size;
time_binned_index = (start_time:bin_size:end_time)/bin_size;
start_time_index = start_time/bin_size;

for j = 1: num_direc
    x_hand_pos_current_direc = x_hand_pos_binned(:,:,j);
    y_hand_pos_current_direc = y_hand_pos_binned(:,:,j);

    for i = 1:length(time_binned_index)
        firing_data_current_direc = flattened_firing_data(1:start_time_index*num_neurons+(i-1)*num_neurons,(j-1)*(num_trials)+1:(j)*num_trials);
%         flattened_firing_data((j-1)*(num_trials)+1:j*num_trials)
        mean_centered_x_pcr_data = x_hand_pos_current_direc(:,time_binned_index(i)); % - mean(x_hand_pos_current_direc(:,time_binned_index(i)));
        mean_centered_y_pcr_data = y_hand_pos_current_direc(:,time_binned_index(i)); %- mean(y_hand_pos_current_direc(:,time_binned_index(i)));

        [whole_feat_space, ~, ~, ~, ~, sorted_eigenvectors, k] = getPCA(firing_data_current_direc, 0.8);
        V = whole_feat_space(:,1:k);
        Z = (V'*firing_data_current_direc-mean(firing_data_current_direc,1))'; % project principal components up to top k comp to data
    
%         PCR formula from: https://rpubs.com/esobolewska/pcr-step-by-step
%         size_sorted_eigenvectors_k= size(sorted_eigenvectors(:, 1:k))
%         size2= size(inv(Z'*Z))
%         size3= size(Z')
%         size4= size((V*inv(Z'*Z)*Z'))
%         size5= size(x_hand_pos(:,:,j))
%         size6 = size(V*(inv(Z'*Z)*Z'))
%         size7 = size(mean_centered_x_pcr_data)
    %     Bx = regress(x_hand_pos', sorted_eigenvectors(:, 1:k));
    %     By = regress(y_hand_pos', sorted_eigenvectors(:, 1:k));
    
    %     Bx = sorted_eigenvectors(:,1:k)*(inv(Z'*Z)*Z')*x_hand_pos';
    %     By = sorted_eigenvectors(:,1:k)*(inv(Z'*Z')*Z')*y_hand_pos';
    % 
        Bx = (V*inv(Z'*Z)*Z')*mean_centered_x_pcr_data;
        By = (V*inv(Z'*Z)*Z')*mean_centered_y_pcr_data;
        
        x_hand_pos_mean = squeeze(mean(x_hand_pos,1));
        y_hand_pos_mean = squeeze(mean(y_hand_pos,1));
        modelParameters.pcr(j,i).bx = Bx;
        modelParameters.pcr(j,i).by = By;
        modelParameters.pcr(j,i).mean_firing = mean(firing_data_current_direc);
        modelParameters.pcr(j,i).ex = x_hand_pos_mean;
        modelParameters.pcr(j,i).ey = y_hand_pos_mean;
    end
end

% % Define the neural network architecture
% num_hidden_neurons = 30; % Adjust based on your dataset
% net = patternnet(num_hidden_neurons, 'trainbr');
% 
% % Divide data into training, validation, and test sets
% net.divideParam.trainRatio = 0.70;
% net.divideParam.valRatio = 0.30;
% net.divideParam.testRatio = 0;
% 
% % Train the network
% [net, tr] = train(net, training_data', labels');

% numHiddenUnits = 120;
% numClasses = 8;
% 
% layers = [ ...
%     sequenceInputLayer(size(training_data, 2))
%     bilstmLayer(numHiddenUnits,OutputMode="last")
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer]
% 
% options = trainingOptions("adam", ...
%     InitialLearnRate=0.002,...
%     MaxEpochs=150, ...
%     Shuffle="never", ...
%     GradientThreshold=1, ...
%     Verbose=false, ...
%     Plots="training-progress");
% 
% net = trainNetwork(training_data', categorical_labels', layers, options);

% building a simple nn model
layers = [
    sequenceInputLayer(size(training_data, 2))
    
    fullyConnectedLayer(100)
    batchNormalizationLayer
    reluLayer
    
%     fullyConnectedLayer(50)
%     batchNormalizationLayer
%     reluLayer
    
    fullyConnectedLayer(num_direc) % Replace numClasses with the actual number of classes
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',60, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% convert label to categorical labels
categorical_labels  = categorical(training_labels); 
% categorical_labels = full(ind2vec(labels));

% Train the network
net = trainNetwork(training_data', categorical_labels', layers, options);

modelParameters.net = net;


% ---------- defined functions ----------

function [min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(trial)
[num_trial, num_direc] = size(trial);

min_trial_duration = inf;
max_trial_duration = 0;
% find the min trial duration
for j = 1:num_direc
    for i = 1:num_trial
        current_trial_duration = size(trial(i,j).spikes,2);
        if current_trial_duration < min_trial_duration
            min_trial_duration = current_trial_duration; % Update minTrialDuration if current trial duration is shorter
        end
        if current_trial_duration > max_trial_duration
            max_trial_duration = current_trial_duration; % Update if current duration is longer
        end
    end
end
end


function same_size_data = get_same_size_data(trial, min_trial_duration)
same_size_data = struct;
[num_trial, num_direc] = size(trial);
for j = 1:num_direc
    for i = 1:num_trial
        all_neuro_spikes_data = trial(i,j).spikes(:,1:min_trial_duration);
        tempx = trial(i, j).handPos(1, 1:min_trial_duration);
        tempy = trial(i, j).handPos(2, 1:min_trial_duration);
        same_size_data(i, j).spikes_adjusted = all_neuro_spikes_data;
        same_size_data(i, j).x_hand_pos_adjusted = tempx;
        same_size_data(i, j).x_hand_pos_adjusted = tempy;
    end
end
end

function binned_data = get_binned_firing_rates(trial, bin_size)
binned_data = struct;
[num_trial, num_direc] = size(trial);

[min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(trial);

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



% function [x_hand_pos, y_hand_pos] = get_hand_pos_data (trial, bin_size)
% % 
% % binned_hand_pos_data = struct;
% [num_trial, num_direc] = size(trial);
% 
% [min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(trial);
% 
% x_hand_pos = zeros(num_trials,num_time_bins,num_direc);
% y_hand_pos = zeros(num_trials,num_time_bins,num_direc);
% 
% for j = 1:num_direc
%     for i = 1:num_trial
%         tempx = trial(i, j).handPos(1, 1:min_trial_duration);
%         tempy = trial(i, j).handPos(2, 1:min_trial_duration);
%         binned_handPos_x = tempx(1:bin_size:end);
%         binned_handPos_y = tempy(1:bin_size:end);
%         x_hand_pos(i,:,j) = binned_handPos_x;
%         y_hand_pos(i,:,j) = binned_handPos_y;
%     end
% end
% end

function [x_hand_pos_binned, y_hand_pos_binned, x_hand_pos, y_hand_pos] = get_hand_pos_data (trial, bin_size)
% 
% binned_hand_pos_data = struct;
[num_trial, num_direc] = size(trial);

[min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(trial);
% [num_neurons, num_time_bins] = size(binned_data(1,1).binned_firing_rates);

num_time_bins = length(1:bin_size:min_trial_duration);

    x_hand_pos = zeros(num_trial, min_trial_duration, num_direc);
    y_hand_pos = zeros(num_trial, min_trial_duration, num_direc);
    x_hand_pos_binned = zeros(num_trial, num_time_bins, num_direc);
    y_hand_pos_binned = zeros(num_trial, num_time_bins, num_direc);

    for j = 1:num_direc
        for i = 1:num_trial
            tempx = trial(i, j).handPos(1, 1:min_trial_duration);
            tempy = trial(i, j).handPos(2, 1:min_trial_duration);

            x_hand_pos(i,:,j) = tempx;
            y_hand_pos(i,:,j) = tempy;

            binned_handPos_x = tempx(1:bin_size:end);
            binned_handPos_y = tempy(1:bin_size:end);

            x_hand_pos_binned(i,:,j) = binned_handPos_x;
            y_hand_pos_binned(i,:,j) = binned_handPos_y;
        end
    end
% x_hand_pos = zeros(num_trial, maxSize, num_direc);
% y_hand_pos = zeros(num_trial, maxSize, num_direc);
% 
% x_hand_pos_binned = zeros(num_trial, num_time_bins, num_direc);
% y_hand_pos_binned = zeros(num_trial, num_time_bins, num_direc);
% 
% for j = 1:num_direc
%     for i = 1:num_trial
%         currentSize = size(trial(i, j).handPos, 2);
%         padSize = maxSize - currentSize;
% 
%         x_hand_pos(i, :, j) = [trial(i, j).handPos(1, :), trial(i, j).handPos(1, end) * ones(1, padSize)];
%         y_hand_pos(i, :, j) = [trial(i, j).handPos(2, :), trial(i, j).handPos(2, end) * ones(1, padSize)];
% 
%         x_hand_pos_binned(i, :, j) = x_hand_pos(i, 1:bin_size:end, j);
%         y_hand_pos_binned(i, :, j) = y_hand_pos(i, 1:bin_size:end, j);
%     end
% end
end


function [data, labels] = get_data_and_labels(binned_data)
[num_trial, num_direc] = size(binned_data);
num_neurons = size(binned_data(1,1).binned_firing_rates, 1);
% num_bins = length(binned_data(1,1).binned_firing_rates(1,:));

% make it to 2D data of (num_trial x num_direc, num_neurons)
data = zeros(num_trial * num_direc, num_neurons);
labels = zeros(num_trial * num_direc, 1);

count = 1;
for j = 1:num_direc
    for i = 1:num_trial
        % Flatten neuron firing rates for each trial
        avg_firing_rates = mean(binned_data(i, j).binned_firing_rates, 2); 

        data(count, :) = avg_firing_rates';
        labels(count) = j; % Direction label
        count = count + 1;
    end
end
end


function test_avg_rates = get_test_data(binned_test_data)
[num_trial, num_direc] = size(binned_test_data);
num_neurons = size(binned_test_data(1,1).binned_firing_rates, 1);
% num_bins = length(binned_data(1,1).binned_firing_rates(1,:));

% make it to 2D data of (num_trial x num_direc, num_neurons)
data = zeros(num_trial * num_direc, num_neurons);
% labels = zeros(num_trial * num_direc, 1);

count = 1;
for j = 1:num_direc
    for i = 1:num_trial
        % Flatten neuron firing rates for each trial
        avg_firing_rates = mean(binned_test_data(i, j).binned_firing_rates, 2); 

        data(count, :) = avg_firing_rates';
%         labels(count) = j; % Direction label
        count = count + 1;
    end
end
end


function [whole_feat_space, new_feat_space, eigenvalues, index, explained_variance, sorted_eigenvectors, k] = getPCA(binned_rates, threshold)
if nargin < 2
        threshold = 0.8;
end
% according to steps in: https://towardsdatascience.com/a-step-by-step-introduction-to-pca-c0d78e26a0dd
% Step 1: Standardize the data
% binned_rates = binned_data.binned_firing_rates; % binned_firing_rates is (num_neurons x binned_rates)

eps = 1e-25;
standard_binned_data = (binned_rates - mean(binned_rates,2))./(std(binned_rates,0,2) + eps);

% Step 2: Compute the covariance matrix
cov_matrix = (standard_binned_data'*standard_binned_data)/(size(standard_binned_data,1)-1);

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
top_k_eigenvectors = sorted_eigenvectors(:, 1:k);

% Step 7: Compute the new k-dimensional feature space
whole_feat_space = standard_binned_data * sorted_eigenvectors;
new_feat_space = standard_binned_data * top_k_eigenvectors;

end

end