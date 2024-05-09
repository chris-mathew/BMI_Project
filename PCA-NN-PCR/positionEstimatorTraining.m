function [modelParameters] = positionEstimatorTraining(training_data)
 
all_trial_data = training_data;
[num_trials, num_direc] = size(all_trial_data);

min_trial_duration = findMinMaxTrialDuration(all_trial_data);

bin_size = 20;

% window_size = 50;
% binned_data = get_binned_firing_rates(all_trial_data, bin_size, window_size);

binned_data = get_binned_firing_rates(all_trial_data, bin_size);
[num_neurons, num_time_bins] = size(binned_data(1,1).binned_firing_rates);

modelParameters = struct;

% resturcture data into 2D as a better input to NN model 
flattened_firing_data = zeros(num_neurons * num_time_bins, num_trials * num_direc);

    for j = 1:num_direc
        for i = 1:num_trials
            column_index = (j-1) * num_trials + i;

            for k = 1:num_time_bins
                row_start = (k-1) * num_neurons + 1;
                row_end = k * num_neurons;

                firing_rates = binned_data(i,j).binned_firing_rates(:,k);

                flattened_firing_data(row_start:row_end, column_index) = firing_rates;
            end
        end
    end


modelParameters.pcr = struct;
modelParameters.max_timebin_index = floor(min_trial_duration/bin_size);


[x_hand_pos_binned, y_hand_pos_binned, x_hand_pos, y_hand_pos] = get_hand_pos_data (all_trial_data, bin_size);
start_time = 320;
end_time = num_time_bins*bin_size;
time_binned_index = (start_time:bin_size:end_time)/bin_size;
start_time_index = start_time/bin_size;

x_hand_pos_binned = x_hand_pos_binned(:,time_binned_index(1):time_binned_index(end),:);
y_hand_pos_binned = y_hand_pos_binned(:,time_binned_index(1):time_binned_index(end),:);

for j = 1: num_direc
    x_hand_pos_current_direc = x_hand_pos_binned(:,:,j);
    y_hand_pos_current_direc = y_hand_pos_binned(:,:,j);

    for i = 1:length(time_binned_index)
        firing_data_current_direc = flattened_firing_data(1:start_time_index*num_neurons+(i-1)*num_neurons,(j-1)*(num_trials)+1:(j)*num_trials);
%         flattened_firing_data((j-1)*(num_trials)+1:j*num_trials)
        mean_centered_x_pcr_data = x_hand_pos_current_direc(:,i) - mean(x_hand_pos_current_direc(:,i));
        mean_centered_y_pcr_data = y_hand_pos_current_direc(:,i) - mean(y_hand_pos_current_direc(:,i));

        [whole_feat_space, ~, ~, k] = getPCA(firing_data_current_direc, 0.8);
        
        V = whole_feat_space(:,1:k);
        Z = V'*(firing_data_current_direc - mean(firing_data_current_direc,1)); % project principal components up to top k comp to data
        Z = Z';
%         PCR formula from: https://rpubs.com/esobolewska/pcr-step-by-step
        Bx = (V*inv(Z'*Z)*Z')*mean_centered_x_pcr_data;
        By = (V*inv(Z'*Z)*Z')*mean_centered_y_pcr_data;
        
        x_hand_pos_mean = squeeze(mean(x_hand_pos,1));
        y_hand_pos_mean = squeeze(mean(y_hand_pos,1));
        modelParameters.pcr(j,i).bx = Bx;
        modelParameters.pcr(j,i).by = By;
        modelParameters.pcr(j,i).mean_firing = mean(firing_data_current_direc,1);
        modelParameters.pcr(j,i).ex = x_hand_pos_mean;
        modelParameters.pcr(j,i).ey = y_hand_pos_mean;
    end
end

% define NN architecture
% preprocessing data
[training_data, training_labels] = get_data_and_labels(binned_data);

% convert to categorical labels
categorical_labels  = categorical(training_labels); 
% categorical_labels = full(ind2vec(labels));

num_samples = size(training_data, 1);
num_training_data = floor(0.8 * num_samples); % 80% for training

shuffle_index = randperm(num_samples);
shuffled_data = training_data(shuffle_index, :);
shuffled_labels = categorical_labels(shuffle_index, :);

% split data into training and validation 
training_data = shuffled_data(1:num_training_data, :);
categorical_labels = shuffled_labels(1:num_training_data, :);
valData = shuffled_data(num_training_data+1:end, :);
valLabels = shuffled_labels(num_training_data+1:end, :);

% building a simple nn model
layers = [
    sequenceInputLayer(size(training_data, 2))

    fullyConnectedLayer(100)
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.4)

    fullyConnectedLayer(50)
    batchNormalizationLayer
    leakyReluLayer
    dropoutLayer(0.3)
    
%     layerNormalizationLayer
    fullyConnectedLayer(num_direc)
    softmaxLayer
    classificationLayer];

options = trainingOptions('rmsprop', ...
    'MaxEpochs',150, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ... 
    'ValidationData', {valData', valLabels'}, ...
    'ValidationFrequency', 5);
    %'Plots', 'training-progress');

% Train the network
net = trainNetwork(training_data', categorical_labels', layers, options);

modelParameters.net = net;


% ---------- defined functions ----------

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


function binned_data = get_binned_firing_rates(trial, bin_size)
binned_data = struct;
[num_trial, num_direc] = size(trial);

% [min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(trial);

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


function [x_hand_pos_binned, y_hand_pos_binned, x_hand_pos, y_hand_pos] = get_hand_pos_data (trial, bin_size)
% 
% binned_hand_pos_data = struct;
[num_trial, num_direc] = size(trial);

% [min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(trial);
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
end


function [data, labels] = get_data_and_labels(binned_data)
[num_trial, num_direc] = size(binned_data);
num_neurons = size(binned_data(1,1).binned_firing_rates, 1);
% num_bins = length(binned_data(1,1).binned_firing_rates(1,:));

% make it to 2D data of (num_trial x num_direc, num_neurons) (400,98)
data = zeros(num_trial * num_direc, num_neurons);
labels = zeros(num_trial * num_direc, 1); %(400,1)

count = 1;
for j = 1:num_direc
    for i = 1:num_trial
        % flatten neuron firing rates for each trial
        avg_firing_rates = mean(binned_data(i, j).binned_firing_rates, 2); 

        data(count, :) = avg_firing_rates';
        labels(count) = j; % Direction label
        count = count + 1;
    end
end
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



end
