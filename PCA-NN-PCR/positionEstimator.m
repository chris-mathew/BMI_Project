function [x, y] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand

modelParameters = modelParameters;
net = modelParameters.net;
bin_size = 20;
past_current_trial = test_data;
start_time = 320;
% window_size = 50;

%  binning to get binned firing rates
binned_data = get_binned_firing_rates(past_current_trial, bin_size);
% binned_data = get_binned_firing_rates(past_current_trial, bin_size, window_size);
[num_neurons, num_time_bins] = size(binned_data(1,1).binned_firing_rates);

% get average rate for NN to predict direction
test_avg_rates = get_test_data(binned_data);


[num_neurons, spikes_data_length] = size(past_current_trial.spikes);

% disp(spikes_data_length)
test_time_bin = (spikes_data_length/bin_size) - (start_time/bin_size)+1;
% disp(test_time_bin)
max_timebin_index = modelParameters.max_timebin_index;
data_max_time_bin = max_timebin_index - (start_time/bin_size);

if spikes_data_length <= max_timebin_index*bin_size

firing_rates = reshape(binned_data.binned_firing_rates,[],1);
% disp(size(spikes_data_length))
% pred_direc = ;
pred_direc = classify(net, test_avg_rates');
pred_direc = double(pred_direc);

bx = modelParameters.pcr(pred_direc,test_time_bin).bx;
by = modelParameters.pcr(pred_direc,test_time_bin).by;
ex = modelParameters.pcr(pred_direc,test_time_bin).ex(:,pred_direc);
ey = modelParameters.pcr(pred_direc,test_time_bin).ey(:,pred_direc);
% ex = 0;
% ey = 0;
mean_firing = modelParameters.pcr(pred_direc,test_time_bin).mean_firing;
x = (firing_rates-mean(mean_firing))'*bx + ex;
y = (firing_rates-mean(mean_firing))'*by + ey;
% x = (firing_rates)'*bx + ex;
% y = (firing_rates)'*by + ey;
x = x(spikes_data_length,1);
y = y(spikes_data_length,1);
% disp(x)
elseif spikes_data_length > max_timebin_index*bin_size

firing_rates = reshape(binned_data.binned_firing_rates,[],1);
pred_direc = classify(net, test_avg_rates');

bx = modelParameters.pcr(pred_direc,data_max_time_bin).bx;
by = modelParameters.pcr(pred_direc,data_max_time_bin).by;
ex = modelParameters.pcr(pred_direc,data_max_time_bin).ex(:,pred_direc);
ey = modelParameters.pcr(pred_direc,data_max_time_bin).ey(:,pred_direc);
% ex = 0;
% ey = 0;
mean_firing = modelParameters.pcr(pred_direc,data_max_time_bin).mean_firing;
x = (firing_rates(1:length(bx)) - mean(firing_rates(1:length(bx))))'*bx + ex;
y = (firing_rates(1:length(bx)) - mean(firing_rates(1:length(bx))))'*by + ey;
% x = (firing_rates(1:length(bx)))'*bx + ex;
% y = (firing_rates(1:length(bx)))'*by + ey;
x = x(max_timebin_index*bin_size,1);
y = y(max_timebin_index*bin_size,1);
% x=0;
% y=0;
end

   
end

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

% function binned_data = get_binned_firing_rates(trial, bin_size, scale_window)
% binned_data = struct;
% [num_trial, num_direc] = size(trial);
% 
% [min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(trial);
% 
% win = 10 * (scale_window / bin_size);
% normstd = scale_window / bin_size;
% alpha = (win - 1) / (2 * normstd);
% temp1 = -(win - 1) / 2 : (win - 1) / 2;
% gaussian_window = exp((-1 / 2) * (alpha * temp1 / ((win - 1) / 2)) .^ 2)';
% gaussian_window = gaussian_window / sum(gaussian_window);
% 
% for j = 1:num_direc
%     for i = 1:num_trial
%         all_neuro_spikes_data = trial(i,j).spikes(:,1:min_trial_duration); % all_neuro_spikes is of (98 x spike_duration)
%         num_neurons = size(trial(i,j).spikes,1);
% %         each_spike_duration = size(all_neuro_spikes_data,2);
%         binned_time = 1:bin_size:min_trial_duration + bin_size;
%         binned_spikes = zeros(num_neurons,length(binned_time)-1);
% 
%         smoothed_spikes = zeros(size(binned_spikes));
% 
%         for n = 1:num_neurons
%             spike_index = find(all_neuro_spikes_data(n, :) == 1);
%             binned_spikes(n, :) = histcounts(spike_index, binned_time);
%             
%             smoothed_spikes(n, :) = conv(binned_spikes(n, :), gaussian_window, 'same');
%             
%         end
%         binned_firing_rates = smoothed_spikes*(1000/bin_size);
% 
% %         tempx = trial(i, j).handPos(1, 1:min_trial_duration);
% %         tempy = trial(i, j).handPos(2, 1:min_trial_duration);
% %         binned_handPos_x = tempx(1:bin_size:end);
% %         binned_handPos_y = tempy(1:bin_size:end);
% 
%         binned_data(i, j).binned_spikes = binned_spikes;
%         binned_data(i, j).binned_firing_rates = binned_firing_rates;
% %         binned_data(i, j).binned_handPos = [binned_handPos_x; binned_handPos_y];
%     end
% end 
% end

function [x_hand_pos, y_hand_pos] = get_hand_pos_data (trial, bin_size)
% 
% binned_hand_pos_data = struct;
[num_trial, num_direc] = size(trial);

[min_trial_duration, max_trial_duration] = findMinMaxTrialDuration(trial);

x_hand_pos = zeros(num_trials,num_time_bins,num_direc);
y_hand_pos = zeros(num_trials,num_time_bins,num_direc);

for j = 1:num_direc
    for i = 1:num_trial
        tempx = trial(i, j).handPos(1, 1:min_trial_duration);
        tempy = trial(i, j).handPos(2, 1:min_trial_duration);
        binned_handPos_x = tempx(1:bin_size:end);
        binned_handPos_y = tempy(1:bin_size:end);
        x_hand_pos(i,:,j) = binned_handPos_x;
        y_hand_pos(i,:,j) = binned_handPos_y;
    end
end
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
test_avg_rates = zeros(num_trial * num_direc, num_neurons);
% labels = zeros(num_trial * num_direc, 1);

count = 1;
for j = 1:num_direc
    for i = 1:num_trial
        % Flatten neuron firing rates for each trial
        avg_firing_rates = mean(binned_test_data(i, j).binned_firing_rates, 2); 

        test_avg_rates(count, :) = avg_firing_rates';
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
