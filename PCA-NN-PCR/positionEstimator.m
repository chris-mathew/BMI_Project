function [x, y, modelParameters] = positionEstimator(past_current_trial, modelParameters)

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
  
% newParameters = modelParameters;
% modelParameters = modelParameters;
net = modelParameters.net;
bin_size = 20;
% past_current_trial = test_data;
start_time = 320;
% window_size = 50;

%  binning to get binned firing rates
binned_data = get_binned_firing_rates(past_current_trial, bin_size);
% binned_data = get_binned_firing_rates(past_current_trial, bin_size, window_size);
% [num_neurons, num_time_bins] = size(binned_data(1,1).binned_firing_rates);

% get average rate for NN to predict direction
test_avg_rates = get_test_data(binned_data);


[~, spikes_data_length] = size(past_current_trial.spikes);

% disp(spikes_data_length)
test_time_bin = (spikes_data_length/bin_size) - (start_time/bin_size)+1;
% disp(test_time_bin)
max_timebin_index = modelParameters.max_timebin_index;
data_max_time_bin = max_timebin_index - (start_time/bin_size);
data_max_time = max_timebin_index*bin_size;

if spikes_data_length <= data_max_time

firing_rates = reshape(binned_data.binned_firing_rates,[],1);
% disp(size(spikes_data_length))
% pred_direc = ;
pred_direc = classify(net, test_avg_rates');
pred_direc = double(pred_direc);

bx = modelParameters.pcr(pred_direc,test_time_bin).bx;
by = modelParameters.pcr(pred_direc,test_time_bin).by;
ex = modelParameters.pcr(pred_direc,test_time_bin).ex(:,pred_direc);
ey = modelParameters.pcr(pred_direc,test_time_bin).ey(:,pred_direc);
mean_firing = modelParameters.pcr(pred_direc,test_time_bin).mean_firing;
x = (firing_rates-mean(mean_firing))'*bx + ex;
y = (firing_rates-mean(mean_firing))'*by + ey;
x = x(spikes_data_length,1);
y = y(spikes_data_length,1);
% disp(x)

modelParameters.pred_direc = pred_direc;

elseif spikes_data_length > data_max_time

firing_rates = reshape(binned_data.binned_firing_rates,[],1);

% see if use last stored or prediction is better
% pred_direc = classify(net, test_avg_rates');
pred_direc = modelParameters.pred_direc;

bx = modelParameters.pcr(pred_direc,data_max_time_bin).bx;
by = modelParameters.pcr(pred_direc,data_max_time_bin).by;
ex = modelParameters.pcr(pred_direc,data_max_time_bin).ex(:,pred_direc);
ey = modelParameters.pcr(pred_direc,data_max_time_bin).ey(:,pred_direc);
% mean_firing = modelParameters.pcr(pred_direc,data_max_time_bin).mean_firing;
x = (firing_rates(1:length(bx)) - mean(firing_rates(1:length(bx))))'*bx + ex;
y = (firing_rates(1:length(bx)) - mean(firing_rates(1:length(bx))))'*by + ey;

x = x(data_max_time,1);
y = y(data_max_time,1);
end

end

% ---------- defined functions ----------

function min_trial_duration = findMinMaxTrialDuration(trial)
[num_trial, num_direc] = size(trial);

min_trial_duration = inf;
% find the min trial duration
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

min_trial_duration = findMinMaxTrialDuration(trial);

for j = 1:num_direc
    for i = 1:num_trial
        all_neuro_spikes_data = trial(i,j).spikes(:,:); % all_neuro_spikes is of (98 x spike_duration)
        num_neurons = size(trial(i,j).spikes,1);
%         each_spike_duration = size(all_neuro_spikes_data,2);
        binned_time = 1:bin_size:min_trial_duration + bin_size;
        binned_spikes = zeros(num_neurons,length(binned_time)-1);
        for n = 1:num_neurons
            spike_index = find(all_neuro_spikes_data(n, :) == 1);
            binned_spikes(n, :) = histcounts(spike_index, binned_time);
        end
        binned_firing_rates = binned_spikes*(1000/bin_size);


        binned_data(i, j).binned_spikes = binned_spikes;
        binned_data(i, j).binned_firing_rates = binned_firing_rates;
%         binned_data(i, j).binned_handPos = [binned_handPos_x; binned_handPos_y];
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
