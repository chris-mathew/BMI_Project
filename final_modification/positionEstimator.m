% Group Name: Cortex Crusaders
% Members: Xing Lim, Qian Tong Lim, Naiyira Hudaha Hussain Naweed, Christopher Mathew

function [x, y, modelParameters] = positionEstimator(past_current_trial, modelParameters)

    % SVM Prediction of Direction
    % referenced and modified from: https://uk.mathworks.com/help/stats/classreg.learning.classif.compactclassificationsvm.predict.html
    t_length = 320;
    X_test = mean(past_current_trial.spikes(:, 1:t_length), 2)';
    
    % Get the confidence scores from each SVM
    svm_confidences = zeros(1, length(modelParameters.svmModel));
    for numSvm = 1:length(modelParameters.svmModel)
        svm_confidences(numSvm) = SVMPred(modelParameters.svmModel{numSvm}, X_test);
    end
    
    % Use the confidence scores to predict the class
    pred_direc = determineClass(svm_confidences);
    modelParameters.pred_direc = pred_direc;
    
    % parameter initialization
    bin_interval = 20;
    time_start = 320;
    binned_data = get_binned_firing_rates(past_current_trial, bin_interval);
    
    % getting current time bin index from current data
    [~, current_trial_length] = size(past_current_trial.spikes);
    test_time_bin = (current_trial_length/bin_interval) - (time_start/bin_interval)+1;
         
    training_max_timebin = modelParameters.max_timebin_index; % max time bin from training dataset
    training_largest_timebin = training_max_timebin - (time_start/bin_interval); % largest index for pcr parameters gotten from training
    training_spike_length = training_max_timebin*bin_interval; % limit of spike length used in training
    
    % check if testing input length exceed training data length
    if current_trial_length <= training_spike_length
    
        firing_rates = reshape(binned_data.binned_firing_rates,[],1);
        
        % extract stored parameter based on current spike length and
        % predicted direction from SVM
        bx = modelParameters.pcr(pred_direc,test_time_bin).bx;
        by = modelParameters.pcr(pred_direc,test_time_bin).by;
        ex = modelParameters.pcr(pred_direc,test_time_bin).ex(:,pred_direc);
        ey = modelParameters.pcr(pred_direc,test_time_bin).ey(:,pred_direc);
        mean_firing = modelParameters.pcr(pred_direc,test_time_bin).mean_firing;
        
        % predict x and y hand position from pcr model 
        x = (firing_rates-mean(mean_firing))'*bx + ex;
        y = (firing_rates-mean(mean_firing))'*by + ey;
        x = x(current_trial_length,1);
        y = y(current_trial_length,1);

        % storing current predicted direction for if input data is longer
        % than training data
        modelParameters.pred_direc = pred_direc;
    
    elseif current_trial_length > training_spike_length
    
        firing_rates = reshape(binned_data.binned_firing_rates,[],1);
        
        % extract stored parameter and direction based on last index from 
        % largest training spike length
        bx = modelParameters.pcr(pred_direc,training_largest_timebin).bx;
        by = modelParameters.pcr(pred_direc,training_largest_timebin).by;
        ex = modelParameters.pcr(pred_direc,training_largest_timebin).ex(:,pred_direc);
        ey = modelParameters.pcr(pred_direc,training_largest_timebin).ey(:,pred_direc);

        % predict x and y hand position from pcr model 
        x = (firing_rates(1:length(bx)) - mean(firing_rates(1:length(bx))))'*bx + ex;
        y = (firing_rates(1:length(bx)) - mean(firing_rates(1:length(bx))))'*by + ey;
        x = x(training_spike_length,1);
        y = y(training_spike_length,1);
    end
end

% ========================= defined functions =========================
function y_pred = determineClass(svm_confidences)
    % find the class with the highest confidence
    [maxConfidence, y_pred] = max(svm_confidences);

    % for cases where multiple classes have the same highest confidence
    if sum(svm_confidences == maxConfidence) > 1
        candidates = find(svm_confidences == maxConfidence);
        y_pred = candidates(randi(length(candidates)));
    end
end

function confidence = SVMPred(model_param, X)
    % compute  kernel between X and each support vector
    K = zeros(size(X, 1), size(model_param.X, 1));
    for i = 1:size(X, 1)
        for j = 1:size(model_param.X, 1)
            distanceSquared = sum((X(i, :) - model_param.X(j, :)).^2);
            K(i, j) = exp(-model_param.gamma * distanceSquared);
        end
    end
    %  confidence score is the decision function value
    confidence = (K * (model_param.alpha .* model_param.y)) + model_param.b;
end


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



function binned_data = get_binned_firing_rates(trial, bin_interval)
binned_data = struct;
[num_trial, num_direc] = size(trial);

min_trial_duration = findMinMaxTrialDuration(trial);

for j = 1:num_direc
    for i = 1:num_trial
        all_neuro_spikes_data = trial(i,j).spikes(:,:); % all_neuro_spikes is of (98 x spike_duration)
        num_neurons = size(trial(i,j).spikes,1);
%         each_spike_duration = size(all_neuro_spikes_data,2);
        binned_time = 1:bin_interval:min_trial_duration + bin_interval;
        binned_spikes = zeros(num_neurons,length(binned_time)-1);
        for n = 1:num_neurons
            spike_index = find(all_neuro_spikes_data(n, :) == 1);
            binned_spikes(n, :) = histcounts(spike_index, binned_time);
        end
        binned_firing_rates = binned_spikes*(1000/bin_interval);


        binned_data(i, j).binned_spikes = binned_spikes;
        binned_data(i, j).binned_firing_rates = binned_firing_rates;
    end
end 
end


