% MEC E 686 - Assignment 3
% Sina Kazemi
%% Main
clear all; close all; clc

% From Assignment 1 and 2
DataA1 = load('DataA1.mat');
DataA2 = load('DataA2.mat');
DataA2.Data.P1.Ensemble = DataA1.Data.P1.Ensemble;
DataA2.Data.P1.FilteredEnsemble = DataA1.Data.P1.FilteredEnsemble;
DataA2.Data.P2.Ensemble = DataA1.Data.P2.Ensemble;
DataA2.Data.P3.Ensemble = DataA1.Data.P3.Ensemble;
Data = DataA2.Data;
clear DataA1 DataA2

Data = process_and_downsample_EMG(Data);      % A3 Q1, Q2, Q3
Data = extract_EMG_gait_cycle(Data);          % A3 Q4
Data = normalize_EMG_time(Data);              % A3 Q5
plot_muscle_activation_and_kinematics(Data);  % A3 Q6

save('DataA3.mat', 'Data')

%% Functions

function Data = process_and_downsample_EMG(Data)
% A3 Q1, Q2, Q3: Loads raw EMG, processes it (filter, demean, rectify), calculates quiet standing SD, and downsamples to 100 Hz

    muscles = {'RF', 'BF', 'TA', 'LG'};
    fs_emg = 2000; % Raw EMG fs

    % Q1
    % High-pass Butterworth (4th order, 10 Hz cut-off)
    [b_hp, a_hp] = butter(4, 10 / (fs_emg / 2), 'high');
    % Low-pass Butterworth (4th order, 500 Hz cut-off)
    [b_lp, a_lp] = butter(4, 500 / (fs_emg / 2), 'low');

    for p = 1:3
        p_name = sprintf('P%d', p);
        for t = 1:6
            t_name = sprintf('T%d', t);
            
            filename = sprintf('Data\\FPEMG_P_%d_T_%d.txt', p, t);
            if ~isfile(filename)
                warning('FPEMG file not found: %s', filename);
                continue;
            end

            % Read raw text file
            col_names = {'Frame','SubFrame','Fx','Fy','Fz','Mx','My','Mz', ...
                         'Cx','Cy','Cz','RF','BF','TA','LG'};
            opts = delimitedTextImportOptions( ...
                'NumVariables',     numel(col_names), ...
                'Delimiter',        '\t', ...
                'DataLines',        6, ...          
                'VariableNames',    col_names, ...
                'VariableTypes',    repmat({'double'}, 1, numel(col_names)));
            
            raw_data = readtable(filename, opts);

            ProcessedEMG = struct();
            QuietSD = struct();

            for m = 1:length(muscles)
                m_name = muscles{m};
                raw_emg = raw_data.(m_name);

                % Q1
                emg_hp = filtfilt(b_hp, a_hp, raw_emg);
                emg_lp = filtfilt(b_lp, a_lp, emg_hp);
                emg_demeaned = emg_lp - mean(emg_lp);
                emg_rectified = abs(emg_demeaned);

                % Q2
                QuietSD.(m_name) = std(emg_rectified(1:fs_emg));

                % Q3
                num_mocap_frames = Data.(p_name).(t_name).NumFr;
                indices = 1:20:height(raw_data);
                
                if length(indices) > num_mocap_frames
                    indices = indices(1:num_mocap_frames);
                end
                
                ProcessedEMG.(m_name) = emg_rectified(indices);
            end

            % Store the results
            Data.(p_name).(t_name).EMG.Processed = ProcessedEMG;
            Data.(p_name).(t_name).EMG.QuietSD = QuietSD;
            
            fprintf('Processed EMG for %s %s\n', p_name, t_name);
        end
    end
    fprintf('Q1, Q2, Q3: EMG processing, SD calculation, and downsampling complete.\n');
end

function Data = extract_EMG_gait_cycle(Data)
% A3 Q4: Extracts the processed EMG data for the force plate gait cycle, ONLY for trials where the Right foot landed on the plate 

    muscles = {'RF', 'BF', 'TA', 'LG'};

    for p = fieldnames(Data)'
        p_name = p{1};
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};

            if ~isfield(Data.(p_name).(t_name), 'GaitEvents'); continue; end
            
            GE = Data.(p_name).(t_name).GaitEvents;
            
            % Extract the RIGHT foot
            if ~strcmp(GE.Side, 'Right') || isnan(GE.HS_FP)
                continue; 
            end

            s = GE.HS_FP; 
            e = GE.HS_Next; 

            ExtractedEMG = struct();
            
            for m = 1:length(muscles)
                m_name = muscles{m};
                
                % Grab the EMG array from Q3
                full_emg = Data.(p_name).(t_name).EMG.Processed.(m_name);
                ExtractedEMG.(m_name) = full_emg(s:e);
            end
            
            % Store the data
            Data.(p_name).(t_name).EMG.Extracted = ExtractedEMG;
            
            fprintf('Extracted EMG for %s %s (Right side, frames %d to %d)\n', ...
                p_name, t_name, s, e);
        end
    end
    fprintf('Q4: Extracted EMG for Right-foot gait cycles.\n');
end

function Data = normalize_EMG_time(Data)
% A3 Q5: Time normalizes the extracted EMG time series to 101 samples

    muscles = {'RF', 'BF', 'TA', 'LG'};

    for p = fieldnames(Data)'
        p_name = p{1};
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};

            if ~isfield(Data.(p_name).(t_name), 'EMG') || ...
               ~isfield(Data.(p_name).(t_name).EMG, 'Extracted')
                continue; 
            end
            
            Ex = Data.(p_name).(t_name).EMG.Extracted;
            NormalizedEMG = struct();
            
            for m = 1:length(muscles)
                m_name = muscles{m};
                raw_array = Ex.(m_name);
                raw_array = raw_array(:); 

                % Normalize to 101 samples
                NormalizedEMG.(m_name) = nrm(raw_array);
            end
            
            Data.(p_name).(t_name).EMG.Normalized = NormalizedEMG;
        end
    end
    fprintf('Q5: Time-normalized extracted EMG to 101 samples.\n');
end

function plot_muscle_activation_and_kinematics(Data)
% A3 Q6: Generates a 3x1 figure per participant
% Top/Middle: Ensemble average Knee and Ankle F/E kinematics
% Bottom: Visual representation of muscle activity exceeding 3x Quiet SD

    muscles = {'RF', 'BF', 'TA', 'LG'};
    % Color map : T1=Red, T2=Green, T3=Blue
    colors = {'r', 'g', 'b', 'c', 'm', 'k'}; 
    
    for p = fieldnames(Data)'
        p_name = p{1};
        
        if ~isfield(Data.(p_name), 'Ensemble'); continue; end
        
        fig = figure('Name', sprintf('%s - Muscle Activation and Kinematics (Q6)', p_name), ...
            'Color', 'w', 'Units', 'normalized', 'Position', [0.2 0.1 0.6 0.8]);
        
        % TOP SUBPLOT: KNEE F/E
        subplot(3, 1, 1); hold on;
        ylabel('Knee F/E (deg)', 'FontWeight', 'bold');
        
        E_knee = Data.(p_name).Ensemble.Right.JointAngles.Knee; 
        plot_band_col(E_knee, 1, 'b');
        
        xlim([0 100]);
        set(gca, 'XTickLabel', []); 
        
        % MIDDLE SUBPLOT: ANKLE F/E
        subplot(3, 1, 2); hold on;
        ylabel('Ankle F/E (deg)', 'FontWeight', 'bold');
        
        E_ankle = Data.(p_name).Ensemble.Right.JointAngles.Ankle;
        plot_band_col(E_ankle, 1, 'b');
        
        xlim([0 100]);
        set(gca, 'XTickLabel', []);
        
        % BOTTOM SUBPLOT: EMG ACTIVATION
        subplot(3, 1, 3); hold on;
        xlabel('% Gait cycle', 'FontWeight', 'bold');
        
        yticks(1:4);
        yticklabels(muscles);
        ylim([0.5 4.8]);
        xlim([0 100]);
        
        valid_trial_count = 0;
        
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};
            
            if ~isfield(Data.(p_name).(t_name), 'EMG') || ...
               ~isfield(Data.(p_name).(t_name).EMG, 'Normalized')
                continue; 
            end
            
            valid_trial_count = valid_trial_count + 1;
            c = colors{mod(valid_trial_count-1, length(colors)) + 1};
            
            NormEMG = Data.(p_name).(t_name).EMG.Normalized;
            QuietSD = Data.(p_name).(t_name).EMG.QuietSD;
            
            x_time = linspace(0, 100, 101);
            
            for m = 1:length(muscles)
                m_name = muscles{m};
                
                % Calculate the threshold: 3 * Quiet Standing SD
                threshold = 3 * QuietSD.(m_name);
                
                % Find indices where muscle is active
                active_idx = NormEMG.(m_name) > threshold;
                
                % Base Y: RF=1, BF=2, TA=3, LG=4
                y_base = m;
                % T1 is below, T2 is middle, T3 is above
                y_offset = (valid_trial_count - 2) * 0.25; 
                y_plot = y_base + y_offset;
                
                % Plot active regions as dense square scatter points
                y_line = NaN(1, length(x_time)); % Create empty array
                y_line(active_idx) = y_plot;     % Fill only active frames
                
                plot(x_time, y_line, 'Color', c, 'LineWidth', 6);
            end
        end
        
        % ADD CUSTOM LEGEND
        h = zeros(valid_trial_count, 1);
        leg_labels = cell(valid_trial_count, 1);
        for i = 1:valid_trial_count
            h(i) = plot(nan, nan, 's', 'MarkerFaceColor', colors{i}, ...
                'MarkerEdgeColor', 'none', 'MarkerSize', 8);
            leg_labels{i} = sprintf('Trial %d', i);
        end
        legend(h, leg_labels, 'Location', 'northeast', 'Box', 'off');
    end
    fprintf('Q6: Generated all Muscle Activation and Kinematics figures.\n');
end

%% Core Math Helpers

function nd = nrm(raw)
% Linear interpolation to 101 samples
nd = interp1(1:size(raw,1), raw, linspace(1,size(raw,1),101)');
end


function plot_band(S, col)
% Plots mean +/-1 SD shaded band from a stats struct with Mean and SD fields
if isempty(S.Mean); return; end
mu=S.Mean(:)'; sd=S.SD(:)';
x=linspace(0,100,numel(mu)); v=~isnan(mu)&~isnan(sd);
fc=struct('r',[1 .8 .8],'b',[.8 .8 1]);
fill([x(v),fliplr(x(v))],[mu(v)+sd(v),fliplr(mu(v)-sd(v))], ...
    fc.(col),'EdgeColor','none','FaceAlpha',0.5);
plot(x, mu, col, 'LineWidth', 2);
end

function plot_band_col(S, col_idx, color)
% Plots one angle column (col_idx) of a 3D JCS stats struct as a shaded band
if isempty(S.Mean); return; end
plot_band(struct('Mean',S.Mean(:,col_idx),'SD',S.SD(:,col_idx)), color);
end
