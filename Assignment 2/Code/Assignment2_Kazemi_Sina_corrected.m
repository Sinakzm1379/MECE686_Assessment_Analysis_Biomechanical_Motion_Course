% MEC E 686 - Assignment 1
% Sina Kazemi
%% Main
clear all; close all; clc

% From Assignment 1
load('Data.mat')
Data.P1 = rmfield(Data.P1,{'Ensemble','FilteredEnsemble'});
Data.P2 = rmfield(Data.P2,'Ensemble');
Data.P3 = rmfield(Data.P3,'Ensemble');

Data = load_and_downsample_kinetics(Data);    % A2 Q1
Data = calculate_JCR(Data);                   % A2 Q2
Data = calculate_COM(Data);                   % A2 Q3
Data = extract_gait_cycle_data(Data);         % A2 Q4
Data = filter_extracted_data(Data);           % A2 Q5
Data = calculate_kinematics(Data);            % A2 Q6
Data = calculate_inverse_dynamics(Data);      % A2 Q7
Data = time_normalize_kinetics(Data);         % A2 Q8
Data = calculate_ensemble_kinetics(Data);     % A2 Q9
plot_inverse_dynamics_ensemble(Data,'Local');% A2 Q10

% save('DataA2.mat', 'Data')

%% Functions

function Data = load_and_downsample_kinetics(Data)
% Q1: Loads kinetic data (GRF, COP, Tz) from FPEMG files and downsamples
% from 2000 Hz to 100 Hz to synchronize with the motion capture data

    for p = 1:3
        p_name = sprintf('P%d', p);
        for t = 1:6
            t_name = sprintf('T%d', t);
            filename = sprintf('Data\\FPEMG_P_%d_T_%d.txt', p, t);

            if ~isfile(filename)
                warning('Kinetic file not found: %s', filename);
                continue;
            end

            col_names = {'Frame','SubFrame','Fx','Fy','Fz','Mx','My','Mz', ...
                         'Cx','Cy','Cz','RF','BF','TA','LG'};

            opts = delimitedTextImportOptions( ...
                'NumVariables',     numel(col_names), ...
                'Delimiter',        '\t', ...
                'DataLines',        6, ...          % data starts at row 6
                'VariableNames',    col_names, ...
                'VariableTypes',    repmat({'double'}, 1, numel(col_names)));

            raw_data = readtable(filename, opts);

            % Downsample: 2000 Hz → 100 Hz (keep every 20th row)
            num_mocap_frames = Data.(p_name).(t_name).NumFr;
            indices = 1:20:height(raw_data);

            if length(indices) > num_mocap_frames
                indices = indices(1:num_mocap_frames);
            end

            % Pack into Kinetics struct
            Kin.GRF = [raw_data.Fx(indices), raw_data.Fy(indices), raw_data.Fz(indices)];
            Kin.COP = [raw_data.Cx(indices), raw_data.Cy(indices), raw_data.Cz(indices)];
            Kin.Tz  =  raw_data.Mz(indices);

            Data.(p_name).(t_name).Kinetics = Kin;

            fprintf('Loaded and downsampled kinetics: %s %s (%d frames)\n', ...
                    p_name, t_name, length(indices));
        end
    end
end

function Data = calculate_JCR(Data)
% Q2: Calculates Joint Centers of Rotation.
% JCR_ground = midpoint of MH5+MH1 markers (guide slide 4)
% COP is stored separately in Kinetics and used only in the moment eq.

    for p = fieldnames(Data)'
        p_name = p{1};
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};

            MK  = Data.(p_name).(t_name).MK;
            Kin = Data.(p_name).(t_name).Kinetics;
            LCS = Data.(p_name).(t_name).LCS;
            JCR = struct();

            % RIGHT SIDE
            % Convert mm -> m
            JCR.Right.Ground = (MK.RMH5 + MK.RMH1) / 2 / 1000;
            JCR.Right.Ankle  = (MK.RLM  + MK.RMM)  / 2 / 1000;
            JCR.Right.Knee   = (MK.RLE  + MK.RME)  / 2 / 1000;
            JCR.Right.Hip    = calc_hip_jcr(MK, LCS, 'Right');

            % LEFT SIDE
            JCR.Left.Ground  = (MK.LMH5 + MK.LMH1) / 2 / 1000;
            JCR.Left.Ankle   = (MK.LLM  + MK.LMM)  / 2 / 1000;
            JCR.Left.Knee    = (MK.LLE  + MK.LME)  / 2 / 1000;
            JCR.Left.Hip     = calc_hip_jcr(MK, LCS, 'Left');

            Data.(p_name).(t_name).JCR = JCR;
        end
    end
    fprintf('Q2: JCR_ground now uses MH5+MH1 midpoint (not COP).\n');
end
function hip_jcr = calc_hip_jcr(MK, LCS, side)
% Gordon et al. hip JCR estimation using pelvic landmarks
    N = size(MK.LASIS, 1);
    RASIS = MK.RASIS / 1000;
    LASIS = MK.LASIS / 1000;
    P_pelvis_origin = (RASIS + LASIS) / 2;
    hip_jcr = zeros(N, 3);
    for i = 1:N
        R_GP = LCS.Pelvis(:,:,i);
        diff_global = (RASIS(i,:) - LASIS(i,:))';
        diff_pelv   = R_GP * diff_global;
        PW = norm(diff_pelv);
        if strcmp(side, 'Right')
            offset_pelv = [-0.19*PW; -0.30*PW;  0.36*PW];
        else
            offset_pelv = [-0.19*PW; -0.30*PW; -0.36*PW];
        end
        offset_global = R_GP' * offset_pelv;
        hip_jcr(i,:) = P_pelvis_origin(i,:) + offset_global';
    end
end

function Data = calculate_COM(Data)
% Q3: COM of foot, shank, thigh using guide slide 6 formulas.

    for p = fieldnames(Data)'
        p_name = p{1};
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};

            JCR = Data.(p_name).(t_name).JCR;
            COM = struct();

            for side = {'Right', 'Left'}
                s = side{1};
                % Foot
                COM.(s).Foot  = JCR.(s).Ground + 0.500 * (JCR.(s).Ankle - JCR.(s).Ground);
                % Shank
                COM.(s).Shank = JCR.(s).Ankle  + 0.567 * (JCR.(s).Knee  - JCR.(s).Ankle);
                % Thigh
                COM.(s).Thigh = JCR.(s).Knee   + 0.567 * (JCR.(s).Hip   - JCR.(s).Knee);
            end

            Data.(p_name).(t_name).COM = COM;
        end
    end
    fprintf('Q3: COMs calculated correctly (foot uses MH-midpoint ground JCR).\n');
end

function Data = extract_gait_cycle_data(Data)
% Q4: Extracts the GRF, COP, Tz, and COM

    for p = fieldnames(Data)'
        p_name = p{1};
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};
            
            % Skip if gait events weren't calculated or no valid strike
            if ~isfield(Data.(p_name).(t_name), 'GaitEvents'); continue; end
            GE = Data.(p_name).(t_name).GaitEvents;
            if strcmp(GE.Side, 'None') || isnan(GE.HS_FP); continue; end
            
            % Define start and end frames
            s = GE.HS_FP; 
            e = GE.HS_Next; 
            side = GE.Side; % 'Right' or 'Left'
            
            Kin = Data.(p_name).(t_name).Kinetics;
            COM = Data.(p_name).(t_name).COM;
            
            Ex = struct();
            Ex.Side = side;
            
            % 1. Extract Kinetics (GRF, COP, Tz)
            Ex.Kinetics.GRF = Kin.GRF(s:e, :);
            Ex.Kinetics.COP = Kin.COP(s:e, :);
            Ex.Kinetics.Tz  = Kin.Tz(s:e);
            
            % 2. Extract COM for the landing side ONLY
            Ex.COM.Thigh = COM.(side).Thigh(s:e, :);
            Ex.COM.Shank = COM.(side).Shank(s:e, :);
            Ex.COM.Foot  = COM.(side).Foot(s:e, :);
            
            Data.(p_name).(t_name).Extracted = Ex;
            
            fprintf('Extracted %s %s: %s side, frames %d to %d\n', ...
                p_name, t_name, side, s, e);
        end
    end
end

function Data = filter_extracted_data(Data)
% Q5: Filters the data using a 4th-order zero phase-shift, low-pass Butterworth filter with a 10 Hz cut-off

    fc = 10; % Cut-off frequency in Hz
    fo = 4;  % Filter order

    for p = fieldnames(Data)'
        p_name = p{1};
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};
            
            if ~isfield(Data.(p_name).(t_name), 'Extracted'); continue; end
            
            Ex = Data.(p_name).(t_name).Extracted;
            fs = Data.(p_name).(t_name).Freq;
            
            Wn = fc / (0.5 * fs);
            [b, a] = butter(fo, Wn, 'low');
            
            Filt = struct();
            Filt.Side = Ex.Side;
            
            % Filter Kinetics
            Filt.Kinetics.GRF = filtfilt(b, a, Ex.Kinetics.GRF);
            Filt.Kinetics.COP = filtfilt(b, a, Ex.Kinetics.COP);
            Filt.Kinetics.Tz  = filtfilt(b, a, Ex.Kinetics.Tz);
            
            % Filter COM
            Filt.COM.Thigh = filtfilt(b, a, Ex.COM.Thigh);
            Filt.COM.Shank = filtfilt(b, a, Ex.COM.Shank);
            Filt.COM.Foot  = filtfilt(b, a, Ex.COM.Foot);
            
            Data.(p_name).(t_name).Filtered = Filt;
        end
    end
    fprintf('Q5: Filtered GRF, COP, Tz, and COM (10 Hz low-pass Butterworth).\n');
end

function Data = calculate_kinematics(Data)

    for p = fieldnames(Data)'
        p_name = p{1};
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};

            if ~isfield(Data.(p_name).(t_name), 'Filtered'); continue; end

            Filt = Data.(p_name).(t_name).Filtered;
            GE   = Data.(p_name).(t_name).GaitEvents;
            fs   = Data.(p_name).(t_name).Freq;

            s = GE.HS_FP;
            e = GE.HS_Next;
            k = GE.Side(1);  % 'Right' or 'Left'

            Kinematics = struct();
            segments = {'Thigh', 'Shank', 'Foot'};

            for i = 1:length(segments)
                seg = segments{i};

                % Linear acceleration (global frame)
                pos_COM = Filt.COM.(seg);
                V = diff(pos_COM, 1, 1) * fs;
                V = [V; V(end,:)];
                V = sgolayfilt(V, 3, 11);
                a = diff(V, 1, 1) * fs;
                a = [a; a(end,:)];
                a = sgolayfilt(a, 3, 11);
                Kinematics.(seg).a_COM_global = a;

                % Angular kinematics via spin matrix
                LCS_full  = Data.(p_name).(t_name).LCS.([seg '_' k]);
                LCS_slice = LCS_full(:,:,s:e);  % 3x3xN, Global-to-Local
                N_fr = size(LCS_slice, 3);

                omega_local  = zeros(N_fr, 3);
                omega_global = zeros(N_fr, 3);

                for ii = 2:N_fr
                    R_prev = LCS_slice(:,:,ii-1);  % Global-to-Local at t-1
                    R_curr = LCS_slice(:,:,ii);    % Global-to-Local at t
                    
                    S_global = fs * (eye(3) - R_prev' * R_curr); 
                    
                    % Extract global angular velocity directly from the global skew matrix
                    w_glob = [S_global(3,2); S_global(1,3); S_global(2,1)];
                    omega_global(ii,:) = w_glob';
                    
                    % Rotate to local frame for Newton-Euler (I*alpha + w x Iw)
                    omega_local(ii,:) = (R_curr * w_glob)';
                end
                omega_local(1,:)  = omega_local(2,:);
                omega_global(1,:) = omega_global(2,:);

                % Smooth
                omega_local  = sgolayfilt(omega_local,  3, 11);
                omega_global = sgolayfilt(omega_global, 3, 11);

                % Angular acceleration in LOCAL frame
                alpha_local = diff(omega_local, 1, 1) * fs;
                alpha_local = [alpha_local; alpha_local(end,:)];
                alpha_local = sgolayfilt(alpha_local, 3, 11);

                Kinematics.(seg).w_local      = omega_local;
                Kinematics.(seg).a_local      = alpha_local;
                Kinematics.(seg).omega_global = omega_global;
            end

            Data.(p_name).(t_name).Kinematics = Kinematics;
        end
    end
    fprintf('Q6: Kinematics done (LCS=Global-to-Local, omega_global = R''*omega_local).\n');
end

function Data = calculate_inverse_dynamics(Data)

    masses = struct('P1', 76.5, 'P2', 65.8, 'P3', 82.7);
    g = [0, 0, -9.81];

    for p = fieldnames(Data)'
        p_name = p{1};
        BM = masses.(p_name);

        for t = fieldnames(Data.(p_name))'
            t_name = t{1};
            if ~isfield(Data.(p_name).(t_name), 'Kinematics'); continue; end

            GE   = Data.(p_name).(t_name).GaitEvents;
            s    = GE.HS_FP;
            e    = GE.HS_Next;
            side = GE.Side;
            k    = side(1);

            Kin = Data.(p_name).(t_name).Filtered.Kinetics;
            COM = Data.(p_name).(t_name).Filtered.COM;
            JCR = Data.(p_name).(t_name).JCR.(side);

            % JCR_ground = MH midpoint in metres (set in Q2)
            JCR_slice = struct( ...
                'Ground', JCR.Ground(s:e,:), ...
                'Ankle',  JCR.Ankle(s:e,:),  ...
                'Knee',   JCR.Knee(s:e,:),   ...
                'Hip',    JCR.Hip(s:e,:));

            m_foot  = 0.0145 * BM;
            m_shank = 0.0465 * BM;
            m_thigh = 0.1000 * BM;

            L_foot  = mean(vecnorm(JCR_slice.Ankle - JCR_slice.Ground, 2, 2));
            L_shank = mean(vecnorm(JCR_slice.Knee  - JCR_slice.Ankle,  2, 2));
            L_thigh = mean(vecnorm(JCR_slice.Hip   - JCR_slice.Knee,   2, 2));

            I_foot  = build_inertia(m_foot,  L_foot,  'Foot');
            I_shank = build_inertia(m_shank, L_shank, 'Shank');
            I_thigh = build_inertia(m_thigh, L_thigh, 'Thigh');

            N = e - s + 1;
            F_ankle = zeros(N,3); M_ankle = zeros(N,3); P_ankle = zeros(N,1);
            F_knee  = zeros(N,3); M_knee  = zeros(N,3); P_knee  = zeros(N,1);
            F_hip   = zeros(N,3); M_hip   = zeros(N,3); P_hip   = zeros(N,1);

            % Pelvis angular velocity (Global-to-Local convention)
            LCS_pelvis = Data.(p_name).(t_name).LCS.Pelvis(:,:,s:e);
            N_fr = size(LCS_pelvis, 3);
            fs   = Data.(p_name).(t_name).Freq;
            w_pelvis_global = zeros(N_fr, 3);
            for ii = 2:N_fr
                R_prev = LCS_pelvis(:,:,ii-1);
                R_curr = LCS_pelvis(:,:,ii);
                
                S_global = fs * (eye(3) - R_prev' * R_curr);
                w_glob = [S_global(3,2); S_global(1,3); S_global(2,1)];
                
                w_pelvis_global(ii,:) = w_glob';
            end
            w_pelvis_global(1,:) = w_pelvis_global(2,:);
            w_pelvis_global = sgolayfilt(w_pelvis_global, 3, 11);

            w_glob_foot  = Data.(p_name).(t_name).Kinematics.Foot.omega_global;
            w_glob_shank = Data.(p_name).(t_name).Kinematics.Shank.omega_global;
            w_glob_thigh = Data.(p_name).(t_name).Kinematics.Thigh.omega_global;

            for i = 1:N
                a_foot  = Data.(p_name).(t_name).Kinematics.Foot.a_COM_global(i,:);
                a_shank = Data.(p_name).(t_name).Kinematics.Shank.a_COM_global(i,:);
                a_thigh = Data.(p_name).(t_name).Kinematics.Thigh.a_COM_global(i,:);

                F_grf = Kin.GRF(i,:);
                T_z   = [0, 0, Kin.Tz(i)/1000];   % N·mm -> N·m
                P_COP = Kin.COP(i,:) / 1000;      % mm -> m

                % LCS is Global-to-Local
                R_foot  = Data.(p_name).(t_name).LCS.(['Foot_'  k])(:,:,s+i-1);
                R_shank = Data.(p_name).(t_name).LCS.(['Shank_' k])(:,:,s+i-1);
                R_thigh = Data.(p_name).(t_name).LCS.(['Thigh_' k])(:,:,s+i-1);

                % Segment moment in LOCAL frame
                w_f = Data.(p_name).(t_name).Kinematics.Foot.w_local(i,:)';
                a_f = Data.(p_name).(t_name).Kinematics.Foot.a_local(i,:)';
                M_seg_foot_local  = I_foot  * a_f + cross(w_f, I_foot  * w_f);
                M_seg_foot  = R_foot'  * M_seg_foot_local;   % local -> global

                w_s = Data.(p_name).(t_name).Kinematics.Shank.w_local(i,:)';
                a_s = Data.(p_name).(t_name).Kinematics.Shank.a_local(i,:)';
                M_seg_shank_local = I_shank * a_s + cross(w_s, I_shank * w_s);
                M_seg_shank = R_shank' * M_seg_shank_local;  % local -> global

                w_t = Data.(p_name).(t_name).Kinematics.Thigh.w_local(i,:)';
                a_t = Data.(p_name).(t_name).Kinematics.Thigh.a_local(i,:)';
                M_seg_thigh_local = I_thigh * a_t + cross(w_t, I_thigh * w_t);
                M_seg_thigh = R_thigh' * M_seg_thigh_local;  % local -> global

                P_com_f = COM.Foot(i,:);
                P_com_s = COM.Shank(i,:);
                P_com_t = COM.Thigh(i,:);
                P_jcr_a = JCR_slice.Ankle(i,:);
                P_jcr_k = JCR_slice.Knee(i,:);
                P_jcr_h = JCR_slice.Hip(i,:);

                % GRF less than 20 N, the foot is off the plate
                if F_grf(3) < 20
                    F_grf = [0, 0, 0];
                    T_z   = [0, 0, 0];
                    P_COP = P_com_f; % Sets the lever arm to 0 to neutralize the cross product
                end
                
                % ANKLE
                F_ankle(i,:) = m_foot*(a_foot - g) + F_grf;
                M_ankle(i,:) = M_seg_foot' + T_z + ...
                    cross(P_com_f - P_jcr_a, F_ankle(i,:)) + ...
                    cross(P_COP   - P_com_f, F_grf);
                P_ankle(i) = dot(M_ankle(i,:), w_glob_shank(i,:) - w_glob_foot(i,:));

                % KNEE
                F_knee(i,:) = m_shank*(a_shank - g) + F_ankle(i,:);
                M_knee(i,:) = M_ankle(i,:) + M_seg_shank' + ...
                    cross(P_com_s - P_jcr_k, F_knee(i,:)) + ...
                    cross(P_jcr_a - P_com_s, F_ankle(i,:));
                P_knee(i) = dot(M_knee(i,:), w_glob_shank(i,:) - w_glob_thigh(i,:));

                % HIP
                F_hip(i,:) = m_thigh*(a_thigh - g) + F_knee(i,:);
                M_hip(i,:) = M_knee(i,:) + M_seg_thigh' + ...
                    cross(P_com_t - P_jcr_h, F_hip(i,:)) + ...
                    cross(P_jcr_k - P_com_t, F_knee(i,:));
                P_hip(i) = dot(M_hip(i,:), w_pelvis_global(i,:) - w_glob_thigh(i,:));
            end

            ID.Ankle.Force = F_ankle; ID.Ankle.Moment = M_ankle; ID.Ankle.Power = P_ankle;
            ID.Knee.Force  = F_knee;  ID.Knee.Moment  = M_knee;  ID.Knee.Power  = P_knee;
            ID.Hip.Force   = F_hip;   ID.Hip.Moment   = M_hip;   ID.Hip.Power   = P_hip;

            Data.(p_name).(t_name).InverseDynamics = ID;
        end
    end
    fprintf('Q7: Inverse Dynamics done (R''=Local-to-Global for all rotations).\n');
end
function I = build_inertia(m, L, seg_type, R_dumas)
    % R_dumas: Rotation matrix from Dumas frame to the Local Anatomical Frame
    % If not provided, defaults to Identity (no rotation)
    if nargin < 4
        R_dumas = eye(3); 
    end

    switch seg_type
        case 'Foot'
            Ixx=(0.17*L)^2; Iyy=(0.37*L)^2; Izz=(0.36*L)^2;
            Ixy=(0.13*L)^2; Ixz=-(0.08*L)^2; Iyz=0;
        case 'Shank'
            Ixx=(0.28*L)^2; Iyy=(0.10*L)^2; Izz=(0.28*L)^2;
            Ixy=-(0.04*L)^2; Ixz=-(0.02*L)^2; Iyz=(0.05*L)^2;
        case 'Thigh'
            Ixx=(0.29*L)^2; Iyy=(0.15*L)^2; Izz=(0.30*L)^2;
            Ixy=(0.07*L)^2; Ixz=-(0.02*L)^2; Iyz=-(0.07*L)^2;
    end
    
    % The raw Dumas inertia matrix
    I_matrix = [Ixx, Ixy, Ixz; 
                Ixy, Iyy, Iyz; 
                Ixz, Iyz, Izz];
                
    I = m * (R_dumas * I_matrix * R_dumas');
end

function Data = time_normalize_kinetics(Data)
% Q8: Time normalizes the inverse dynamics results

    for p = fieldnames(Data)'
        p_name = p{1};
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};
            
            if ~isfield(Data.(p_name).(t_name), 'InverseDynamics'); continue; end
            
            ID = Data.(p_name).(t_name).InverseDynamics;
            NormID = struct();
            
            % Carry over the landing side for later ensemble averaging
            NormID.Side = Data.(p_name).(t_name).Extracted.Side;
            
            joints = {'Ankle', 'Knee', 'Hip'};
            
            for j = 1:length(joints)
                jn = joints{j};
                
                % Normalize Force (N x 3 -> 101 x 3)
                NormID.(jn).Force  = nrm(ID.(jn).Force);
                
                % Normalize Moment (N x 3 -> 101 x 3)
                NormID.(jn).Moment = nrm(ID.(jn).Moment);
                
                % Normalize Power (N x 1 -> 101 x 1)
                NormID.(jn).Power  = nrm(ID.(jn).Power);
            end
            
            Data.(p_name).(t_name).NormalizedID = NormID;
        end
    end
    fprintf('Q8: Time-normalized joint forces, moments, and powers to 101 samples.\n');
end

function Data = calculate_ensemble_kinetics(Data)
% Q9: Calculates the ensemble average (mean and SD)

    joints = {'Ankle', 'Knee', 'Hip'};
    prox_segments = {'Shank', 'Thigh', 'Pelvis'};
    
    for p = fieldnames(Data)'
        p_name = p{1};
        
        % Initialize accumulators for Left and Right trials
        Acc = struct();
        for side = {'Right', 'Left'}
            s_name = side{1};
            for j = 1:3
                jn = joints{j};
                Acc.(s_name).Global.Force.(jn)  = [];
                Acc.(s_name).Global.Moment.(jn) = [];
                Acc.(s_name).Local.Force.(jn)   = [];
                Acc.(s_name).Local.Moment.(jn)  = [];
                Acc.(s_name).Power.(jn)         = [];
            end
        end
        
        % Loop through all trials to gather data
        for t = fieldnames(Data.(p_name))'
            t_name = t{1};
            if ~isfield(Data.(p_name).(t_name), 'NormalizedID'); continue; end
            
            GE = Data.(p_name).(t_name).GaitEvents;
            s = GE.HS_FP; 
            e = GE.HS_Next;
            k = GE.Side(1); % 'R' or 'L'
            side = GE.Side;
            
            NormID = Data.(p_name).(t_name).NormalizedID;
            RawID  = Data.(p_name).(t_name).InverseDynamics;
            LCS    = Data.(p_name).(t_name).LCS;
            
            for j = 1:3
                jn = joints{j};
                pn = prox_segments{j};
                
                % 1. Accumulate Global Data (Already Normalized in Q8)
                % Stack 101x3 arrays along the 3rd dimension
                Acc.(side).Global.Force.(jn)  = cat(3, Acc.(side).Global.Force.(jn),  NormID.(jn).Force);
                Acc.(side).Global.Moment.(jn) = cat(3, Acc.(side).Global.Moment.(jn), NormID.(jn).Moment);
                % Stack 101x1 arrays along the 2nd dimension
                Acc.(side).Power.(jn) = [Acc.(side).Power.(jn), NormID.(jn).Power]; 
                
                % 2. Calculate Local Data (Raw -> Rotate -> Normalize -> Accumulate)
                % Retrieve the proximal segment's Rotation Matrix (Global-to-Local)
                if strcmp(pn, 'Pelvis')
                    R_prox = LCS.Pelvis(:,:,s:e); % Pelvis isn't side-specific
                else
                    R_prox = LCS.([pn '_' k])(:,:,s:e);
                end
                
                N_frames = e - s + 1;
                F_loc = zeros(N_frames, 3);
                M_loc = zeros(N_frames, 3);
                
                % Rotate the Global vectors into the Local frame frame-by-frame
                for i = 1:N_frames
                    F_loc(i,:) = (R_prox(:,:,i) * RawID.(jn).Force(i,:)')';
                    M_loc(i,:) = (R_prox(:,:,i) * RawID.(jn).Moment(i,:)')';
                end
                
                % Time-normalize the newly rotated local arrays
                F_loc_norm = nrm(F_loc);
                M_loc_norm = nrm(M_loc);
                
                Acc.(side).Local.Force.(jn)  = cat(3, Acc.(side).Local.Force.(jn),  F_loc_norm);
                Acc.(side).Local.Moment.(jn) = cat(3, Acc.(side).Local.Moment.(jn), M_loc_norm);
            end
        end
        
        % Calculate Mean and SD using the 'msd' helper function
        Ensemble = struct();
        for side = {'Right', 'Left'}
            s_name = side{1};
            for j = 1:3
                jn = joints{j};
                Ensemble.(s_name).Global.Force.(jn)  = msd(Acc.(s_name).Global.Force.(jn), 3);
                Ensemble.(s_name).Global.Moment.(jn) = msd(Acc.(s_name).Global.Moment.(jn), 3);
                Ensemble.(s_name).Local.Force.(jn)   = msd(Acc.(s_name).Local.Force.(jn), 3);
                Ensemble.(s_name).Local.Moment.(jn)  = msd(Acc.(s_name).Local.Moment.(jn), 3);
                Ensemble.(s_name).Power.(jn)         = msd(Acc.(s_name).Power.(jn), 2);
            end
        end
        Data.(p_name).EnsembleID = Ensemble;
    end
    fprintf('Q9: Calculated ensemble averages (Global and Local) for all participants.\n');
end

function plot_inverse_dynamics_ensemble(Data,plotType)
% Q10: Generates 3 figures per participant for Inverse Dynamics Ensemble Averages:
% 1. 3x3 Local Joint Reaction Forces (N)
% 2. 3x3 Local Net Joint Moments (N.m)
% 3. 1x3 Joint Powers (W)
% Mean +/- 1 SD bounds. Left = Red, Right = Blue.


    joints = {'Ankle', 'Knee', 'Hip'};
    comps = {'X', 'Y', 'Z'};

    for p = fieldnames(Data)'
        p_name = p{1};
        if ~isfield(Data.(p_name), 'EnsembleID'); continue; end
        
        E = Data.(p_name).EnsembleID;

        % FIGURE 1: Global Joint Forces (3x3)
        fig_f = figure('Name', sprintf('%s - %s Joint Forces (Q10)', p_name, plotType), ...
            'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
        for c = 1:3 
            for j = 1:3 
                subplot(3, 3, (c-1)*3 + j); hold on; grid on;
                if c == 1; title(joints{j}, 'FontWeight', 'bold'); end
                if j == 1; ylabel(sprintf('Force %s (N)', comps{c})); end
                if c == 3; xlabel('% Gait Cycle'); end
                
                % Plot Bands
                plot_band_col(E.Left.(plotType).Force.(joints{j}), c, 'r');
                plot_band_col(E.Right.(plotType).Force.(joints{j}), c, 'b');
                
                xlim([0 100]);
                if c == 1 && j == 1
                    legend([plot(nan,nan,'r','LineWidth',2), plot(nan,nan,'b','LineWidth',2)], ...
                        {'Left','Right'}, 'Location', 'best');
                end
            end
        end

        % FIGURE 2: Global Joint Moments (3x3)
        fig_m = figure('Name', sprintf('%s - %s Joint Moments (Q10)', p_name, plotType), ...
            'Color', 'w', 'Units', 'normalized', 'Position', [0.15 0.15 0.8 0.8]);
        for c = 1:3 
            for j = 1:3 
                subplot(3, 3, (c-1)*3 + j); hold on; grid on;
                if c == 1; title(joints{j}, 'FontWeight', 'bold'); end
                if j == 1; ylabel(sprintf('Moment %s (N.m)', comps{c})); end
                if c == 3; xlabel('% Gait Cycle'); end
                
                % Plot Bands
                plot_band_col(E.Left.(plotType).Moment.(joints{j}), c, 'r');
                plot_band_col(E.Right.(plotType).Moment.(joints{j}), c, 'b');
                
                xlim([0 100]);
                if c == 1 && j == 1
                    legend([plot(nan,nan,'r','LineWidth',2), plot(nan,nan,'b','LineWidth',2)], ...
                        {'Left','Right'}, 'Location', 'best');
                end
            end
        end

        % FIGURE 3: Joint Power (1x3)
        fig_p = figure('Name', sprintf('%s - Joint Power (Q10)', p_name), ...
            'Color', 'w', 'Units', 'normalized', 'Position', [0.2 0.4 0.7 0.35]);
        for j = 1:3 % Cols: Ankle, Knee, Hip
            subplot(1, 3, j); hold on; grid on;
            
            % Titles and Labels
            title(joints{j}, 'FontWeight', 'bold');
            if j == 1; ylabel('Power (W)'); end
            xlabel('% Gait Cycle');
            
            % Plot Bands
            plot_band(E.Left.Power.(joints{j}), 'r');
            plot_band(E.Right.Power.(joints{j}), 'b');
            
            xlim([0 100]);
            
            if j == 1
                legend([plot(nan,nan,'r','LineWidth',2), plot(nan,nan,'b','LineWidth',2)], ...
                    {'Left','Right'}, 'Location', 'best');
            end
        end
    end
    fprintf('Q10: Generated all Inverse Dynamics figures.\n');
end


%% Core Math Helpers

function nd = nrm(raw)
% Linear interpolation to 101 samples
nd = interp1(1:size(raw,1), raw, linspace(1,size(raw,1),101)');
end

function S = msd(M, dim)
% Mean and SD along dim
if isempty(M); S.Mean=[]; S.SD=[]; return; end
S.Mean=mean(M,dim); S.SD=std(M,0,dim);
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
