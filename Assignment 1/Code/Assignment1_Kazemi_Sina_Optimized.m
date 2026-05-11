% MEC E 686 - Assignment 1
% Sina Kazemi
%% Main
clear all; close all; clc

Data = struct();
for i = 1:3
    for j = 1:6
        [Data.("P"+i).("T"+j).MK, ...
            Data.("P"+i).("T"+j).Freq, ...
            Data.("P"+i).("T"+j).NumFr] = load_mocap_data("Data\MC_P_"+i+"_T_"+j+".txt", i, j);
    end
end

Data = repair_missing_markers(Data);        % Interpolate short gaps
Data = unify_walking_direction(Data);       % Align all trials to -Y walking direction
Data = calculate_all_LCS(Data);             % Q1
Data = calculate_helical_angles(Data);      % Q2
Data = calculate_3D_angles(Data);           % Q3
Data = detect_gait_events(Data);            % Q4
Data = extract_gait_cycle_data(Data);       % Q5
Data = time_normalize_data(Data);           % Q6
Data = calculate_ensemble_averages(Data);   % Q7
plot_ensemble_averages(Data);               % Q8
Data = filter_and_plot(Data,'P1');          % Q9

% animate_trial_LCS(Data, 'P1', 'T1', 5);
% plot_figure_1(Data, 'P1', 'T1');
clearvars -except Data
% save('Data.mat', 'Data')

%% Functions

function [markers, frequency, num_frames] = load_mocap_data(filename, Pnum, Ptr)
% Loads motion capture data from .txt file
% Returns markers struct (N x 3 per marker), sampling frequency, and frame count
fileID = fopen(filename, 'r');
fgetl(fileID); frequency = str2double(fgetl(fileID));
fclose(fileID);

fileID = fopen(filename, 'r');
fgetl(fileID); fgetl(fileID);
marker_list = strtrim(strsplit(strtrim(fgetl(fileID)), '\t'));
marker_list  = marker_list(~cellfun('isempty', marker_list));
fclose(fileID);

opts = detectImportOptions(filename,'FileType','text','Delimiter','\t','NumHeaderLines',5);
data = readtable(filename, opts);
num_frames = height(data);

markers = struct();
for i = 1:length(marker_list)
    c = 3 + (i-1)*3;
    markers.(marker_list{i}) = data{:, c:c+2};
end
fprintf('Loaded P%d T%d: %d frames @ %d Hz\n', Pnum, Ptr, num_frames, frequency);
end

function Data = repair_missing_markers(Data)
% Scans all trials for NaN marker data
% Short gaps (<= max_gap frames) are filled by linear interpolation
% Trials with longer gaps or fully missing markers are flagged Data.Pn.Tn.Bad=true
max_gap = 200; % frames
for p = fieldnames(Data)'
    p_name = p{1};
    for t = fieldnames(Data.(p_name))'
        t_name = t{1};
        MK = Data.(p_name).(t_name).MK;
        n  = Data.(p_name).(t_name).NumFr;
        is_bad = false;
        for m = fieldnames(MK)'
            mk  = m{1};
            col = MK.(mk);
            nan_rows = any(isnan(col), 2);
            if ~any(nan_rows); continue; end
            % Find contiguous NaN segments
            d = diff([0; nan_rows; 0]);
            gap_len = find(d==-1) - find(d==1);   % length of each gap
            if any(gap_len > max_gap) || sum(nan_rows) == n
                fprintf('WARNING: %s %s — marker %s has %d NaN frames (max=%d). Trial excluded.\n', ...
                    p_name, t_name, mk, sum(nan_rows), max_gap);
                is_bad = true; break;
            end
            % Interpolate each axis over the short gap
            x = (1:n)';
            for ax = 1:3
                good = ~isnan(col(:,ax));
                col(:,ax) = interp1(x(good), col(good,ax), x, 'linear', 'extrap');
            end
            Data.(p_name).(t_name).MK.(mk) = col;
            fprintf('Repaired %s %s — marker %s (%d frames interpolated)\n', ...
                p_name, t_name, mk, sum(nan_rows));
        end
        if is_bad; Data.(p_name).(t_name).Bad = true; end
    end
end
end

function Data = unify_walking_direction(Data)
% Ensures all subjects walk in the -Y direction
% Flips X and Y (180 deg about Z) if RASIS moves toward +Y
for p = fieldnames(Data)'
    p_name = p{1};
    for t = fieldnames(Data.(p_name))'
        t_name = t{1};
        MK = Data.(p_name).(t_name).MK;
        if MK.RASIS(end,2) > MK.RASIS(1,2)
            for m = fieldnames(MK)'
                Data.(p_name).(t_name).MK.(m{1})(:,1:2) = -MK.(m{1})(:,1:2);
            end
            fprintf('Flipped direction: %s %s\n', p_name, t_name);
        end
    end
end
end

function Data = calculate_all_LCS(Data)
% Q1: Computes LCS rotation matrices for pelvis, thighs, shanks, and feet
% Stored as 3x3xN arrays. Axes per Assignment Guide
% Left-side segments negate the transverse reference vector (ME2LE, MM2LM, TP)
for p = fieldnames(Data)'
    p_name = p{1};
    for t = fieldnames(Data.(p_name))'
        t_name = t{1};
        MK = Data.(p_name).(t_name).MK;

        % Pelvis: z = RASIS-LASIS, y from ASIS-PSIS plane, x = y x z
        pm = (MK.RPSIS + MK.LPSIS) / 2;
        vz = nv(MK.RASIS - MK.LASIS);
        vy = nv(cross(nv(MK.RASIS - pm), nv(MK.LASIS - pm), 2));
        Data.(p_name).(t_name).LCS.Pelvis  = packR(nv(cross(vy,vz,2)), vy, vz);

        % Thighs: y = GT-LE, x = y x ME2LE, z = x x y
        Data.(p_name).(t_name).LCS.Thigh_R = lcs_thigh(MK.RLE,MK.RME,MK.RGT, 1);
        Data.(p_name).(t_name).LCS.Thigh_L = lcs_thigh(MK.LLE,MK.LME,MK.LGT,-1);

        % Shanks: y = HF-LM, x = y x MM2LM, z = x x y
        Data.(p_name).(t_name).LCS.Shank_R = lcs_shank(MK.RLM,MK.RMM,MK.RHF, 1);
        Data.(p_name).(t_name).LCS.Shank_L = lcs_shank(MK.LLM,MK.LMM,MK.LHF,-1);

        % Feet: TP = b x a (plantar normal), SP = c x TP, y=TP, x=TP x SP
        Data.(p_name).(t_name).LCS.Foot_R  = lcs_foot(MK.RCA,MK.RMH1,MK.RMH5,MK.RMH2, 1);
        Data.(p_name).(t_name).LCS.Foot_L  = lcs_foot(MK.LCA,MK.LMH1,MK.LMH5,MK.LMH2,-1);

        fprintf('LCS done: %s %s\n', p_name, t_name);
    end
end
end

function R = lcs_thigh(LE, ME, GT, sgn)
me2le = sgn * nv(LE - ME);
vy = nv(GT - LE); vx = nv(cross(vy,me2le,2));
R  = packR(vx, vy, nv(cross(vx,vy,2)));
end

function R = lcs_shank(LM, MM, HF, sgn)
mm2lm = sgn * nv(LM - MM);
vy = nv(HF - LM); vx = nv(cross(vy,mm2lm,2));
R  = packR(vx, vy, nv(cross(vx,vy,2)));
end

function R = lcs_foot(CA, MH1, MH5, MH2, sgn)
TP = sgn * nv(cross(nv(MH5-CA), nv(MH1-CA), 2));
SP = nv(cross(nv(MH2-CA), TP, 2));
vx = nv(cross(TP,SP,2));
R  = packR(vx, TP, nv(cross(vx,TP,2)));
end

function Data = calculate_helical_angles(Data)
% Q2: Helical angle (deg) per joint. R = R_prox * R_dist' per frame
% Angle computed from skew-symmetric part of R (Assignment Guide)
for p = fieldnames(Data)'
    p_name = p{1};
    for t = fieldnames(Data.(p_name))'
        t_name = t{1};
        if ~isfield(Data.(p_name).(t_name),'LCS'); continue; end
        L  = Data.(p_name).(t_name).LCS;
        HA.Hip_R   = helical_batch(L.Pelvis,  L.Thigh_R);
        HA.Knee_R  = helical_batch(L.Thigh_R, L.Shank_R);
        HA.Ankle_R = helical_batch(L.Shank_R, L.Foot_R);
        HA.Hip_L   = helical_batch(L.Pelvis,  L.Thigh_L);
        HA.Knee_L  = helical_batch(L.Thigh_L, L.Shank_L);
        HA.Ankle_L = helical_batch(L.Shank_L, L.Foot_L);
        Data.(p_name).(t_name).Helical_Angles = HA;
        fprintf('Helical done: %s %s\n', p_name, t_name);
    end
end
end

function Data = calculate_3D_angles(Data)
% Q3: JCS angles [F/E, A/A, ER/IR] per Grood & Suntay (1983)
% Sign corrections applied to match guide conventions (Figure 3)
for p = fieldnames(Data)'
    p_name = p{1};
    for t = fieldnames(Data.(p_name))'
        t_name = t{1};
        if ~isfield(Data.(p_name).(t_name),'LCS'); continue; end
        L  = Data.(p_name).(t_name).LCS;
        JA.Hip_R   = jcs_batch(L.Pelvis,  L.Thigh_R, 'R');
        JA.Knee_R  = jcs_batch(L.Thigh_R, L.Shank_R, 'R');
        JA.Ankle_R = jcs_batch(L.Shank_R, L.Foot_R,  'R');
        JA.Hip_L   = jcs_batch(L.Pelvis,  L.Thigh_L, 'L');
        JA.Knee_L  = jcs_batch(L.Thigh_L, L.Shank_L, 'L');
        JA.Ankle_L = jcs_batch(L.Shank_L, L.Foot_L,  'L');
        JA.Knee_R(:,1) = -JA.Knee_R(:,1); % F/E sign correction
        JA.Knee_L(:,1) = -JA.Knee_L(:,1); % F/E sign correction
        Data.(p_name).(t_name).Joint_Angles = JA;
        fprintf('3D angles done: %s %s\n', p_name, t_name);
    end
end
end

function Data = detect_gait_events(Data)
% Q4: Identifies FP landing side, FP heel-strike, and next heel-strike
% Heel strikes = local minima of Z. FP landing confirmed by Y within plate bounds (+/-30mm)
for p = fieldnames(Data)'
    p_name = p{1};
    for t = fieldnames(Data.(p_name))'
        t_name = t{1};
        MK   = Data.(p_name).(t_name).MK;
        freq = Data.(p_name).(t_name).Freq;
        Data.(p_name).(t_name).GaitEvents = struct('Side','None','HS_FP',NaN,'HS_Next',NaN);
        if ~isfield(MK,'FP1'); continue; end

        fp_y = [MK.FP1(1,2),MK.FP2(1,2),MK.FP3(1,2),MK.FP4(1,2)];
        ylo  = min(fp_y)-30; yhi = max(fp_y)+30;
        mpd  = round(0.7*freq);
        [~,lR] = findpeaks(-MK.RCA(:,3),'MinPeakDistance',mpd);
        [~,lL] = findpeaks(-MK.LCA(:,3),'MinPeakDistance',mpd);

        GE = fp_strike(MK.RCA, lR, ylo, yhi, 'Right');
        if strcmp(GE.Side,'None')
            GE = fp_strike(MK.LCA, lL, ylo, yhi, 'Left');
        end
        Data.(p_name).(t_name).GaitEvents = GE;
        fprintf('Gait %s %s: %s | FP:%d Next:%d\n', p_name, t_name, GE.Side, GE.HS_FP, GE.HS_Next);
    end
end
end

function Data = extract_gait_cycle_data(Data)
% Q5: Trims angles to FP gait cycle (HS_FP to HS_Next) for the landing side only
for p = fieldnames(Data)'
    p_name = p{1};
    for t = fieldnames(Data.(p_name))'
        t_name = t{1};
        if ~isfield(Data.(p_name).(t_name),'GaitEvents'); continue; end
        GE = Data.(p_name).(t_name).GaitEvents;
        if strcmp(GE.Side,'None') || isnan(GE.HS_FP); continue; end

        s=GE.HS_FP; e=GE.HS_Next; k=GE.Side(1);
        HA = Data.(p_name).(t_name).Helical_Angles;
        JA = Data.(p_name).(t_name).Joint_Angles;

        Ex.Side              = GE.Side;
        Ex.Helical.Hip       = HA.("Hip_"  +k)(s:e);
        Ex.Helical.Knee      = HA.("Knee_" +k)(s:e);
        Ex.Helical.Ankle     = HA.("Ankle_"+k)(s:e);
        Ex.JointAngles.Hip   = JA.("Hip_"  +k)(s:e,:);
        Ex.JointAngles.Knee  = JA.("Knee_" +k)(s:e,:);
        Ex.JointAngles.Ankle = JA.("Ankle_"+k)(s:e,:);

        Data.(p_name).(t_name).Extracted = Ex;
        fprintf('Extracted %s %s: %s, frames %d-%d\n', p_name, t_name, GE.Side, s, e);
    end
end
end

function Data = time_normalize_data(Data)
% Q6: Linear interpolation to 101 samples (0-100% gait cycle)
% interp1(unts, linspace(1,ol,nl)) per Assignment Guide
for p = fieldnames(Data)'
    p_name = p{1};
    for t = fieldnames(Data.(p_name))'
        t_name = t{1};
        if ~isfield(Data.(p_name).(t_name),'Extracted'); continue; end
        Ex = Data.(p_name).(t_name).Extracted;
        N.Side              = Ex.Side;
        N.Helical.Hip       = nrm(Ex.Helical.Hip);
        N.Helical.Knee      = nrm(Ex.Helical.Knee);
        N.Helical.Ankle     = nrm(Ex.Helical.Ankle);
        N.JointAngles.Hip   = nrm(Ex.JointAngles.Hip);
        N.JointAngles.Knee  = nrm(Ex.JointAngles.Knee);
        N.JointAngles.Ankle = nrm(Ex.JointAngles.Ankle);
        Data.(p_name).(t_name).Normalized = N;
    end
end
end

function Data = calculate_ensemble_averages(Data)
% Q7: Ensemble mean +/- SD across trials per participant and side
% Helical (101 x Trials): avg dim 2. JCS (101 x 3 x Trials): avg dim 3
joints = {'Hip','Knee','Ankle'};
for p = fieldnames(Data)'
    p_name = p{1};
    t_fields = fieldnames(Data.(p_name));
    RH=cell(1,3); LH=cell(1,3); RJ=cell(1,3); LJ=cell(1,3);

    for t = 1:numel(t_fields)
        t_name = t_fields{t};
        if ~isfield(Data.(p_name).(t_name),'Normalized'); continue; end
        N   = Data.(p_name).(t_name).Normalized;
        isR = strcmpi(N.Side,'Right');
        for j = 1:3
            jn = joints{j};
            if isR; RH{j}=[RH{j},N.Helical.(jn)]; RJ{j}=cat(3,RJ{j},N.JointAngles.(jn));
            else;   LH{j}=[LH{j},N.Helical.(jn)]; LJ{j}=cat(3,LJ{j},N.JointAngles.(jn)); end
        end
    end

    for j = 1:3
        jn = joints{j};
        Data.(p_name).Ensemble.Right.Helical.(jn)     = msd(RH{j},2);
        Data.(p_name).Ensemble.Left.Helical.(jn)      = msd(LH{j},2);
        Data.(p_name).Ensemble.Right.JointAngles.(jn) = msd(RJ{j},3);
        Data.(p_name).Ensemble.Left.JointAngles.(jn)  = msd(LJ{j},3);
    end
    fprintf('Ensemble done: %s\n', p_name);
end
end

function plot_ensemble_averages(Data)
% Q8: Ensemble mean +/-1 SD for helical (1x3) and 3D (3x3) angles per participant
% Left = red, Right = blue
joints = {'Hip','Knee','Ankle'};
rlbls  = {'F/E (deg)','A/A (deg)','ER/IR (deg)'};

for p = fieldnames(Data)'
    p_name = p{1};
    if ~isfield(Data.(p_name),'Ensemble'); continue; end
    E = Data.(p_name).Ensemble;

    figure('Name',sprintf('%s - Helical Angles (Q8)',p_name),'Color','w', ...
        'NumberTitle','off','Units','normalized','Position',[0.1 0.55 0.8 0.35]);
    for j = 1:3
        subplot(1,3,j); hold on; grid on;
        title(joints{j}); xlabel('% Gait Cycle'); ylabel('Helical Angle (deg)');
        plot_band(E.Left.Helical.(joints{j}),  'r');
        plot_band(E.Right.Helical.(joints{j}), 'b');
        xlim([0 100]);
        if j==1; legend([plot(nan,nan,'r','LineWidth',2),plot(nan,nan,'b','LineWidth',2)], ...
                {'Left','Right'},'Location','best'); end
    end

    figure('Name',sprintf('%s - 3D Joint Angles (Q8)',p_name),'Color','w', ...
        'NumberTitle','off','Units','normalized','Position',[0.1 0.05 0.8 0.45]);
    for j = 1:3
        for k = 1:3
            subplot(3,3,(k-1)*3+j); hold on; grid on;
            if k==1; title(joints{j},'FontWeight','bold'); end
            if j==1; ylabel(rlbls{k}); end
            if k==3; xlabel('% Gait Cycle'); end
            plot_band_col(E.Left.JointAngles.(joints{j}),  k, 'r');
            plot_band_col(E.Right.JointAngles.(joints{j}), k, 'b');
            xlim([0 100]);
        end
    end
end
end

function Data = filter_and_plot(Data,p_name)
% Q9: 4th-order zero-phase Butterworth LP filter (5 Hz) on P helical angles
% Recomputes ensemble averages and re-plots for comparison with unfiltered
if ~isfield(Data,p_name); warning('Data not found.'); return; end

joints = {'Hip','Knee','Ankle'};
fo=4; fc=5;
RH=cell(1,3); LH=cell(1,3);

for t = fieldnames(Data.(p_name))'
    t_name = t{1};
    if ~isfield(Data.(p_name).(t_name),'Normalized'); continue; end
    N  = Data.(p_name).(t_name).Normalized;
    [b,a] = butter(fo, fc/(0.5*Data.(p_name).(t_name).Freq));
    isR = strcmpi(N.Side,'Right');
    for j = 1:3
        f = filtfilt(b,a,N.Helical.(joints{j}));
        if isR; RH{j}=[RH{j},f]; else; LH{j}=[LH{j},f]; end
    end
end

FE = struct();
for j = 1:3
    FE.Right.(joints{j}) = msd(RH{j},2);
    FE.Left.(joints{j})  = msd(LH{j},2);
end
Data.(p_name).FilteredEnsemble = FE;

figure('Name',sprintf('%s - Filtered Helical Angles (Q9)',p_name),'Color','w', ...
    'NumberTitle','off','Units','normalized','Position',[0.1 0.1 0.8 0.35]);
for j = 1:3
    subplot(1,3,j); hold on; grid on;
    title(sprintf('Filtered %s',joints{j})); xlabel('% Gait Cycle'); ylabel('Helical Angle (deg)');
    plot_band(FE.Left.(joints{j}),  'r');
    plot_band(FE.Right.(joints{j}), 'b');
    xlim([0 100]);
    if j==1; legend([plot(nan,nan,'r','LineWidth',2),plot(nan,nan,'b','LineWidth',2)], ...
            {'Left','Right'},'Location','best'); end
end
fprintf('Filtered ensemble plotted for %s\n', p_name);
end

%% Core Math Helpers

function v = nv(v)
% Row-wise unit normalization for N x 3 matrices
v = v ./ vecnorm(v,2,2);
end

function R = packR(vx, vy, vz)
% Stacks N row-vector triplets into a 3x3xN rotation matrix array
% R(:,:,i) = [vx(i,:); vy(i,:); vz(i,:)]
N=size(vx,1); R=zeros(3,3,N);
R(1,:,:)=vx'; R(2,:,:)=vy'; R(3,:,:)=vz';
end

function theta = helical_batch(Rp, Rd)
% Vectorized helical angle (deg) for all N frames
% R = Rp*Rd' per frame; angle from skew-symmetric part of R
% acos used when sin(theta) > sqrt(2)/2, else asin
N=size(Rp,3); theta=zeros(N,1);
for i = 1:N
    R = Rp(:,:,i)*Rd(:,:,i)';
    s = 0.5*sqrt((R(2,3)-R(3,2))^2+(R(3,1)-R(1,3))^2+(R(1,2)-R(2,1))^2);
    if s > sqrt(2)/2; theta(i) = rad2deg(acos(max(-1,min(1,(trace(R)-1)/2))));
    else;             theta(i) = rad2deg(asin(max(-1,min(1,s)))); end
end
end

function angles = jcs_batch(Rp, Rd, side)
% Vectorized JCS angles [F/E, A/A, ER/IR] (deg) for all N frames
% e1=Z_prox (lateral), e3=y_dist (vertical), e2=e3 x e1 (floating axis)
N=size(Rp,3); angles=zeros(N,3); isR=(side=='R');
for i = 1:N
    Xp=Rp(1,:,i)'; Yp=Rp(2,:,i)'; Zp=Rp(3,:,i)';
    xd=Rd(1,:,i)'; yd=Rd(2,:,i)'; zd=Rd(3,:,i)';
    e2=cross(yd,Zp); e2=e2/norm(e2);

    sFE=sign(dot(e2,Yp)); if sFE==0; sFE=1; end
    FE = acos(max(-1,min(1,dot(Xp,e2)))) * sFE;
    if isnan(FE); FE=0; end

    dAA=acos(max(-1,min(1,dot(Zp,yd))));
    if isR; AA=-pi/2+dAA; else; AA=pi/2-dAA; end
    if isnan(AA); AA=0; end

    sER=sign(dot(e2,zd)); if sER==0; sER=1; end
    if isR; ERIR=-acos(max(-1,min(1,dot(xd,e2))))*sER;
    else;   ERIR= acos(max(-1,min(1,dot(xd,e2))))*sER; end
    if isnan(ERIR); ERIR=0; end

    angles(i,:) = rad2deg([FE,AA,ERIR]);
end
end

function nd = nrm(raw)
% Linear interpolation to 101 samples: interp1(unts, linspace(1,ol,101))
nd = interp1(1:size(raw,1), raw, linspace(1,size(raw,1),101)');
end

function S = msd(M, dim)
% Mean and SD along dim. Returns empty struct if input is empty
if isempty(M); S.Mean=[]; S.SD=[]; return; end
S.Mean=mean(M,dim); S.SD=std(M,0,dim);
end

function GE = fp_strike(heel, locs, ylo, yhi, side)
% Finds the first heel-strike loc where Y falls within FP bounds
GE = struct('Side','None','HS_FP',NaN,'HS_Next',NaN);
for i = 1:length(locs)-1
    if heel(locs(i),2)>=ylo && heel(locs(i),2)<=yhi
        GE = struct('Side',side,'HS_FP',locs(i),'HS_Next',locs(i+1));
        return;
    end
end
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

%% Visualization

function animate_trial_LCS(Data, p_name, t_name, speed_factor)
% Animates stick figure with LCS axes for a single trial
if nargin<4; speed_factor=1; end
MK=Data.(p_name).(t_name).MK; n=size(MK.RASIS,1);
has_lcs=isfield(Data.(p_name).(t_name),'LCS');
if has_lcs; L=Data.(p_name).(t_name).LCS; end

fig_name=sprintf('Animation: %s-%s',p_name,t_name);
old=findobj('type','figure','Name',fig_name);
if ~isempty(old); close(old); end
figure('Name',fig_name,'Color','w','NumberTitle','off');
ax=axes; grid on; axis equal; hold on;
xlabel('X(mm)'); ylabel('Y(mm)'); zlabel('Z(mm)'); view(3);
xlim([min([MK.RASIS(:,1);MK.RCA(:,1)])-500, max([MK.RASIS(:,1);MK.RCA(:,1)])+500]);
ylim([min([MK.RASIS(:,2);MK.RCA(:,2)])-500, max([MK.RASIS(:,2);MK.RCA(:,2)])+500]);
zlim([0 1800]);
fp_pts=[];
if isfield(MK,'FP1')
    fp_pts=[MK.FP1(1,:);MK.FP2(1,:);MK.FP3(1,:);MK.FP4(1,:);MK.FP1(1,:)];
end

for i = 1:speed_factor:n
    cla(ax);
    if ~isempty(fp_pts); plot3(fp_pts(:,1),fp_pts(:,2),fp_pts(:,3),'k-','LineWidth',2); end
    Rk=(MK.RLE(i,:)+MK.RME(i,:))/2; Lk=(MK.LLE(i,:)+MK.LME(i,:))/2;
    Ra=(MK.RLM(i,:)+MK.RMM(i,:))/2; La=(MK.LLM(i,:)+MK.LMM(i,:))/2;
    Pm=(MK.RASIS(i,:)+MK.LASIS(i,:))/2;
    seg3(MK.RASIS(i,:),MK.LASIS(i,:),'k'); seg3(MK.RPSIS(i,:),MK.LPSIS(i,:),'k');
    seg3(MK.RASIS(i,:),MK.RPSIS(i,:),'k'); seg3(MK.LASIS(i,:),MK.LPSIS(i,:),'k');
    seg3(MK.RGT(i,:),Rk,'b'); seg3(Rk,Ra,'b'); seg3(MK.RCA(i,:),MK.RMH2(i,:),'b');
    seg3(MK.LGT(i,:),Lk,'r'); seg3(Lk,La,'r'); seg3(MK.LCA(i,:),MK.LMH2(i,:),'r');
    if has_lcs; sc=200;
        lcs3(Pm,L.Pelvis(:,:,i),sc);  lcs3(Rk,L.Thigh_R(:,:,i),sc);
        lcs3(Ra,L.Shank_R(:,:,i),sc); lcs3(MK.RCA(i,:),L.Foot_R(:,:,i),sc);
        lcs3(Lk,L.Thigh_L(:,:,i),sc); lcs3(La,L.Shank_L(:,:,i),sc);
        lcs3(MK.LCA(i,:),L.Foot_L(:,:,i),sc);
    end
    title(sprintf('Frame %d/%d',i,n)); drawnow;
end
end

function plot_figure_1(Data, p_name, t_name)
% Recreates Figure 1: heel Z (top) and Y (bottom) trajectories with event markers
MK=Data.(p_name).(t_name).MK; n=size(MK.RCA,1); x=1:n;
figure('Name',sprintf('Figure 1: %s-%s',p_name,t_name),'Color','w', ...
    'NumberTitle','off','Units','normalized','Position',[0.2 0.2 0.6 0.6]);

subplot(2,1,1); hold on; box on;
plot(x,MK.LCA(:,3),'r','LineWidth',1.2,'DisplayName','Left heel');
plot(x,MK.RCA(:,3),'b','LineWidth',1.2,'DisplayName','Right heel');
if isfield(Data.(p_name).(t_name),'GaitEvents')
    GE=Data.(p_name).(t_name).GaitEvents;
    if ~strcmp(GE.Side,'None') && ~isnan(GE.HS_FP)
        heel=MK.(GE.Side(1)+"CA");
        plot(GE.HS_FP, heel(GE.HS_FP,3),'d','MarkerFaceColor','g','MarkerEdgeColor','g','MarkerSize',8,'HandleVisibility','off');
        plot(GE.HS_Next,heel(GE.HS_Next,3),'d','MarkerFaceColor','k','MarkerEdgeColor','k','MarkerSize',8,'HandleVisibility','off');
    end
end
ylabel('Z-coordinate (mm)'); xlim([1 n]); legend('Location','northeast'); set(gca,'XTickLabel',[]);

subplot(2,1,2); hold on; box on;
plot(x,MK.LCA(:,2),'r','LineWidth',1.2,'DisplayName','Left heel');
plot(x,MK.RCA(:,2),'b','LineWidth',1.2,'DisplayName','Right heel');
if isfield(MK,'FP1')
    clrs={[.4 .4 .4],[.5 .5 .5],[.6 .6 .6],[.7 .7 .7]};
    for f=1:4; plot(x,MK.("FP"+f)(:,2),'Color',clrs{f},'DisplayName',"FP"+f); end
end
ylabel('Y-coordinate (mm)'); xlabel('Sample'); xlim([1 n]); legend('Location','northeast');
end

function seg3(p1, p2, col)
plot3([p1(1) p2(1)],[p1(2) p2(2)],[p1(3) p2(3)],'Color',col,'LineWidth',1.5);
end

function lcs3(org, R, sc)
% Plots LCS axes as quiver3 arrows. R is 3x3; rows = X(red), Y(green), Z(blue)
cols={'r','g','b'};
for r=1:3
    quiver3(org(1),org(2),org(3),R(r,1)*sc,R(r,2)*sc,R(r,3)*sc, ...
        cols{r},'LineWidth',2,'MaxHeadSize',0.5,'AutoScale','off');
end
end
