% Final Project EEE 606 
% Kaushik Iyer (1223696175)

%% State Space / KF Approach to track Noisy ECG Signals
% State Space/ KF approach for systems where the dynamical model is not
% known and/or the system is an FIR filter. Helps incorporate noise models
% for the observed noisy input signal and also incorporates the gradient
% noise effects that conventional Adaptive filtering algorithms (LMS, NLMS, 
% RLS, etc) fail to account for.

% Initialisations and Definitions for the system
L       = 15;           % Order of the FIR filter
x       = zeros(L+1,1); % State vector (contains filter weights)
P0      = 10;            % State error variance 
P       = P0*eye(L+1);  % State error covariance matrix
Q0      = 2e0;          % Gradient noise
Q       = Q0*eye(L+1);  % Process noise co-variance matrix
F       = eye(L+1);     % State transition matrix 
xBuffer = zeros(L+1,1); % Convolution buffer
muSS    = 0.001;

% Generate ECG signal (clean and noisy)
N = 2;
y = genECG(N)';
Nvar = 0.4;
nu = Nvar*randn(length(y),1);
yobs = y + nu; 

R = (1/Nvar)*eye(1);   % Observation noise co-variance matrix 
ySS = zeros(length(y),1);
e_meas = zeros(length(y),1);
for i = 1:length(y)
    xBuffer = [yobs(i); xBuffer(1:end-1)];
    H       = xBuffer';
    e_meas(i)  = yobs(i) - H*x;
    % Predict
    x = F*x + muSS*e_meas(i)*xBuffer;
    P = F*P*F + Q;
    % Update
    K = P*H'/(H*P*H' + R);
    x = x + K*(e_meas(i));
    P = (eye(1) - K*H)*P;
    
    ySS(i) = x'*xBuffer;
end

% Initialisations for LMS, NLMS and RLS
muLMS = 0.001;                % Step size LMS
eLMS = zeros(length(y),1);    % Output error LMS 
wLMS = zeros(L+1,1);          % LMS Filter weights
yLMS = zeros(length(y),1);    % LMS Output 
beta = 0.99;                  % Forgetting factor for RLS
betaInv = 1/beta;             % Inverse of beta
delta   = 1e-3;               % Parameter to Initialise Rxx
RxxInv  = (1/delta)*eye(L+1); % Rxx^(-1) Initialisation
eRLS    = zeros(length(y),1); % Output error RLS
yRLS    = zeros(length(y),1); % RLS Output 
wRLS    = zeros(L+1,1);       % RLS Filter weights
muNLMS  = 0.2;                % Step size NLMS
eNLMS   = zeros(length(y),1); % Output error NLMS
yNLMS   = zeros(length(y),1); % NLMS Output 
wNLMS   = zeros(L+1,1);       % NLMS Filter weights
xBuffer = zeros(L+1,1);       % Convolution buffer

% Loop for LMS
for i = 1:length(y)
    xBuffer = [yobs(i); xBuffer(1:end-1)];  % Update buffer
    yLMS(i) = wLMS'*xBuffer;                % Compute output 
    eLMS(i) = y(i) - yLMS(i);               % Compute error
    wLMS    = wLMS + muLMS*eLMS(i)*xBuffer; % Update filter weights
end

xBuffer = zeros(L+1,1);       % Convolution buffer
% Loop for RLS
for i = 1:length(y)
    xBuffer = [yobs(i); xBuffer(1:end-1)];                                    % Update the buffer
    KRLS    = (betaInv*RxxInv*xBuffer)/(1 + betaInv*xBuffer'*RxxInv*xBuffer); % Gain computation
    yRLS(i) = wRLS'*xBuffer;                                                  % Compute output 
    eRLS(i) = y(i) - wRLS'*xBuffer;                                           % Error computation
    wRLS    = wRLS + KRLS*eRLS(i);                                            % Weight update
    RxxInv  = betaInv*RxxInv - betaInv*KRLS*xBuffer'*RxxInv;                  % RxxInv update   
end

xBuffer = zeros(L+1,1);       % Convolution buffer
% Loop for NLMS 
for i = 1:length(y)
    xBuffer = [yobs(i); xBuffer(1:end-1)];                          % Update buffer
    yNLMS(i) = wNLMS'*xBuffer;                                      % Compute output 
    eNLMS(i) = y(i) - yNLMS(i);                                     % Compute error
    wNLMS    = wNLMS + (muNLMS*eNLMS(i)/(norm(xBuffer)^2))*xBuffer; % Update filter weights
end




%% Plots 

% Clean and Observed Sinusoids
figure(1) 
subplot(2,1,1)
plot(y,'LineStyle','-',LineWidth=2)
xlabel('Time Index'); ylabel('Amplitude'); 
title('Clean ECG')
grid on
subplot(2,1,2)
plot(yobs,LineWidth=2)
xlabel('Time Index'); ylabel('Amplitude'); 
title("Observed Noisy ECG (Gaussian Noise with \mu = 0, \sigma^2 = " + Nvar)
grid on


% Error plots for convergence 
figure(2)  
f2s1 = subplot(4,1,1); % KF
plot(e_meas,'LineStyle','-',LineWidth=1.5)
title("Estimation Error for KF-SS (mean value is " + mean(abs(e_meas)) + ")")
xlabel('Time Index'); 
ylabel('Magnitude ')
% f2s1.YAxis.Limits = [0 1.5];
grid on
f2s2 = subplot(4,1,2); % LMS
plot(eLMS,'LineStyle','-',LineWidth=1.5)
title("Estimation Error for LMS (mean value is " + mean(abs(eLMS)) + ")")
xlabel('Time Index');
ylabel('Magnitude ')
% f2s2.YAxis.Limits = [0 1.5];
grid on
f2s3 = subplot(4,1,3); % NLMS
plot(eNLMS,'LineStyle','-',LineWidth=1.5)
title("Estimation Error for NLMS (mean value is " + mean(abs(eNLMS)) + ")")
xlabel('Time Index'); 
ylabel('Magnitude ')
% f2s3.YAxis.Limits = [0 1.5];
grid on
f2s4 = subplot(4,1,4); % RLS
plot(eRLS,'LineStyle','-',LineWidth=1.5)
title("Estimation Error for RLS (mean value is " + mean(abs(eRLS)) + ")")
xlabel('Time Index'); 
ylabel('Magnitude ')
% f2s4.YAxis.Limits = [0 1.5];
grid on

% MSE plots
figure(3)
subplot(4,1,1) % KF
plot(10*log10(abs(e_meas.*e_meas)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for KF-SS (mean value is " + mean(10*log10(abs(e_meas.*e_meas))) + "dB)")
xlabel('Time Index'); ylabel('Magnitude (dB)')
grid on
subplot(4,1,2) % LMS
plot(10*log10(abs(eLMS.*eLMS)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for LMS (mean value is " + mean(10*log10(abs(eLMS(2:end).*eLMS(2:end)))) + "dB)")
xlabel('Time Index'); ylabel('Magnitude (dB)')
grid on
subplot(4,1,3) % NLMS
plot(10*log10(abs(eNLMS.*eNLMS)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for NLMS (mean value is " + mean(10*log10(abs(eNLMS(2:end).*eNLMS(2:end)))) + "dB)")
xlabel('Time Index'); ylabel('Magnitude (dB)')
grid on
subplot(4,1,4) % RLS
plot(10*log10(abs(eRLS.*eRLS)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for RLS (mean value is " + mean(10*log10(abs(eRLS(2:end).*eRLS(2:end)))) + "dB)")
xlabel('Time Index'); ylabel('Magnitude (dB)')
grid on
