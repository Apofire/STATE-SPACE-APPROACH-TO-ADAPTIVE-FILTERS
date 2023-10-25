% Final Project EEE 606 
% Kaushik Iyer (1223696175)

%% State Space Least Mean Squares (SSLMS) Algorithm
% Application to track a sinusoid with unknown amplitude and phase but known
% frequency. Such a case case arises in case of Power Line Interference (PLI) 
% for ECG signals.  Signal model: y[n] = σ sin(ωnΤ + φ) + ν(nT) 
% where σ = (a^2)/2 is the signal power and a is the amplitude
%       ω is the known frequency (60 Hz)
%       T is the sampling time (s)
%       φ is the singal phase (rad)
%       ν is the observation noise
%       y is the observed signal corrupted by noise 
% Definitions and Initialisations
n     = 2;      % Dimension of state space
x     = [1 0]'; % State of the system (initialised to [1 0]' -> [amp, phase]')
Ts    = 1;      % Sampling time 
N     = 1000;   % Number of iterations
mu    = 0.8;
% Parameters of the actual and observed signal
phi    =  pi/4;                             % Phase 
a      = 1.5;                               % Amplitude
wo     = 0.1;                               % Known frequency 
t      = 0:1:N-1;      t = t';              % Time vector
sigma2 = 0.4;
nu     = sigma2*randn(N,1);                 % Noise 
yobs   = ((a^2)/2)*cos(wo*t*Ts + phi) + nu; % Observation signal
y      = yobs - nu;                         % Clean signal
% System matrices for SSLMS 
A = [cos(wo*Ts) sin(wo*Ts);
     -sin(wo*Ts)  cos(wo*Ts)];
C = [1 0];
G = eye(2);
K = [1;1]; 
% Initialise vectors for plots
eSSLMS  = zeros(N,1);  yhat    = zeros(N,1);   xhat = x;
% Loop for SSLMS
for k = 1:N
    xbar = A*xhat;                  % State vector preditcion 
    ybar = C*xbar;                  % Output prediction
    epsilon = yobs(k) - ybar;       % Output prediction error 
    xhat = xbar + K*epsilon;        % State vector update using output error
    yhat(k)   = C*xhat;             % Output estimation 
    eSSLMS(k) = yobs(k) - yhat(k);  % Output estimation error    
    K = mu*G*C'*inv(C*C');          % Compute the observer gain
end

% Initialisations for SSLMSWAM
eSSLMSWAM    = zeros(N,1);
yhatSSLMSWAM = zeros(N,1);
psi          = zeros(n,1);
muSSLMSWAM   = 0.1;
alpha        = 0.01; 

% Loop for SSLMSWAM
for k = 1:N
    xbar = A*xhat;    % State vector preditcion 
    ybar = C*xbar;    % Output prediction

    epsilon = yobs(k) - ybar;  % Output prediction error 
    xhat = xbar + K*epsilon;   % State vector update using output error

    yhatSSLMSWAM(k)   = C*xhat;                % Output estimation 
    eSSLMSWAM(k) = yobs(k) - yhatSSLMSWAM(k);  % Output estimation error 
    
    K          = muSSLMSWAM*G*C';             % Compute the observer gain
    psi        = (A - K*C*A)*psi + G*C'*epsilon;        % Compute gradient for mu
    muSSLMSWAM = muSSLMSWAM + alpha*psi'*A'*C'*epsilon; % Update muSSLMSWAM
end

% Initialisations for LMS 
L       = 3;             % Order of FIR Filter
w       = zeros(L+1,1);  % Filter weights
muLMS   = 0.005;         % step size 
xBuffer = zeros(L+1,1);  % Convolution buffer
eLMS    = zeros(N,1);    % Output error
yLMS    = zeros(N,1);    % LMS Output 

% Loop for LMS
for k = 1:N 
    xBuffer = [yobs(k); xBuffer(1:end-1)];  % Update buffer
    yLMS(k) = w'*xBuffer;                   % Compute output 
    eLMS(k) = y(k) - yLMS(k);               % Compute error
    w       = w + muLMS*eLMS(k)*xBuffer;    % Update filter weights
end

% Initialisations for RLS      
beta    = 0.99;                 % Forgetting factor for RLS
betaInv = 1/beta;               % Inverse of beta
delta   = 1e-3;                    % Parameter to Initialise Rxx
RxxInv  = (1/delta)*eye(L+1);   % Rxx^(-1) Initialisation
eRLS    = zeros(N,1);           % Output error
yRLS    = zeros(N,1);           % RLS Output 
wRLS    = zeros(L+1,1);         % RLS Filter weights

% Loop for RLS
for k = 1:N
    xBuffer = [yobs(k); xBuffer(1:end-1)];                                    % Update the buffer
    KRLS    = (betaInv*RxxInv*xBuffer)/(1 + betaInv*xBuffer'*RxxInv*xBuffer); % Gain computation
    yRLS(k) = wRLS'*xBuffer;                                                  % Compute output 
    eRLS(k) = y(k) - wRLS'*xBuffer;                                           % Error computation
    wRLS    = wRLS + KRLS*eRLS(k);                                            % Weight update
    RxxInv  = betaInv*RxxInv - betaInv*KRLS*xBuffer'*RxxInv;                  % RxxInv update   
end

% Initialisations for NLMS 
muNLMS   = 0.05;         % step size 
eNLMS    = zeros(N,1);    % Output error
yNLMS    = zeros(N,1);    % LMS Output 

% Loop for NLMS 
for k = 1:N
    xBuffer = [yobs(k); xBuffer(1:end-1)];                      % Update buffer
    yNLMS(k) = w'*xBuffer;                                      % Compute output 
    eNLMS(k) = y(k) - yNLMS(k);                                 % Compute error
    w        = w + (muNLMS*eNLMS(k)/(norm(xBuffer)^2))*xBuffer; % Update filter weights
end

%% Plots 

% Clean and Observed Sinusoids
figure(1) 
subplot(2,1,1)
plot(y,'LineStyle','-',LineWidth=2)
xlabel('Time Index'); ylabel('Amplitude'); 
title('Clean Sinusoid')
grid on
subplot(2,1,2)
plot(yobs,LineWidth=2)
xlabel('Time Index'); ylabel('Amplitude'); 
title("Observed Noisy Sinusoid (Gaussian Noise with \mu = 0, \sigma^2 = " + sigma2 + ")")
grid on

% Error plots for convergence 
figure(2)  
f2s1 = subplot(4,1,1); % SSLMS
plot(abs(eSSLMS),'LineStyle','-',LineWidth=1.5)
title("Error for SSLMS (mean value is " + mean(abs(eSSLMS)) + ")")
xlabel('Time Index'); 
ylabel('Magnitude ')
f2s1.YAxis.Limits = [0 1.5];
grid on
f2s2 = subplot(4,1,2); % LMS
plot(abs(eLMS),'LineStyle','-',LineWidth=1.5)
title("Error for LMS (mean value is " + mean(abs(eLMS)) + ")")
xlabel('Time Index');
ylabel('Magnitude ')
f2s2.YAxis.Limits = [0 1.5];
grid on
f2s3 = subplot(4,1,3); % NLMS
plot(abs(eNLMS),'LineStyle','-',LineWidth=1.5)
title("Error for NLMS (mean value is " + mean(abs(eNLMS)) + ")")
xlabel('Time Index'); 
ylabel('Magnitude ')
f2s3.YAxis.Limits = [0 1.5];
grid on
f2s4 = subplot(4,1,4); % RLS
plot(abs(eRLS),'LineStyle','-',LineWidth=1.5)
title("Error for RLS (mean value is " + mean(abs(eRLS)) + ")")
xlabel('Time Index'); 
ylabel('Magnitude ')
f2s4.YAxis.Limits = [0 1.5];
grid on


% MSE plots 
figure(3)  
subplot(4,1,1) % SSLMS
plot(10*log10(abs(eSSLMS.*eSSLMS)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for SSLMS (mean value is " + mean(10*log10(abs(eSSLMS(2:end).*eSSLMS(2:end)))) + "dB)")
xlabel('Time Index'); ylabel('Magnitude (dB)')
grid on
subplot(4,1,2) % LMS
plot(10*log10(abs(eLMS.*eLMS)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for LMS (mean value is " + mean(10*log10(abs(eLMS.*eLMS))) + "dB)")
xlabel('Time Index'); ylabel('Magnitude (dB)')
grid on
subplot(4,1,3) % NLMS
plot(10*log10(abs(eNLMS.*eNLMS)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for NLMS (mean value is " + mean(10*log10(abs(eNLMS.*eNLMS))) + "dB)")
xlabel('Time Index'); ylabel('Magnitude (dB)')
grid on
subplot(4,1,4) % RLS
plot(10*log10(abs(eRLS.*eRLS)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for RLS (mean value is " + mean(10*log10(abs(eRLS.*eRLS))) + "dB)")
xlabel('Time Index'); ylabel('Magnitude (dB)')
grid on

