% Final Project EEE 606 
% Kaushik Iyer (1223696175)

%% State Space Least Mean Squares With Adaptive Memory (SSLMSWAM) Algorithm
% Application to track a Van der Pol oscillations defined by the equations:
% x1' = x2
% x2' = -x1 + m(1 - x1^2)x2
% The state space is given by the constant acceleration model and the
% signal x2 is observed


% Definitions and Initialisations
n             = 3;          % Dimension of state space
x             = [0 0 0]';   % State of the system 
Ts            = 0.01;       % Sampling time 
N             = 2500;       % Number of iterations
muSSLMSWAM(1) = 0.1;        % Initial step size
alpha         = 0.01;       % Learning parameter
psi           = zeros(n,1); % Initial gradient

% System and Gain matrices
A = [1 Ts 0.5*Ts^2; 0 1 Ts; 0 0 1]; % State propagation matrix
C = [1 0 0];                        % Observation matrix
G = [1 0 0; 0.3 0 0; 0.3 0 0];      % Constant Gain

% Generate Van der Pol oscillation
x0   = [2;0];
y    = GenVanderPolOsc(N,2,x0);
y    = y.y(end,1:100)';         % extract only the last column as we observe x2
Nvar = 0.1;                     % Noise variance
nu   = Nvar*randn(length(y),1); % Generate noise  
yobs = y + nu;                  % Observed oscillation (noisy)

epsilon                 = zeros(length(y),1); % Store Output error
muSSLMSWAM(2:length(y)) = 0;                  % Store mu
% Loop for SSLMSWAM
for i = 1:length(y)
    K             = muSSLMSWAM(i)*G*C'/(C*C');                    % Compute Observer Gain 
    epsilon(i)    = yobs(i) - C*A*x;                              % Compute Output Error
    x             = A*x + K*epsilon(i);                           % Update state 
    muSSLMSWAM(i) = muSSLMSWAM(i) + alpha*psi'*A'*C'*epsilon(i);  % Update mu    
    psi           = (A - K*C*A)*psi + G*C'*epsilon(i);            % Update gradient
end

% Initialisations for LMS, NLMS and RLS
L = 3;                        % FIR filter order
muLMS = 0.1;                  % Step size LMS
eLMS = zeros(length(y),1);  % Output error LMS 
wLMS = zeros(L+1,1);          % LMS Filter weights
yLMS = zeros(length(y),1);    % LMS Output 
beta = 0.1;                  % Forgetting factor for RLS
betaInv = 1/beta;             % Inverse of beta
delta   = 1e-3;               % Parameter to Initialise Rxx
RxxInv  = (1/delta)*eye(L+1); % Rxx^(-1) Initialisation
eRLS    = zeros(length(y),1); % Output error RLS
yRLS    = zeros(length(y),1); % RLS Output 
wRLS    = zeros(L+1,1);       % RLS Filter weights
muNLMS  = 0.9;                % Step size NLMS
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

% Clean and Observed Van der Pol oscillations
figure(1) 
subplot(2,1,1)
plot(y,'LineStyle','-',LineWidth=2)
xlabel('Time Index'); ylabel('Amplitude'); 
title('Clean Sinusoid')
grid on
subplot(2,1,2)
plot(yobs,LineWidth=2)
xlabel('Time Index'); ylabel('Amplitude'); 
title("Observed Noisy Sinusoid (Gaussian Noise with \mu = 0, \sigma^2 = " + Nvar)
grid on

% Error plots for convergence 
figure(2)  
f2s1 = subplot(4,1,1); % SSLMSWAM
plot(epsilon,'LineStyle','-',LineWidth=1.5)
title("Estimation Error for SSLMSWAM (mean value is " + mean(abs(epsilon)/5) + ")")
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
subplot(4,1,1) % SSLMSWAM
plot(10*log10(abs(epsilon.*epsilon/50)),'LineStyle','-',LineWidth=1.5)
title("MSE in dB for SSLMSWAM (mean value is " + mean(10*log10(abs(epsilon.*epsilon/50))) + "dB)")
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

% % Adaptation of mu 
% figure(4)
% plot(muSSLMSWAM,'LineStyle','-.',LineWidth=1.5)
% title('Adaptation of \mu for SSLMSWAM')
% xlabel('Time Index'); ylabel('\mu')
% grid on

%% Function to generate Van der Pol oscillations
function y = GenVanderPolOsc(N,m,y0)
% This function generates Van der Pol oscillation for a given damping
% coefficient m and for a given number of samples N, and initial condition y0.
% Inputs -> N (Number of samples)
%        -> m (Damping factor)
%        -> y0 (Initial condition)
% Output -> y (Van der Pol oscillation vector)
% -------------------------------------------------------------------------

% y = ode45( @(t,y) vdp1(t,y,m), [0 N], y0);

function dydt = vdp1(t,y)
  dydt = [y(2); m*(1 - y(1)^2)*y(2) - y(1)]; 
end

y = ode45(@vdp1,[0 N],y0);
end


