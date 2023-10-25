function ECGsig = genECG(t)

x = 0.01:0.01:t;
default=2;%input('Press 1 if u want default ecg signal else press 2:\n'); 
if(default==1)
      li=30/72;  
    
      a_pwav=0.25;
      d_pwav=0.09;
      t_pwav=0.16;  
     
      a_qwav=0.025;
      d_qwav=0.066;
      t_qwav=0.166;
      
      a_qrswav=1.6;
      d_qrswav=0.11;
      
      a_swav=0.25;
      d_swav=0.066;
      t_swav=0.09;
      
      a_twav=0.35;
      d_twav=0.142;
      t_twav=0.2;
      
      a_uwav=0.035;
      d_uwav=0.0476;
      t_uwav=0.433;
else
    rate= 125;%input('\n\nenter the heart beat rate :');
    li=30/rate;
    
    %p wave specifications
%     fprintf('\n\np wave specifications\n');
    d=1;%input('Enter 1 for default specification else press 2: \n');
    if(d==1)
        a_pwav=0.25;
        d_pwav=0.09;
        t_pwav=0.16;
    else
       a_pwav=input('amplitude = ');
       d_pwav=input('duration = ');
       t_pwav=input('p-r interval = ');
       d=0;
    end    
      
    %q wave specifications
%     fprintf('\n\nq wave specifications\n');
%     d=input('Enter 1 for default specification else press 2: \n');
    if(d==1)
        a_qwav=0.025;
        d_qwav=0.066;
        t_qwav=0.166;
    else
       a_qwav=input('amplitude = ');
       d_qwav=input('duration = ');
       t_qwav=0.166;
       d=0;
    end    
        
    %qrs wave specifications
%     fprintf('\n\nqrs wave specifications\n');
%     d=input('Enter 1 for default specification else press 2: \n');
    if(d==1)
        a_qrswav=1.6;
        d_qrswav=0.11;
    else
       a_qrswav=input('amplitude = ');
       d_qrswav=input('duration = ');
       d=0;
    end            
    
    %s wave specifications
%     fprintf('\n\ns wave specifications\n');
%     d=input('Enter 1 for default specification else press 2: \n');
    if(d==1)
        a_swav=0.25;
        d_swav=0.066;
        t_swav=0.09;
    else
       a_swav=input('amplitude = ');
       d_swav=input('duration = ');
       t_swav=0.09;
       d=0;
    end       
    
    %t wave specifications
%     fprintf('\n\nt wave specifications\n');
%     d=input('Enter 1 for default specification else press 2: \n');
    if(d==1)
        a_twav=0.35;
        d_twav=0.142;
        t_twav=0.2;
    else
       a_twav=input('amplitude = ');
       d_twav=input('duration = ');
       t_twav=input('s-t interval = ');
       d=0;
    end    
      
    %u wave specifications
%     fprintf('\n\nu wave specifications\n');
%     d=input('Enter 1 for default specification else press 2: \n');
    if(d==1)
        a_uwav=0.035;
        d_uwav=0.0476;
        t_uwav=0.433;
    else
       a_uwav=input('amplitude = ');
       d_uwav=input('duration = ');
       t_uwav=0.433;
       d=0;
    end        
end

 pwav=p_wav(x,a_pwav,d_pwav,t_pwav,li);  % pwav output 
 qwav=q_wav(x,a_qwav,d_qwav,t_qwav,li);  % qwav output
 qrswav=qrs_wav(x,a_qrswav,d_qrswav,li); % qrswav output
 swav=s_wav(x,a_swav,d_swav,t_swav,li);  % swav output
 twav=t_wav(x,a_twav,d_twav,t_twav,li);  % twav output
 uwav=u_wav(x,a_uwav,d_uwav,t_uwav,li);  % uwav output

 %ecg output
 ECGsig = pwav + qrswav + twav + swav + qwav + uwav;

end

%% Individual functions 

function [pwav]=p_wav(x,a_pwav,d_pwav,t_pwav,li)
l=li;
a=a_pwav;
x=x+t_pwav;
b=(2*l)/d_pwav;
n=100;
p1=1/l;
p2=0;
for i = 1:n
    harm1=(((sin((pi/(2*b))*(b-(2*i))))/(b-(2*i))+(sin((pi/(2*b))*(b+(2*i))))/...
          (b+(2*i)))*(2/pi))*cos((i*pi*x)/l);             
    p2=p2+harm1;
end
pwav1=p1+p2;
pwav=a*pwav1;

end
% -------------------------------------------------------------------------

function [qwav]=q_wav(x,a_qwav,d_qwav,t_qwav,li)
l=li;
x=x+t_qwav;
a=a_qwav;
b=(2*l)/d_qwav;
n=100;
q1=(a/(2*b))*(2-b);
q2=0;
for i = 1:n
    harm5=(((2*b*a)/(i*i*pi*pi))*(1-cos((i*pi)/b)))*cos((i*pi*x)/l);
    q2=q2+harm5;
end
qwav=-1*(q1+q2);
end
% -------------------------------------------------------------------------

function [qrswav]=qrs_wav(x,a_qrswav,d_qrswav,li)
l=li;
a=a_qrswav;
b=(2*l)/d_qrswav;
n=100;
qrs1=(a/(2*b))*(2-b);
qrs2=0;
for i = 1:n
    harm=(((2*b*a)/(i*i*pi*pi))*(1-cos((i*pi)/b)))*cos((i*pi*x)/l);
    qrs2=qrs2+harm;
end
qrswav=qrs1+qrs2;
end
% -------------------------------------------------------------------------

function [swav]=s_wav(x,a_swav,d_swav,t_swav,li)
l=li;
x=x-t_swav;
a=a_swav;
b=(2*l)/d_swav;
n=100;
s1=(a/(2*b))*(2-b);
s2=0;
for i = 1:n
    harm3=(((2*b*a)/(i*i*pi*pi))*(1-cos((i*pi)/b)))*cos((i*pi*x)/l);
    s2=s2+harm3;
end
swav=-1*(s1+s2);
end
% -------------------------------------------------------------------------

function [twav]=t_wav(x,a_twav,d_twav,t_twav,li)
l=li;
a=a_twav;
x=x-t_twav-0.045;
b=(2*l)/d_twav;
n=100;
t1=1/l;
t2=0;
for i = 1:n
    harm2=(((sin((pi/(2*b))*(b-(2*i))))/(b-(2*i))+(sin((pi/(2*b))*(b+(2*i))))/...
          (b+(2*i)))*(2/pi))*cos((i*pi*x)/l);             
    t2=t2+harm2;
end
twav1=t1+t2;
twav=a*twav1;
end
% -------------------------------------------------------------------------

function [uwav]=u_wav(x,a_uwav,d_uwav,t_uwav,li)
l=li;
a=a_uwav;
x=x-t_uwav;
b=(2*l)/d_uwav;
n=100;
u1=1/l;
u2=0;
for i = 1:n
    harm4=(((sin((pi/(2*b))*(b-(2*i))))/(b-(2*i))+(sin((pi/(2*b))*(b+(2*i))))/...
          (b+(2*i)))*(2/pi))*cos((i*pi*x)/l);             
    u2=u2+harm4;
end
uwav1=u1+u2;
uwav=a*uwav1;
end
