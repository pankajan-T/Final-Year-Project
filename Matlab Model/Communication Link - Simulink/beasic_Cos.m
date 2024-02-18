clc;

clear;


t =0: 0.0001 : 0.1;

Vi= 5;
Vq= 6;

f= 50 ;

ph1=pi/6;

Vs=Vi*sin(2*pi*f*t);
Vc=Vq*cos(2*pi*f*t);

%plot(t,Vs);

%generate combined signal

Vp = Vs+Vc;

%plot(t,Vp);



%Accessing in-phase compoenent
Vi1=transpose(sin(2*pi*f*t));

Vpc1=Vp*Vi1;


% size(transpose(sin(2*pi*f*t)))


plot(t,Vpc1);


grid on;
