function Score = GLV_SIM(Ref_img, Dis_img)

% ========================================================================
% GLV_SIM Index with automatic downsampling
% Copyright(c) 2019 Gao Minjuan,Dang Hongshe,Wei Lili,Wang Hailong and Zhang Xuande
% All Rights Reserved.
%----------------------------------------------------------------------
% This is an implementation of the algorithm presented in the following
% paper: 
% Gao Minjuan,Dang Hongshe,Wei Lili,Wang Hailong and Zhang Xuande, Combining Global and Local Variation 
% for Image Quality Assessment, Acta Automatica Sinica, VOL. , NO. , 2019
%----------------------------------------------------------------------
%Input : (1) Ref_img: reference image
%        (2) Dis_img: distorted image
%
%Output: (1) Score: the objective score given by using GLV_SIM IQA metric
%-----------------------------------------------------------------------
% Email: zhangxuande@sust.edu.cn 
% 2019-12-15

Ref_img = double(Ref_img);
Dis_img = double(Dis_img);

if size(Ref_img,3)==3 
    Ref_img = 0.299 * double(Ref_img(:,:,1)) + 0.587 * double(Ref_img(:,:,2)) + 0.114 * double(Ref_img(:,:,3));
    Dis_img = 0.299 * double(Dis_img(:,:,1)) + 0.587 * double(Dis_img(:,:,2)) + 0.114 * double(Dis_img(:,:,3));
end
%==================================
% automatic downsampling
[M,N]=size(Ref_img);
f = max(1,round(min(M,N)/256));
%downsampling by f
%use a simple low-pass filter 
if(f>1)
    lpf = ones(f,f);
    lpf = lpf/sum(lpf(:));
    Ref_img = imfilter(Ref_img,lpf,'symmetric','same');
    Dis_img = imfilter(Dis_img,lpf,'symmetric','same');

    Ref_img = Ref_img(1:f:end,1:f:end);
    Dis_img = Dis_img(1:f:end,1:f:end);
end
%=================================
%parameters
a=0.6;
h=80; 
L=255;
k1=0.2;
k2=0.1;
C=(k1*L).^2;
T=(k2*L).^2;
%================================
%Global variation
DM1 = fgl_deriv_maxtrix_norm(a,Ref_img,h);
DM2 = fgl_deriv_maxtrix_norm(a,Dis_img,h);

F_map=(2*DM1.*DM2+C)./(DM1.^2+DM2.^2+C);
%================================
% Local variation
dx=[3 0 -3;10 0 -10;3 0 -3]/16;%scharr
dy=dx';

Ref_dx=conv2(Ref_img,dx,'same');
Ref_dy=conv2(Ref_img,dy,'same');
Ref_GM=(sqrt(Ref_dx.^2+Ref_dy.^2).^1);

Dis_dx=conv2(Dis_img,dx,'same');
Dis_dy=conv2(Dis_img,dy,'same');
Dis_GM=(sqrt(Dis_dx.^2+Dis_dy.^2).^1);

GM_map=(2*Ref_GM.*Dis_GM+T)./(Ref_GM.^2+Dis_GM.^2+T);
%================================
% combining
map=(F_map.^0.7).*(GM_map.^0.3);
%pooling                  
Score=mean2(map);
end



function DM = fgl_deriv_maxtrix_norm( a, Y, h )
%%     'horizon'
        [m,n]  = size(Y);
        J  = 0:(n-1);
        G1 = gamma( J+1 );
        G2 = gamma( a+1-J );
        s  = (-1) .^ J;
        M  = tril( ones(n) );
        T  = meshgrid( (gamma(a+1)/(h^a)) * s ./ (G1.*G2) );
        tt1=(gamma(a+1)/(h^a)) * s ./ (G1.*G2);
        for row=1:m
            R  = toeplitz( Y(row,:)' );
            Dx(row,:) = reshape(sum( R .* M .* T, 2 ), [1,n]);
        end
%%  vertical
        Y=Y';
        [m,n]  = size(Y);
        J  = 0:(n-1);
        G1 = gamma( J+1 );
        G2 = gamma( a+1-J );
        s  = (-1) .^ J;
        M  = tril( ones(n) );
        T  = meshgrid( (gamma(a+1)/(h^a)) * s ./ (G1.*G2) );
        tt2=(gamma(a+1)/(h^a)) * s ./ (G1.*G2);
        
        for row=1:m
            R  = toeplitz( Y(row,:)' );
            Dy(row,:) = reshape(sum( R .* M .* T, 2 ), [1,n]);
        end
        Dy=Dy';
        
        DM=sqrt(Dx.^2+Dy.^2);
end