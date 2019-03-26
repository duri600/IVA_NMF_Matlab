clc
close all
clear all

maxNum = 99;

for sNum = 1:maxNum
    disp(['Sentence #',num2str(sNum)]);
    [x,fs] = audioread(['./mixture/x_2x2_S',num2str(sNum),'.wav']);
    
    nfft = 4096;
    nol = fix(3*nfft/4);
    nshift = nfft-nol;
    maxiter = 50;
    rank = 4;
    z_option = 0;
    outdir = ['v03_output/rank_',num2str(rank),'/z_option_',num2str(z_option)];
    mkdir(outdir);
    
    
    
    %% STFT
    [X_tmp, window] = STFT( x, nfft, nshift, 'hamming' );
    X = X_tmp; %% Freq:I, Frame:J, Mic:M
    [Nfreq, Nframe, Nmic] = size(X);
    
    %% Spectrogram
    figure;
    hold on
    for micInd = 1:Nmic
        subplot(Nmic,1,micInd);
        spectrogram(x(:,micInd),512,384,512,fs,'yaxis');
        colorbar
        colormap jet;
        %             caxis([cmin cmax])
        xlabel('Time(s)')
        ylabel('Frequency(kHz)')
        title(['Sensor ',num2str(micInd)])
        axis xy
    end
    hold off
    pause(0.01)
    
    %% Processing
    % spatial model Initialize
    V_IVA = zeros(Nmic,Nmic,Nfreq);
    W_IVA = zeros(Nmic,Nmic,Nfreq);% inverse of the mixing matrix
    W_im = zeros(Nmic,Nmic,Nfreq);
    w_im = zeros(Nmic,Nfreq);
    wVw = zeros(1,Nfreq);
    r = zeros(Nfreq,Nframe,Nmic);
    Y = zeros(Nfreq,Nframe,Nmic);
    
    % source model Initialize
    if z_option == 0
        T = rand(Nfreq, rank, Nmic);
        V_act = rand(rank,Nframe,Nmic);
    else
        KK = rank*Nmic;
        T = rand(Nfreq, KK);
        V_act = rand(KK,Nframe);
        ZZ = rand(Nmic,KK);
    end
    
    for i = 1:Nfreq
        W_IVA(:,:,i) = eye(Nmic);
        Y(i,:,:) = (W_IVA(:,:,i)*squeeze(X(i,:,:)).').';
    end
    
    % P = max(abs(Y).^2,eps);
    Xp = permute(X,[3,2,1]); % Nmic x Nframe x Nfreq
    
    % Spatial Model
    fprintf('Iteration:    ');
    for iter = 1:maxiter
        fprintf('\b\b\b\b%4d', iter);
        for m = 1:Nmic
            % Aux R
            XpHt = conj( permute( Xp, [2,1,3] ) ); % Nframe x Nmic x Nfreq (matrix-wise Hermitian transpose)
            
            for i = 1:Nfreq
                % (24)
                V_IVA(:,:,i) = (Xp(:,:,i)./(r(i,:,m) + eps))*XpHt(:,:,i)/Nframe; % Nmic x Nmic x Nfreq
                % (25)
                W_im(:,:,i) = inv(W_IVA(:,:,i) * V_IVA(:,:,i));
                w_im(:,i) = squeeze(W_im(:,m,i));
            end
            
            for i = 1:Nfreq
                wVw(1,i) = w_im(:,i)'*V_IVA(:,:,i)*w_im(:,i);
            end
            
            w = (w_im./max(sqrt(wVw),eps)).';
            W_IVA(m,:,:) = w';
            
            for i = 1:Nfreq
                Y(i,:,m) = W_IVA(m,:,i)* Xp(:,:,i);
            end
        end
        
        % normalize
        for i = 1:Nfreq
            Wmdp(:,:,i) = diag(diag(pinv(W_IVA(:,:,i))))*W_IVA(:,:,i);
            Y(i,:,:) = (Wmdp(:,:,i)*Xp(:,:,i)).';
        end
        
        %% Source model(NMF)
        if z_option == 0
            %% eliminate partitioning function z
            for i = 1:Nfreq
                for l = 1:rank
                    for m = 1:Nmic
                        num_temp = 0;
                        denom_temp = 0;
                        for j = 1:Nframe
                            commonVal = 0;
                            for l_ = 1:rank
                                commonVal = commonVal + T(i,l_,m)*V_act(l_,j,m);
                            end
                            num_temp = num_temp +  (abs(Y(i,j,m))^2)*V_act(l,j,m)/(commonVal^2);
                            denom_temp = denom_temp + V_act(l,j,m)/commonVal;
                        end
                        T(i,l,m) = T(i,l,m) * max(sqrt(num_temp/denom_temp),eps);
                    end
                end
            end
            
            for l = 1:rank
                for j = 1:Nframe
                    for m = 1:Nmic
                        num_temp = 0;
                        denom_temp = 0;
                        for i = 1:Nfreq
                            commonVal = 0;
                            for l_ = 1:rank
                                commonVal = commonVal + T(i,l_,m)*V_act(l_,j,m);
                            end
                            num_temp = num_temp + (abs(Y(i,j,m))^2)*T(i,l,m)/(commonVal^2);
                            denom_temp = denom_temp + T(i,l,m)/commonVal;
                        end
                        V_act(l,j,m) = V_act(l,j,m) * max(sqrt(num_temp/denom_temp),eps);
                    end
                end
            end
            
            for i = 1:Nfreq
                for j = 1:Nframe
                    for m = 1:Nmic
                        for l = 1:rank
                            commonval = max(T(i,l,m)*V_act(l,j,m),eps);
                            r(i,j,m) = r(i,j,m) + commonval;
                        end
                    end
                end
            end
        else
            %% employ a continuous-valued z
            for m = 1:Nmic
                for k = 1:KK
                    num_temp = 0;
                    denom_temp = 0;
                    for i = 1:Nfreq
                        for j = 1:Nframe
                            commonVal = 0;
                            for k_ = 1:KK
                                commonVal = commonVal + ZZ(m,k_)*T(i,k_)*V_act(k_,j);
                            end
                            num_temp = num_temp + (abs(Y(i,j,m))^2)*T(i,k)*V_act(k,j)/(commonVal^2);
                            denom_temp = denom_temp + T(i,k)*V_act(k,j)/commonVal;
                        end
                    end
                    ZZ(m,k) = ZZ(m,k)*max(sqrt(num_temp/denom_temp),eps);
                end
            end
            for k = 1:KK
                ZZ(:,k) = ZZ(:,k)./sum(ZZ(:,k));
            end
            
            for i = 1:Nfreq
                for k = 1:KK
                    num_temp = 0;
                    denom_temp = 0;
                    for j = 1:Nframe
                        for m = 1:Nmic
                            commonVal = 0;
                            for k_ = 1:KK
                                commonVal = commonVal + ZZ(m,k_)*T(i,k_)*V_act(k_,j);
                            end
                            num_temp = num_temp + (abs(Y(i,j,m))^2)*ZZ(m,k)*V_act(k,j)/(commonVal^2);
                            denom_temp = denom_temp + ZZ(m,k)*V_act(k,j)/commonVal;
                        end
                    end
                    T(i,k) = T(i,k)*max(sqrt(num_temp/denom_temp),eps);
                end
            end
            
            for k = 1:KK
                for j = 1:Nframe
                    num_temp = 0;
                    denom_temp = 0;
                    for i = 1:Nfreq
                        for m = 1:Nmic
                            commonVal = 0;
                            for k_ = 1:KK
                                commonVal = commonVal + ZZ(m,k_)*T(i,k_)*V_act(k_,j);
                            end
                            num_temp = num_temp + (abs(Y(i,j,m))^2)*ZZ(m,k)*T(i,k)/(commonVal^2);
                            denom_temp = denom_temp + ZZ(m,k)*T(i,k)/commonVal;
                        end
                    end
                    V_act(k,j) = V_act(k,j)*max(sqrt(num_temp/denom_temp),eps);
                end
            end
            
            r = zeros(Nfreq,Nframe,Nmic);
            for i = 1:Nfreq
                for j = 1:Nframe
                    for m = 1:Nmic
                        for k = 1:KK
                            commonval = max(ZZ(m,k)*T(i,k)*V_act(k,j),eps);
                            r(i,j,m) = r(i,j,m) + commonval;
                        end
                    end
                end
            end
        end
    end
    
    %% ISTFT
    y = ISTFT( Y, nshift, window, size(x,1) );
    
    %% Show Spectrogram
    figure;
    hold on
    for micInd = 1:Nmic
        subplot(Nmic,1,micInd);
        spectrogram(y(:,micInd),512,384,512,fs,'yaxis');
        colorbar
        colormap jet;
        %             caxis([cmin cmax])
        xlabel('Time(s)')
        ylabel('Frequency(kHz)')
        title(['Output ',num2str(micInd)])
        axis xy
    end
    hold off
    pause(0.01)
    
    for micInd = 1:Nmic
        audiowrite([outdir,'/output_S',num2str(sNum),'_ch',num2str(micInd),'.wav'],y(:,micInd),fs);
    end
end