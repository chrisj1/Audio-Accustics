spectro = function(path, nfft=1024, window=256, overlap=128, t0=0, plot_spec = T, normalize = F, return_data = F,...){
        
        library(signal)
        library(oce)
        
        data = readWave(path)
        
        # extract signal
        snd = data@left
        
        # demean to remove DC offset
        snd = snd-mean(snd)
        
        # determine duration
        dur = length(snd)/data@samp.rate
        print(data@samp.rate)
        
        # create spectrogram
        spec = specgram(x = snd,
                        n = nfft,
                        Fs = data@samp.rate,
                        window = window,
                        overlap = overlap
        )
        
        # discard phase info
        P = abs(spec$S)
        
        # normalize
        if(normalize){
                P = P/max(P)  
        }
        
        # convert to dB
        P = 10*log10(P)
        
        # config time axis
        if(t0==0){
                t = as.numeric(spec$t)
        } else {
                t = as.POSIXct(spec$t, origin = t0)
        }
        
        # rename freq
        f = spec$f
        
        if(plot_spec){
                
                # change plot colour defaults
                par(bg = "black")
                par(col.lab="white")
                par(col.axis="white")
                par(col.main="white")
                
                # plot spectrogram
                imagep(t,f, t(P), col = oce.colorsViridis, drawPalette = T,
                       ylab = 'Frequency [Hz]', axes = F,...)
                
                box(col = 'white')
                axis(2, labels = T, col = 'white')
                
                # add x axis
                if(t0==0){
                        
                        axis(1, labels = T, col = 'white')
                        
                }else{
                        
                        axis.POSIXct(seq.POSIXt(t0, t0+dur, 10), side = 1, format = '%H:%M:%S', col = 'white', las = 1)
                        mtext(paste0(format(t0, '%B %d, %Y')), side = 1, adj = 0, line = 2, col = 'white')
                        
                }
        }
        
        if(return_data){
                
                # prep output
                spec = list(
                        t = t,
                        f = f,
                        p = t(P)
                )
                
                return(spec)  
        }
}

# call the spectrogram function
print(spectro(path='/Users/chrisjerrett/Desktop/sounds/652019/5161.190605114502.wav',
        nfft=4096,
        window=4096,
        overlap=128,
        t0=0,
        plot_spec = T,
        normalize = T,
        return_data = T
))

