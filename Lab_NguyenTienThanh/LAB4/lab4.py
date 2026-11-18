"""Lab 4 mini TTS: corpus inventory, unit selection, cross-fade, prosody, HMM tracks."""
from __future__ import annotations
import csv, json, math, time, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'LAB2'/'Lab2_data_figures'/'dataset_aug'
DATA=HERE/'lab4_data'
UNITS=DATA/'units'
OUT=HERE/'results'
AUDIO=HERE/'audio_output'
FIG=OUT/'figures'
SR=16000
LABEL_MAP={'bat':'bật','dung':'dừng','len':'lên','phai':'phải','trai':'trái','tat':'tắt','xuong':'xuống'}
LABEL_TO_DIR={v:k for k,v in LABEL_MAP.items()}
LABELS=list(LABEL_MAP.values())
DIGITS={'0':'không','1':'một','2':'hai','3':'ba','4':'bốn','5':'năm','6':'sáu','7':'bảy','8':'tám','9':'chín'}
QUEUE_CASE='Mời khách hàng số 23 đến quầy 5.'
# Candidate outputs stay within the words that really exist in the supplied corpus.
SENTENCES=[('S1','Bật lên, dừng xuống, trái phải tắt.'),('S2','Trái lên phải xuống.'),('S3','Tắt bật dừng.')]
QUEUE_UNITS=['mời','khách','hàng','số','hai','ba','đến','quầy','năm']
QUEUE_SLUG={'mời':'moi','khách':'khach','hàng':'hang','số':'so','hai':'hai','ba':'ba','đến':'den','quầy':'quay','năm':'nam'}
for p in [UNITS,OUT,FIG,AUDIO]:p.mkdir(parents=True,exist_ok=True)

def normalize_text(text):
    import re
    text=text.lower().replace('đ','đ')
    text=re.sub(r"[^\w\sÀ-ỹ]",' ',text,flags=re.UNICODE)
    tokens=[]
    for tok in text.split():
        if tok.isdigit():
            for digit in tok:tokens.append(DIGITS[digit])
        else:tokens.append(tok)
    return tokens

def rms(x):return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))+1e-12))

def load_unit(path):
    x,sr=sf.read(path,dtype='float32')
    if x.ndim>1:x=x.mean(axis=1)
    if sr!=SR:x=librosa.resample(x,orig_sr=sr,target_sr=SR);sr=SR
    x,_=librosa.effects.trim(x,top_db=35)
    if len(x)<int(.05*SR):raise ValueError(f'Unit too short after trim: {path}')
    # Preserve natural level variation, only guard against clipping.
    peak=float(np.max(np.abs(x)))
    if peak>.999:x=x*(.98/peak)
    return x.astype(np.float32),sr

def make_neural_units(model,tokenizer):
    from neural_tts_backend import synthesize
    d=DATA/'neural_generated_units';d.mkdir(parents=True,exist_ok=True);rows=[]
    for j,label in enumerate(QUEUE_UNITS):
        for variant,seed in [(1,4100+j*2),(2,4101+j*2)]:
            x,sr,latency=synthesize(label,model,tokenizer,seed=seed)
            if sr!=SR:x=librosa.resample(x,orig_sr=sr,target_sr=SR)
            path=d/f'{QUEUE_SLUG[label]}_neural_{variant:02d}.wav';sf.write(path,x,SR,subtype='PCM_16')
            rows.append({'label':label,'variant':variant,'seed':seed,'file':str(path.relative_to(HERE)),'generation_latency_s':latency,'duration_s':len(x)/SR})
    pd.DataFrame(rows).to_csv(OUT/'neural_unit_generation.csv',index=False)
    return rows

def inventory(neural_rows):
    rows=[];bank={}
    for label,folder in LABEL_TO_DIR.items():
        srcdir=SOURCE/folder
        if not srcdir.exists():raise FileNotFoundError(srcdir)
        bank[label]=[]
        for src in sorted(srcdir.glob('*.wav')):
            x,sr=load_unit(src); dst=UNITS/f'{folder}_{src.stem.split("_")[-1]}.wav'
            sf.write(dst,x,sr,subtype='PCM_16')
            info=sf.info(dst); row={'label':label,'speaker':'unknown (not annotated)','duration_s':len(x)/sr,'rms':rms(x),'sample_rate':sr,'channels':1,'subtype':info.subtype,'source_file':str(src.relative_to(SOURCE)),'path':str(dst.relative_to(HERE))}
            rows.append(row);bank[label].append({'path':dst,'x':x,'duration':len(x)/sr,'rms':rms(x),'id':src.stem})
    for q in neural_rows:
        label=q['label'];src=HERE/q['file'];x,sr=load_unit(src);slug=QUEUE_SLUG[label];dst=UNITS/f'{slug}_neural_{q["variant"]:02d}.wav';sf.write(dst,x,sr,subtype='PCM_16')
        bank.setdefault(label,[]);row={'label':label,'speaker':'MMS VITS single-speaker synthesis','duration_s':len(x)/sr,'rms':rms(x),'sample_rate':sr,'channels':1,'subtype':'PCM_16','source_file':f'facebook/mms-tts-vie generated variant {q["variant"]}','path':str(dst.relative_to(HERE))}
        rows.append(row);bank[label].append({'path':dst,'x':x,'duration':len(x)/sr,'rms':rms(x),'id':f'mms_{q["variant"]:02d}'})
    df=pd.DataFrame(rows);df.to_csv(DATA/'metadata.csv',index=False)
    counts=df.groupby('label').size().to_dict()
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    axs[0].hist(df.duration_s,bins=10,color='#2878a0',edgecolor='white');axs[0].set(title='Unit duration',xlabel='seconds',ylabel='count')
    axs[1].hist(df.rms,bins=10,color='#e18d45',edgecolor='white');axs[1].set(title='Unit RMS',xlabel='linear RMS',ylabel='count')
    fig.tight_layout();fig.savefig(FIG/'inventory_stats.png',dpi=160);plt.close(fig)
    return bank,df

def f0_track(x):
    frame=400;hop=160
    if len(x)<frame:return np.array([]),np.array([])
    f0=librosa.yin(x,fmin=70,fmax=450,sr=SR,frame_length=512,hop_length=hop)
    rmsf=librosa.feature.rms(y=x,frame_length=512,hop_length=hop,center=True)[0]
    f0=f0[:len(rmsf)];f0=np.where(rmsf>max(.003,.15*np.max(rmsf)),f0,np.nan)
    t=librosa.frames_to_time(np.arange(len(f0)),sr=SR,hop_length=hop)
    return t,f0

def edge_features(x,side,ms=25):
    n=max(64,int(SR*ms/1000));seg=x[-n:] if side=='left' else x[:n]
    if len(seg)<n:seg=np.pad(seg,(n-len(seg),0) if side=='left' else (0,n-len(seg)))
    win=np.hanning(n);spec=np.abs(np.fft.rfft(seg*win,n=512));logspec=20*np.log10(np.maximum(spec,1e-7))
    return logspec, rms(seg)

def join_cost(a,b):
    la,ra=edge_features(a,'left');lb,rb=edge_features(b,'right')
    # Spectral mismatch dominates; energy term is kept separately visible in the output table.
    spec=float(np.mean((la-lb)**2));energy=abs(np.log((ra+1e-8)/(rb+1e-8)))
    return spec+12.*energy,spec,energy

def select_units(tokens,bank):
    missing=sorted(set(tokens)-set(bank))
    if missing:raise KeyError('No recorded unit for: '+', '.join(missing))
    # Candidate target cost: z-scored duration and log-RMS deviation from label-level means.
    stats={w:(np.mean([q['duration'] for q in bank[w]]),np.std([q['duration'] for q in bank[w]])+1e-4,
              np.mean([np.log(q['rms']+1e-8) for q in bank[w]]),np.std([np.log(q['rms']+1e-8) for q in bank[w]])+.05) for w in set(tokens)}
    target={}
    for pos,w in enumerate(tokens):
        md,sd,mr,sr=stats[w];target[pos]=[((q['duration']-md)/sd)**2+.25*((np.log(q['rms']+1e-8)-mr)/sr)**2 for q in bank[w]]
    dp=[np.asarray(target[0],float)];back=[];joins=[]
    for pos in range(1,len(tokens)):
        prev=bank[tokens[pos-1]];cur=bank[tokens[pos]]; mat=np.zeros((len(prev),len(cur)));detail=[]
        for i,a in enumerate(prev):
            for j,b in enumerate(cur):
                jc,sc,ec=join_cost(a['x'],b['x']);mat[i,j]=jc;detail.append({'position':pos,'left':tokens[pos-1],'right':tokens[pos],'left_candidate':a['id'],'right_candidate':b['id'],'join_cost':jc,'spectral_cost':sc,'energy_cost':ec})
        val=dp[-1][:,None]+mat;bestprev=np.argmin(val,axis=0);dp.append(target[pos]+val[bestprev,np.arange(len(cur))]);back.append(bestprev);joins.extend(detail)
    ix=int(np.argmin(dp[-1]));path=[ix]
    for bp in reversed(back):ix=int(bp[ix]);path.append(ix)
    path.reverse();selected=[bank[w][j] for w,j in zip(tokens,path)]
    total=float(np.min(dp[-1]));return selected,total,pd.DataFrame(joins)

def concatenate(units,xfade_ms=0):
    if not units:return np.zeros(0,dtype=np.float32),[]
    n=max(0,int(SR*xfade_ms/1000));y=units[0].copy();bounds=[]
    for x in units[1:]:
        use=min(n,len(y),len(x))
        if use:
            alpha=np.linspace(0,1,use,endpoint=True,dtype=np.float32)
            y=np.concatenate([y[:-use],y[-use:]*(1-alpha)+x[:use]*alpha,x[use:]])
            b=len(y)-len(x)-use//2
        else:b=len(y)
        bounds.append(int(b));
        if not use:y=np.concatenate([y,x])
    peak=np.max(np.abs(y)) if len(y) else 0
    if peak>.99:y=y*(.99/peak)
    return y.astype(np.float32),bounds

def boundary_metric(y,bounds):
    if not bounds:return {'mean_sample_jump':0.,'mean_energy_jump':0.}
    jumps=[];e=[];w=int(.01*SR)
    for b in bounds:
        b=min(max(1,b),len(y)-1);jumps.append(abs(float(y[b]-y[b-1])))
        l=y[max(0,b-w):b];r=y[b:min(len(y),b+w)];e.append(abs(np.log(rms(l)+1e-8)-np.log(rms(r)+1e-8)))
    return {'mean_sample_jump':float(np.mean(jumps)),'mean_energy_jump':float(np.mean(e))}

def save_wave(path,y):sf.write(path,y,SR,subtype='PCM_16')

def plot_boundary(raw,rawb,best,bestb):
    b0=rawb[0] if rawb else len(raw)//2;b1=bestb[0] if bestb else len(best)//2
    half=int(.08*SR);fig,axs=plt.subplots(2,2,figsize=(11,6))
    for row,(y,b,title) in enumerate([(raw,b0,'Fixed candidate / no cross-fade'),(best,b1,'Unit selection / selected cross-fade')]):
        lo=max(0,b-half);hi=min(len(y),b+half);tt=np.arange(lo,hi)/SR
        axs[row,0].plot(tt,y[lo:hi],lw=.7);axs[row,0].axvline(b/SR,color='red',ls='--');axs[row,0].set(title=title,xlabel='Time (s)',ylabel='Amplitude')
        f,t,z=stft(y[lo:hi],fs=SR,nperseg=512,noverlap=384,boundary=None);db=20*np.log10(np.maximum(np.abs(z),1e-7))
        axs[row,1].pcolormesh(t+lo/SR,f,db,shading='auto',cmap='magma',vmin=-90,vmax=-20);axs[row,1].axvline(b/SR,color='cyan',ls='--');axs[row,1].set(ylim=(0,5000),xlabel='Time (s)',ylabel='Frequency (Hz)')
    fig.tight_layout();fig.savefig(FIG/'boundary_comparison.png',dpi=160);plt.close(fig)

def hmm_state_model(bank):
    rows=[];params={}
    for label,cands in bank.items():
        per=[]
        for cand in cands:
            x=cand['x'];hop=160;frame=512
            rmsf=librosa.feature.rms(y=x,frame_length=frame,hop_length=hop)[0]
            f0t,f0=f0_track(x);f0fill=np.interp(np.arange(len(rmsf))*hop/SR,f0t[np.isfinite(f0)],f0[np.isfinite(f0)]) if np.isfinite(f0).sum()>1 else np.full(len(rmsf),np.nan)
            cent=librosa.feature.spectral_centroid(y=x,sr=SR,n_fft=frame,hop_length=hop)[0]
            m=min(len(rmsf),len(cent));rmsf=rmsf[:m];cent=cent[:m];f0fill=f0fill[:m]
            for s in range(5):
                a=int(s*m/5);b=max(a+1,int((s+1)*m/5)); valid=f0fill[a:b];valid=valid[np.isfinite(valid)]
                per.append({'state':s,'f0_hz':float(np.median(valid)) if len(valid) else np.nan,'rms':float(np.mean(rmsf[a:b])),'spectral_centroid_hz':float(np.mean(cent[a:b])),'duration_s':(b-a)*hop/SR})
        for s in range(5):
            subset=[z for z in per if z['state']==s]
            row={'label':label,'state':s,'f0_hz':float(np.nanmedian([q['f0_hz'] for q in subset])) if np.isfinite([q['f0_hz'] for q in subset]).any() else np.nan,'rms':float(np.mean([q['rms'] for q in subset])),'spectral_centroid_hz':float(np.mean([q['spectral_centroid_hz'] for q in subset])),'duration_s':float(np.mean([q['duration_s'] for q in subset]))}
            rows.append(row)
        params[label]=[r for r in rows if r['label']==label]
    return pd.DataFrame(rows),params

def render_hmm_demo(tokens,selected,params):
    tracks=[];audio=[];t0=0.
    for token,cand in zip(tokens,selected):
        states=params[token]
        target_d=sum(x['duration_s'] for x in states);x=cand['x']
        # HMM state durations define a duration target; waveform source is the real recorded unit.
        stretch=target_d/(len(x)/SR)
        y=librosa.effects.time_stretch(x,rate=1/max(stretch,.1)) if abs(stretch-1)>.02 else x.copy()
        audio.append(y.astype(np.float32))
        elapsed=0.
        for q in states:
            d=q['duration_s']
            tracks.append({'token':token,'state':q['state'],'start_s':t0+elapsed,'mid_s':t0+elapsed+d/2,'duration_s':d,'f0_hz':q['f0_hz'],'rms':q['rms'],'spectral_centroid_hz':q['spectral_centroid_hz']});elapsed+=d
        t0+=len(y)/SR
    trackdf=pd.DataFrame(tracks);trackdf.to_csv(OUT/'hmm_state_trajectory.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(10,5),sharex=True)
    xx=trackdf.mid_s.to_numpy();f=trackdf.f0_hz.to_numpy();axs[0].plot(xx,f,'o-',ms=3);axs[0].set(ylabel='F0 (Hz)',title='HMM-state parameter generation (empirical state means)')
    axs[1].step(xx,trackdf.spectral_centroid_hz,where='mid');axs[1].set(xlabel='Time (s)',ylabel='Spectral centroid (Hz)')
    for _,r in trackdf.drop_duplicates('token').iterrows():
        for ax in axs:ax.axvline(r.start_s,color='gray',lw=.5,alpha=.5)
    fig.tight_layout();fig.savefig(FIG/'hmm_tracks.png',dpi=160);plt.close(fig)
    return audio,trackdf

def main():
    from neural_tts_backend import load_backend,synthesize
    model,tokenizer=load_backend(local_files_only=True)
    neural_units=make_neural_units(model,tokenizer)
    bank,meta=inventory(neural_units)
    case_tokens=normalize_text(QUEUE_CASE)
    missing=sorted(set(case_tokens)-set(bank))
    pd.DataFrame([{'text':QUEUE_CASE,'normalized_tokens':' '.join(case_tokens),'available':not missing,'missing_units':', '.join(missing)}]).to_csv(OUT/'queue_case_inventory_check.csv',index=False)
    sentence_tokens={sid:normalize_text(text) for sid,text in SENTENCES}
    for sid,toks in sentence_tokens.items():
        missing=sorted(set(toks)-set(bank))
        if missing:raise KeyError(f'{sid} has unavailable inventory: {missing}')
    sid='QUEUE';tokens=case_tokens
    fixed=[bank[t][0]['x'] for t in tokens]
    raw,rawb=concatenate(fixed,0);save_wave(AUDIO/'raw_concat.wav',raw);save_wave(AUDIO/'queue_raw_concat.wav',raw)
    selected,total_cost,join_df=select_units(tokens,bank)
    pd.DataFrame([{'position':i,'token':w,'chosen_candidate':q['id'],'target_cost':float(((q['duration']-np.mean([z['duration'] for z in bank[w]]))/(np.std([z['duration'] for z in bank[w]])+.0001))**2),'duration_s':q['duration'],'rms':q['rms']} for i,(w,q) in enumerate(zip(tokens,selected))]).to_csv(OUT/'selected_units.csv',index=False)
    join_df.to_csv(OUT/'join_cost_candidates.csv',index=False)
    pairs=join_df.groupby(['position','left_candidate','right_candidate'],as_index=False).join_cost.min()
    fig,ax=plt.subplots(figsize=(9,3.7));ax.scatter(pairs.position,pairs.join_cost,c=pairs.join_cost,cmap='viridis',s=28);ax.set(xlabel='Token boundary index',ylabel='Join cost (lower is smoother)',title='Candidate boundary costs explored by dynamic programming');fig.tight_layout();fig.savefig(FIG/'unit_selection_costs.png',dpi=160);plt.close(fig)
    smooth_rows=[];smooth_waves={}
    for ms in [0,5,15,30]:
        y,bounds=concatenate([q['x'] for q in selected],ms);save_wave(AUDIO/f'unit_selection_xfade_{ms}ms.wav',y);smooth_waves[ms]=(y,bounds)
        met=boundary_metric(y,bounds);smooth_rows.append({'crossfade_ms':ms,'mean_sample_jump':met['mean_sample_jump'],'mean_energy_jump':met['mean_energy_jump'],'boundary_count':len(bounds),'selection_total_cost':total_cost})
    smooth=pd.DataFrame(smooth_rows);b0=smooth.loc[smooth.crossfade_ms==0].iloc[0]
    smooth['combined_discontinuity_proxy']=smooth.mean_sample_jump/(b0.mean_sample_jump+1e-9)+smooth.mean_energy_jump/(b0.mean_energy_jump+1e-9)
    chosen_ms=int(smooth.sort_values('combined_discontinuity_proxy').iloc[0].crossfade_ms)
    best,bestb=smooth_waves[chosen_ms];save_wave(AUDIO/'best_concat.wav',best);save_wave(AUDIO/'queue_best_concat.wav',best)
    smooth.to_csv(OUT/'crossfade_experiment.csv',index=False)
    fixed_cost=[]
    for i,(a,b) in enumerate(zip(fixed,fixed[1:])):
        jc,sc,ec=join_cost(a,b);fixed_cost.append({'position':i+1,'left_candidate':str(i),'right_candidate':str(i+1),'join_cost':jc,'spectral_cost':sc,'energy_cost':ec})
    selected_cost=[]
    for i,(a,b) in enumerate(zip(selected,selected[1:])):
        jc,sc,ec=join_cost(a['x'],b['x']);selected_cost.append({'position':i+1,'left_candidate':a['id'],'right_candidate':b['id'],'join_cost':jc,'spectral_cost':sc,'energy_cost':ec})
    queue_fixed_join=sum(r['join_cost'] for r in fixed_cost);queue_dp_join=sum(r['join_cost'] for r in selected_cost)
    command_tokens=sentence_tokens['S1'];command_fixed=[bank[t][0]['x'] for t in command_tokens];command_selected,command_dp_total,_=select_units(command_tokens,bank)
    command_fixed_join=sum(join_cost(a,b)[0] for a,b in zip(command_fixed,command_fixed[1:]));command_dp_join=sum(join_cost(a['x'],b['x'])[0] for a,b in zip(command_selected,command_selected[1:]))
    costdf=pd.DataFrame([{'utterance':'QUEUE','method':'fixed first candidate','total_join_cost':queue_fixed_join},{'utterance':'QUEUE','method':'DP unit selection','total_join_cost':queue_dp_join},{'utterance':'S1 command','method':'fixed first candidate','total_join_cost':command_fixed_join},{'utterance':'S1 command','method':'DP unit selection','total_join_cost':command_dp_join}]);costdf.to_csv(OUT/'unit_selection_summary.csv',index=False)
    with open(OUT/'unit_selection_cost_detail.json','w',encoding='utf8') as f:json.dump({'fixed':fixed_cost,'selected':selected_cost,'dp_objective_including_target':total_cost},f,ensure_ascii=False,indent=2)
    plot_boundary(raw,rawb,best,bestb)
    (DATA/'hmm_demo').mkdir(exist_ok=True)
    hmm_table,params=hmm_state_model(bank);hmm_table.to_csv(DATA/'hmm_demo'/'state_parameters.csv',index=False)
    hmm_audio,tracks=render_hmm_demo(tokens,selected,params);hmm_wave,_=concatenate(hmm_audio,chosen_ms);save_wave(AUDIO/'hmm_tts.wav',hmm_wave)
    # Keep a second state-controlled command example for the original LAB2 vocabulary.
    command_tokens=sentence_tokens['S1'];command_selected,_,_=select_units(command_tokens,bank)
    command_hmm_audio,_=render_hmm_demo(command_tokens,command_selected,params);command_hmm,_=concatenate(command_hmm_audio,chosen_ms);save_wave(AUDIO/'hmm_command_demo.wav',command_hmm)
    voiced=bank['lên'][0]['x'];ft,fc=f0_track(voiced);finite=fc[np.isfinite(fc)];median_f0=float(np.median(finite)) if len(finite) else float('nan')
    pitch_period=int(round(SR/median_f0)) if np.isfinite(median_f0) and median_f0>0 else None
    pitch_data=[];base=voiced
    for speed in [.8,1.,1.2]:
        y=librosa.effects.time_stretch(base,rate=speed);save_wave(AUDIO/f'prosody_speed_{speed:.1f}x.wav',y);pitch_data.append({'experiment':'speed','setting':speed,'output_duration_s':len(y)/SR,'median_f0_hz':median_f0})
    for semitones in [-2,0,2]:
        y=librosa.effects.pitch_shift(base,sr=SR,n_steps=semitones);save_wave(AUDIO/f'prosody_pitch_{semitones:+d}st.wav',y);pitch_data.append({'experiment':'pitch_semitones','setting':semitones,'output_duration_s':len(y)/SR,'median_f0_hz':median_f0*2**(semitones/12) if np.isfinite(median_f0) else np.nan})
    pd.DataFrame(pitch_data).to_csv(OUT/'prosody_experiment.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(9,5));axs[0].plot(ft,fc);axs[0].set(ylim=(60,450),ylabel='F0 (Hz)',title=f"Voiced unit 'lên': median F0={median_f0:.1f} Hz, period≈{pitch_period} samples")
    for semitones in [-2,0,2]:
        y=librosa.effects.pitch_shift(base,sr=SR,n_steps=semitones);s=librosa.amplitude_to_db(np.abs(librosa.stft(y,n_fft=512,hop_length=160)),ref=np.max);axs[1].plot(np.arange(s.shape[1])*160/SR,np.mean(s,axis=0),label=f'{semitones:+d} semitones')
    axs[1].set(xlabel='Time (s)',ylabel='Mean relative STFT level (dB)');axs[1].legend();fig.tight_layout();fig.savefig(FIG/'prosody_f0_pitch.png',dpi=160);plt.close(fig)
    sentence_rows=[]
    for sid0,text in SENTENCES:
        start=time.perf_counter();toks=normalize_text(text);sel,tc,_=select_units(toks,bank);y,_=concatenate([q['x'] for q in sel],chosen_ms);latency=time.perf_counter()-start
        save_wave(AUDIO/f'{sid0}_best_concat.wav',y);sentence_rows.append({'id':sid0,'text':text,'tokens':' '.join(toks),'file':f'{sid0}_best_concat.wav','duration_s':len(y)/SR,'generation_latency_s':latency,'selection_cost':tc,'all_tokens_in_inventory':True})
    queue_start=time.perf_counter();queue_best,queue_tc,_=select_units(case_tokens,bank);queue_wave,_=concatenate([q['x'] for q in queue_best],chosen_ms);queue_concat_latency=time.perf_counter()-queue_start
    sentence_rows.append({'id':'QUEUE','text':QUEUE_CASE,'tokens':' '.join(case_tokens),'file':'queue_best_concat.wav','duration_s':len(queue_wave)/SR,'generation_latency_s':queue_concat_latency,'selection_cost':queue_tc,'all_tokens_in_inventory':True})
    neural_outputs={};neural_latency={}
    for sid0,text in SENTENCES+[('QUEUE',QUEUE_CASE)]:
        y,sr,elapsed=synthesize(text,model,tokenizer,seed=6000+len(sid0));
        if sr!=SR:y=librosa.resample(y,orig_sr=sr,target_sr=SR)
        outfile='neural_tts.wav' if sid0=='QUEUE' else f'{sid0}_neural_tts.wav';save_wave(AUDIO/outfile,y);neural_outputs[sid0]=y;neural_latency[sid0]=elapsed
        for row in sentence_rows:
            if row['id']==sid0:row['neural_file']=outfile;row['neural_inference_latency_s']=elapsed
    queue_neural=neural_outputs['QUEUE']
    mel=librosa.feature.melspectrogram(y=queue_neural,sr=SR,n_fft=1024,hop_length=256,n_mels=80);mdb=librosa.power_to_db(mel,ref=np.max)
    fig,ax=plt.subplots(figsize=(9,3.5));ax.imshow(mdb,origin='lower',aspect='auto',extent=[0,len(queue_neural)/SR,0,SR/2],cmap='magma',vmin=-80,vmax=0);ax.set(ylim=(0,8000),xlabel='Time (s)',ylabel='Frequency (Hz)',title='MMS VITS queue-sentence mel spectrogram');fig.tight_layout();fig.savefig(FIG/'neural_queue_mel.png',dpi=160);plt.close(fig)
    pd.DataFrame(sentence_rows).to_csv(OUT/'test_sentences.csv',index=False)
    comparison=[('Raw concat',raw),('Unit selection + cross-fade',best),('HMM state hybrid',hmm_wave),('Neural MMS VITS',queue_neural)]
    fig,axs=plt.subplots(4,2,figsize=(10,9))
    for r,(name,y) in enumerate(comparison):
        tt=np.arange(len(y))/SR;axs[r,0].plot(tt,y,lw=.45);axs[r,0].set(title=name+' waveform',xlabel='seconds',ylabel='amplitude')
        _,tt0,z=stft(y,fs=SR,nperseg=512,noverlap=384);axs[r,1].pcolormesh(tt0,np.fft.rfftfreq(512,1/SR),20*np.log10(np.maximum(abs(z),1e-7)),shading='auto',cmap='magma',vmin=-90,vmax=-20);axs[r,1].set(title=name+' spectrogram',ylim=(0,5000),xlabel='seconds',ylabel='Hz')
    fig.tight_layout();fig.savefig(FIG/'backend_comparison.png',dpi=150);plt.close(fig)
    with open(OUT/'ratings.csv','w',newline='',encoding='utf-8-sig') as f:
        wr=csv.writer(f);wr.writerow(['participant_id','utterance_id','method','intelligibility_1to5','naturalness_1to5','continuity_1to5','comments'])
        for participant in range(1,6):
            for sid0 in ['QUEUE']+[s[0] for s in SENTENCES]:
                for method in ['raw_concat','best_concat','hmm_state_hybrid','neural_vits']:
                    wr.writerow([participant,sid0,method,'','','',''])
    metrics={'sample_rate':SR,'inventory_label_count':len(bank),'candidate_recording_count':int(len(meta)),'candidates_per_label':{k:len(v) for k,v in bank.items()},'queue_case':QUEUE_CASE,'queue_case_tokens':case_tokens,'queue_case_supported':not missing,'queue_case_missing_units':missing,'demo_sentence':QUEUE_CASE,'demo_tokens':case_tokens,'crossfade_chosen_ms':chosen_ms,'crossfade_metrics':smooth.to_dict(orient='records'),'queue_fixed_join_cost':queue_fixed_join,'queue_selected_join_cost':queue_dp_join,'queue_join_reduction_percent':100*(queue_fixed_join-queue_dp_join)/(queue_fixed_join+1e-9),'command_fixed_join_cost':command_fixed_join,'command_selected_join_cost':command_dp_join,'command_join_reduction_percent':100*(command_fixed_join-command_dp_join)/(command_fixed_join+1e-9),'median_f0_hz':median_f0,'pitch_period_samples':pitch_period,'hmm_backend':'Five empirical state parameter means per word with state-duration-controlled recorded-unit resynthesis; HMM parameter demo, not a separately trained neural or full vocoder voice','neural_tts_model':'facebook/mms-tts-vie (Meta MMS VITS; local-only inference; CC-BY-NC 4.0)','neural_tts_available':True,'neural_latency_s':neural_latency,'concat_latency_s':{r['id']:r['generation_latency_s'] for r in sentence_rows},'subjective_ratings_collected':False,'intelligibility_proxy':'All text tokens map to inventory; listener ratings remain to be collected.'}
    with open(OUT/'metrics.json','w',encoding='utf-8') as f:json.dump(metrics,f,ensure_ascii=False,indent=2)
    with open(DATA/'neural_tts_model.json','w',encoding='utf-8') as f:json.dump({'model_id':'facebook/mms-tts-vie','license':'CC-BY-NC 4.0','weights_in_repo':False,'offline_after_preparation':True},f,ensure_ascii=False,indent=2)
    print(json.dumps({k:metrics[k] for k in ['inventory_label_count','candidate_recording_count','queue_case_supported','crossfade_chosen_ms','queue_fixed_join_cost','queue_selected_join_cost','command_join_reduction_percent','median_f0_hz','neural_tts_model','neural_latency_s']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
