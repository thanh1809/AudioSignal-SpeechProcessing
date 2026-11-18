"""Reproducible end-to-end Lab 5 pipeline. Run: python lab5.py"""
from __future__ import annotations
import csv, json, math, warnings
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import FastICA
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / 'lab5_data', ROOT / 'results'
AUDIO, FIG, TAB = OUT / 'audio', OUT / 'figures', OUT / 'tables'
SR, NFFT, WIN, HOP = 16000, 512, 400, 160
SPEAKERS = ['spk01', 'spk02', 'spk03']
SEED = 457
np.random.seed(SEED)
for d in (AUDIO, FIG, TAB): d.mkdir(parents=True, exist_ok=True)

def write_csv(path, rows, fields=None):
    if not rows: return
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def read_audio(path):
    x, sr = sf.read(path, dtype='float64', always_2d=False)
    if x.ndim == 2: x = x.mean(axis=1)
    if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    x=np.asarray(x, dtype=np.float64)
    return x / max(np.max(np.abs(x)), 1e-9) * 0.8

def save_audio(path, x):
    x=np.asarray(x, dtype=np.float64)
    sf.write(path, np.clip(x, -0.999, 0.999), SR, subtype='PCM_16')

def power(x): return float(np.mean(np.asarray(x, dtype=float)**2)+1e-15)
def snr_db(s, e):
    n=min(len(s),len(e)); s=np.asarray(s[:n]); e=np.asarray(e[:n])
    return float(10*np.log10((np.sum(s*s)+1e-15)/(np.sum((s-e)**2)+1e-15)))

def make_noise(s, v, target):
    if len(v)<len(s): v=np.tile(v, int(np.ceil(len(s)/len(v))))
    v=v[:len(s)]; alpha=np.sqrt(power(s)/(power(v)*10**(target/10)))
    return s+alpha*v, float(alpha)

def stft(x):
    return librosa.stft(x, n_fft=NFFT, hop_length=HOP, win_length=WIN, window='hann', center=True)

def enhance(y, noise, method='ss', beta=1.0, floor=0.02):
    Y=stft(y); N=stft(noise); Pn=np.mean(np.abs(N)**2, axis=1, keepdims=True)
    Py=np.abs(Y)**2
    if method=='ss':
        P=np.maximum(Py-beta*Pn, floor*Pn)
        Z=np.sqrt(P)*np.exp(1j*np.angle(Y))
    else:
        Ps=np.maximum(Py-Pn,0.0); H=Ps/(Ps+Pn+1e-12); Z=H*Y
    return librosa.istft(Z, hop_length=HOP, win_length=WIN, window='hann', length=len(y))

def embedding(x, use_delta=True, cmn=False):
    M=librosa.feature.mfcc(y=x, sr=SR, n_mfcc=20, n_fft=NFFT,
        win_length=WIN, hop_length=HOP, window='hann', center=True)
    if cmn: M=M-M.mean(axis=1, keepdims=True)
    feats=[M.mean(axis=1), M.std(axis=1)]
    if use_delta:
        D=librosa.feature.delta(M, width=9, mode='nearest')
        feats.extend([D.mean(axis=1),D.std(axis=1)])
    e=np.concatenate(feats); return e/(np.linalg.norm(e)+1e-12)

def cosine(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def model_bank(use_delta=True, cmn=False):
    return {spk:np.mean([embedding(read_audio(DATA/'clean'/f'{spk}_enroll_{i:02d}.wav'),use_delta,cmn) for i in range(1,5)],axis=0) for spk in SPEAKERS}

def score_table(x, models): return {s:cosine(embedding(x),m) for s,m in models.items()}

def plot_spec(signals, path, title):
    ncols=2 if len(signals)>4 else 1; nrows=int(np.ceil(len(signals)/ncols))
    fig,axs=plt.subplots(nrows,ncols,figsize=(9,2.4*nrows), constrained_layout=True)
    axs=np.atleast_1d(axs).ravel()
    for ax,(name,x) in zip(axs,signals.items()):
        D=librosa.amplitude_to_db(np.abs(stft(x)),ref=np.max)
        ax.imshow(D,origin='lower',aspect='auto',extent=[0,len(x)/SR,0,SR/2],cmap='magma',vmin=-75,vmax=0)
        ax.set(title=name,xlabel='Time (s)',ylabel='Frequency (Hz)')
    for ax in axs[len(signals):]: ax.set_visible(False)
    fig.suptitle(title); fig.savefig(path,dpi=160); plt.close(fig)

def ensure_data():
    files=list((DATA/'clean').glob('spk01_enroll_*.wav'))+list((DATA/'clean').glob('spk02_enroll_*.wav'))
    if len(files)<8: raise FileNotFoundError('Need 4 enrollment WAV files for each of two source voices; see README.md.')
    # Speaker 3 is a deterministic pitch-shifted derived voice based on speaker 2.
    for split,n in [('enroll',4),('dev',2),('test',2)]:
        for i in range(1,n+1):
            p=DATA/'clean'/f'spk03_{split}_{i:02d}.wav'
            if not p.exists():
                x=read_audio(DATA/'clean'/f'spk02_{split}_{i:02d}.wav')
                shifted=librosa.effects.pitch_shift(x,sr=SR,n_steps=-4.0)
                save_audio(p,shifted)
    if not (DATA/'noise'/'fan.wav').exists():
        rng=np.random.default_rng(SEED); t=np.arange(SR*6)/SR
        fan=.35*np.sin(2*np.pi*90*t)+.16*np.sin(2*np.pi*180*t)+.08*np.sin(2*np.pi*270*t)+.08*rng.normal(size=len(t))
        save_audio(DATA/'noise'/'fan.wav',fan/(np.max(np.abs(fan))+1e-9)*.6)
    if not (DATA/'noise'/'cafeteria.wav').exists():
        rng=np.random.default_rng(SEED+1); w=rng.normal(size=SR*6)
        sos=butter(4,[250,4500],btype='bandpass',fs=SR,output='sos'); w=sosfiltfilt(sos,w)
        # Slowly varying speech-like babble texture without intelligible borrowed speech.
        env=.55+.4*np.sin(2*np.pi*.7*np.arange(len(w))/SR)**2
        save_audio(DATA/'noise'/'cafeteria.wav',w*env/(np.max(np.abs(w*env))+1e-9)*.55)
    if not (DATA/'separation'/'source_A.wav').exists():
        save_audio(DATA/'separation'/'source_A.wav',read_audio(DATA/'clean'/'spk01_test_01.wav'))
    if not (DATA/'separation'/'source_B.wav').exists():
        save_audio(DATA/'separation'/'source_B.wav',read_audio(DATA/'clean'/'spk02_test_01.wav'))
    manifest=[]
    for p in sorted((DATA/'clean').glob('spk*_*.wav')):
        x=read_audio(p); spk=p.name.split('_')[0]
        provenance={'spk01':'Microsoft SAPI David TTS','spk02':'Microsoft SAPI Zira TTS','spk03':'pitch-shifted derivative of spk02'}[spk]
        manifest.append({'file':p.name,'speaker_class':spk,'split':p.stem.split('_')[1],'sample_rate_hz':SR,'duration_s':round(len(x)/SR,4),'provenance':provenance})
    write_csv(DATA/'metadata.csv',manifest)

def run_enhancement():
    clean=read_audio(DATA/'clean'/'spk01_test_01.wav'); noise=read_audio(DATA/'noise'/'fan.wav')
    rows=[]; chosen={}
    for r in (0,5,10):
        y,a=make_noise(clean,noise,r); measured=10*np.log10(power(clean)/power(y-clean)); noise_scaled=noise*a
        save_audio(AUDIO/f'noisy_{r}dB.wav',y)
        ss=enhance(y,noise_scaled,'ss',beta=1.0,floor=.02); wi=enhance(y,noise_scaled,'wiener')
        save_audio(AUDIO/f'enhanced_ss_{r}dB.wav',ss); save_audio(AUDIO/f'enhanced_wiener_{r}dB.wav',wi)
        for name,z in [('noisy',y),('spectral_subtraction',ss),('wiener',wi)]:
            rows.append({'target_snr_db':r,'method':name,'measured_snr_in_db':round(float(measured),4),'snr_out_db':round(snr_db(clean,z),4),'snr_improvement_db':round(snr_db(clean,z)-measured,4),'noise_scale_alpha':round(a,7)})
        if r==5: chosen={'clean':clean,'noisy':y,'ss':ss,'wiener':wi}
    write_csv(TAB/'enhancement_snr.csv',rows)
    plot_spec({'Clean':chosen['clean'],'Noisy 5 dB':chosen['noisy'],'Spectral subtraction':chosen['ss'],'Wiener':chosen['wiener']},FIG/'enhancement_spectrograms.png','Speech enhancement at 5 dB input SNR')
    # Parameter sweep required by the handout.
    sweep=[]
    for r in (0,5,10):
        y,a=make_noise(clean,noise,r); noise_scaled=noise*a
        for b in (1.0,1.5,2.0):
            for g in (0.0,.01,.05):
                z=enhance(y,noise_scaled,'ss',b,g)
                sweep.append({'snr_in_db':r,'beta':b,'floor_gamma':g,'snr_out_db':round(snr_db(clean,z),4),'delta_snr_db':round(snr_db(clean,z)-r,4)})
    write_csv(TAB/'ss_parameter_sweep.csv',sweep)
    # Broadband cafeteria noise at 5 dB as a second noise condition.
    caf=read_audio(DATA/'noise'/'cafeteria.wav'); y,alpha=make_noise(clean,caf,5)
    save_audio(AUDIO/'noisy_cafeteria_5dB.wav',y)
    save_audio(AUDIO/'cafeteria_wiener_5dB.wav',enhance(y,caf*alpha,'wiener'))
    return chosen

def run_speaker_id(chosen):
    models=model_bank(True,False); rows=[]; cm_rows=[]; truths=[]; preds=[]
    conditions=['clean','noisy_5dB','spectral_subtraction_5dB','wiener_5dB']
    for spk in SPEAKERS:
        for idx in (1,2):
            clean=read_audio(DATA/'clean'/f'{spk}_test_{idx:02d}.wav')
            noise=read_audio(DATA/'noise'/'fan.wav'); y,alpha=make_noise(clean,noise,5); noise_scaled=noise*alpha
            ss=enhance(y,noise_scaled,'ss',1,.02); wi=enhance(y,noise_scaled,'wiener')
            xs={'clean':clean,'noisy_5dB':y,'spectral_subtraction_5dB':ss,'wiener_5dB':wi}
            for cond,x in xs.items():
                scores=score_table(x,models); pred=max(scores,key=scores.get)
                rows.append({'speaker_true':spk,'utterance':idx,'condition':cond,'predicted':pred,'correct':int(pred==spk),'top_score':round(scores[pred],5),'true_score':round(scores[spk],5),'margin':round(scores[spk]-max(v for k,v in scores.items() if k!=spk),5),**{f'score_{k}':round(v,5) for k,v in scores.items()}})
                if cond=='clean': truths.append(spk);preds.append(pred)
    write_csv(TAB/'speaker_id_scores.csv',rows)
    mat=confusion_matrix(truths,preds,labels=SPEAKERS)
    for i,s in enumerate(SPEAKERS): cm_rows.append({'true':s,**{f'pred_{t}':int(mat[i,j]) for j,t in enumerate(SPEAKERS)}})
    write_csv(TAB/'confusion_matrix.csv',cm_rows)
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(mat,cmap='Blues'); ax.set_xticks(range(3),SPEAKERS);ax.set_yticks(range(3),SPEAKERS);ax.set(xlabel='Predicted speaker',ylabel='True speaker',title='Clean test confusion matrix')
    for i in range(3):
        for j in range(3): ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='black')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIG/'confusion_matrix.png',dpi=160);plt.close(fig)
    ab=[]
    for delta,cmn in [(False,False),(True,False),(True,True)]:
        bank=model_bank(delta,cmn); ok=0; total=0
        for spk in SPEAKERS:
            for i in (1,2):
                x=read_audio(DATA/'clean'/f'{spk}_test_{i:02d}.wav'); e=embedding(x,delta,cmn)
                pred=max(bank,key=lambda q:cosine(e,bank[q]));ok+=pred==spk;total+=1
        ab.append({'features':'MFCC+delta' if delta else 'MFCC-only','cmn':cmn,'clean_test_accuracy':round(ok/total,4)})
    write_csv(TAB/'speaker_feature_ablation.csv',ab)
    enrollment=[]
    for count in (1,2,4):
        bank={spk:np.mean([embedding(read_audio(DATA/'clean'/f'{spk}_enroll_{i:02d}.wav')) for i in range(1,count+1)],axis=0) for spk in SPEAKERS}
        correct=0;total=0
        for spk in SPEAKERS:
            for i in (1,2):
                e=embedding(read_audio(DATA/'clean'/f'{spk}_test_{i:02d}.wav'))
                correct+=max(bank,key=lambda q:cosine(e,bank[q]))==spk;total+=1
        enrollment.append({'enrollment_utterances_per_speaker':count,'clean_test_accuracy':round(correct/total,4),'n_test':total})
    write_csv(TAB/'enrollment_count_ablation.csv',enrollment)
    cond_acc=[]
    for cond in conditions:
        subset=[r for r in rows if r['condition']==cond]
        cond_acc.append({'condition':cond,'accuracy':round(np.mean([r['correct'] for r in subset]),4),'n':len(subset)})
    write_csv(TAB/'speaker_condition_accuracy.csv',cond_acc)
    return rows, cond_acc

def run_verification():
    models=model_bank(True,False); dev=[]; test=[]
    for split,target in [('dev',dev),('test',test)]:
        for spk in SPEAKERS:
            for i in (1,2):
                e=embedding(read_audio(DATA/'clean'/f'{spk}_{split}_{i:02d}.wav'))
                for claim in SPEAKERS:
                    target.append({'claimed':claim,'actual':spk,'genuine':int(claim==spk),'score':cosine(e,models[claim])})
    thresholds=np.linspace(-1,1,401); curve=[]
    for th in thresholds:
        gen=[r['score'] for r in dev if r['genuine']]; imp=[r['score'] for r in dev if not r['genuine']]
        far=np.mean(np.array(imp)>=th); frr=np.mean(np.array(gen)<th)
        curve.append({'threshold':float(th),'far':float(far),'frr':float(frr),'gap':float(abs(far-frr))})
    best=min(curve,key=lambda r:(r['gap'],abs(r['threshold'])))
    th=best['threshold']; gen=[r for r in test if r['genuine']];imp=[r for r in test if not r['genuine']]
    far=float(np.mean([r['score']>=th for r in imp])); frr=float(np.mean([r['score']<th for r in gen]))
    write_csv(TAB/'verification_threshold_curve.csv',curve)
    write_csv(TAB/'verification_test_trials.csv',[{**r,'accepted':int(r['score']>=th)} for r in test])
    with (TAB/'verification_summary.json').open('w',encoding='utf-8') as f: json.dump({'threshold_selected_on':'dev','threshold':th,'dev_far':best['far'],'dev_frr':best['frr'],'test_far':far,'test_frr':frr,'n_test_genuine':len(gen),'n_test_impostor':len(imp)},f,indent=2)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(thresholds,[r['far'] for r in curve],label='FAR');ax.plot(thresholds,[r['frr'] for r in curve],label='FRR');ax.axvline(th,color='black',ls='--',label=f'Dev threshold={th:.2f}');ax.set(xlabel='Cosine threshold',ylabel='Error rate',title='Verification FAR and FRR');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(FIG/'far_frr.png',dpi=160);plt.close(fig)

def corr_matrix(S,E): return np.array([[np.corrcoef(S[i],E[j])[0,1] for j in range(2)] for i in range(2)])
def separate(S,A,noise_std=0):
    X=A@S
    if noise_std:
        rng=np.random.default_rng(SEED+int(noise_std*1e6));X=X+rng.normal(0,noise_std,size=X.shape)
    ica=FastICA(n_components=2,whiten='unit-variance',random_state=SEED,max_iter=2000,tol=1e-4)
    E=ica.fit_transform(X.T).T
    C=np.abs(corr_matrix(S,E)); rows,cols=linear_sum_assignment(-C)
    aligned=np.zeros_like(E); aligned_abs=[]
    for i,j in zip(rows,cols):
        sign=1 if np.corrcoef(S[i],E[j])[0,1]>=0 else -1
        aligned[i]=E[j]*sign; aligned_abs.append(abs(np.corrcoef(S[i],E[j])[0,1]))
    return X,E,C,aligned,aligned_abs

def run_separation():
    a=read_audio(DATA/'separation'/'source_A.wav');b=read_audio(DATA/'separation'/'source_B.wav');L=min(len(a),len(b));
    S=np.vstack([a[:L]/np.sqrt(power(a[:L])),b[:L]/np.sqrt(power(b[:L]))])
    settings=[('default',np.array([[1.,.6],[.4,1.]]),0.),('alternate',np.array([[1.,.35],[.7,1.]]),0.),('near_singular',np.array([[1.,.9],[.9,1.]]),0.),('noisy',np.array([[1.,.6],[.4,1.]]),.03)]
    summary=[]
    for name,A,ns in settings:
        X,E,C,aligned,ac=separate(S,A,ns)
        summary.append({'condition':name,'mixing_matrix':json.dumps(A.tolist()),'determinant':round(float(np.linalg.det(A)),5),'mixture_noise_std':ns,'corr_s1_e1':round(float(C[0,0]),5),'corr_s1_e2':round(float(C[0,1]),5),'corr_s2_e1':round(float(C[1,0]),5),'corr_s2_e2':round(float(C[1,1]),5),'matched_corr_s1':round(float(ac[0]),5),'matched_corr_s2':round(float(ac[1]),5),'mean_matched_abs_corr':round(float(np.mean(ac)),5)})
        if name=='default':
            for i in range(2):
                save_audio(AUDIO/f'mix_mic{i+1}.wav',X[i]/max(np.max(np.abs(X[i])),1e-9)*.9)
                save_audio(AUDIO/f'separated_{i+1}.wav',aligned[i]/max(np.max(np.abs(aligned[i])),1e-9)*.9)
            fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(C,vmin=0,vmax=1,cmap='viridis');ax.set_xticks([0,1],['Estimate 1','Estimate 2']);ax.set_yticks([0,1],['Source A','Source B']);ax.set_title('Absolute correlation before assignment')
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{C[i,j]:.3f}',ha='center',va='center',color='white')
            fig.colorbar(im,ax=ax,label='|Pearson correlation|');fig.tight_layout();fig.savefig(FIG/'ica_correlation.png',dpi=160);plt.close(fig)
            plot_spec({'Source A':S[0],'Mic 1 mixture':X[0],'Separated 1 (matched A)':aligned[0],'Source B':S[1],'Mic 2 mixture':X[1],'Separated 2 (matched B)':aligned[1]},FIG/'ica_waveforms_spectrograms.png','Two-source instantaneous mixture and FastICA output')
    write_csv(TAB/'ica_conditions.csv',summary)

def main():
    ensure_data(); chosen=run_enhancement(); _,acc=run_speaker_id(chosen);run_verification();run_separation()
    meta={'sample_rate_hz':SR,'n_fft':NFFT,'window_samples':WIN,'hop_samples':HOP,'seed':SEED,'speakers':SPEAKERS,'enrollment_utterances_per_speaker':4,'dev_utterances_per_speaker':2,'test_utterances_per_speaker':2,'input_snr_db':[0,5,10],'noise_types':['synthetic fan-like','synthetic filtered cafeteria-like'],'synthetic_voice_notice':'spk01 and spk02 are Microsoft SAPI TTS voices; spk03 is a deterministic pitch-shifted derivative of spk02. Results are for a reproducible teaching demonstration, not independent human-speaker validation.'}
    with (TAB/'run_config.json').open('w',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
    print('Lab 5 complete. Results:',OUT)
    print('Input noise targets:',(TAB/'enhancement_snr.csv').read_text(encoding='utf-8-sig').splitlines()[:4])
    print('Speaker condition accuracy:',acc)
    print('Verification:',(TAB/'verification_summary.json').read_text(encoding='utf-8'))
    print('ICA:',(TAB/'ica_conditions.csv').read_text(encoding='utf-8-sig').splitlines()[:3])

if __name__=='__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
