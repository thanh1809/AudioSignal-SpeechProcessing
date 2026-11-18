"""End-to-end Lab 3: dynamic MFCC, diagonal left-to-right Gaussian HMM, DTW, LM."""
from __future__ import annotations
import json, math, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score, confusion_matrix

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'LAB2' / 'Lab2_data_figures' / 'dataset'
OUT = HERE / 'results'
FIG = OUT / 'figures'
MODELS_DIR = HERE / 'models'
LABELS = ['bat', 'dung', 'len', 'xuong', 'trai']
SR = 16000
TRAIN_IDS = [1, 2, 3]
TEST_IDS = [4, 5]
SEED = 7

for p in (OUT, FIG, MODELS_DIR): p.mkdir(parents=True, exist_ok=True)

def wav_path(label, i): return DATA / label / f'{label}_{i:02d}.wav'

def features(path, dynamic=True):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < 512: y = np.pad(y, (0, 512-len(y)))
    y = librosa.effects.preemphasis(y, coef=0.97)
    c = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512,
        win_length=400, hop_length=160, window='hamming', n_mels=26,
        fmin=20, fmax=sr/2).astype(np.float64)
    if dynamic:
        d1 = librosa.feature.delta(c, order=1, mode='nearest')
        d2 = librosa.feature.delta(c, order=2, mode='nearest')
        f = np.vstack((c, d1, d2)).T
    else: f = c.T
    # Per-utterance CMVN, applied independently to each recording.
    f = (f - f.mean(axis=0, keepdims=True)) / (f.std(axis=0, keepdims=True) + 1e-8)
    return f.astype(np.float64), c.T

def log_gaussian_diag(x, means, variances):
    diff = x[:, None, :] - means[None, :, :]
    return -0.5 * (x.shape[1]*np.log(2*np.pi) + np.log(variances).sum(axis=1)[None,:]
                   + (diff*diff/variances[None,:,:]).sum(axis=2))

class LeftRightGaussianHMM:
    """Small educational Bakis HMM with diagonal Gaussian emissions and EM."""
    def __init__(self, n_states=5, n_iter=18, min_covar=1e-2):
        self.n_states, self.n_iter, self.min_covar = n_states, n_iter, min_covar
        self.startprob = np.eye(1, n_states, 0).ravel()
        self.transmat = np.zeros((n_states, n_states))
        for i in range(n_states-1): self.transmat[i, i:i+2] = [0.7, 0.3]
        self.transmat[-1, -1] = 1.
        self.logA = np.where(self.transmat > 0, np.log(np.maximum(self.transmat, 1e-300)), -np.inf)
    def _initialize(self, sequences):
        allx = np.vstack(sequences); d = allx.shape[1]
        means=[]
        for s in range(self.n_states):
            chunks=[]
            for x in sequences:
                lo=int(s*len(x)/self.n_states); hi=max(lo+1,int((s+1)*len(x)/self.n_states))
                chunks.append(x[lo:min(hi,len(x))])
            means.append(np.vstack(chunks).mean(axis=0))
        self.means=np.asarray(means)
        base=np.maximum(allx.var(axis=0), self.min_covar)
        self.vars=np.tile(base, (self.n_states,1))
    def _fb(self, x):
        le=log_gaussian_diag(x,self.means,self.vars); T=len(x); S=self.n_states
        la=np.full((T,S),-np.inf); la[0]=np.log(np.maximum(self.startprob,1e-300))+le[0]
        for t in range(1,T): la[t]=le[t]+logsumexp(la[t-1][:,None]+self.logA,axis=0)
        lb=np.zeros((T,S))
        for t in range(T-2,-1,-1): lb[t]=logsumexp(self.logA+le[t+1][None,:]+lb[t+1][None,:],axis=1)
        ll=logsumexp(la[-1]); lg=la+lb-ll
        gamma=np.exp(np.clip(lg,-745,0))
        xi=np.zeros((S,S))
        for t in range(T-1):
            lx=la[t][:,None]+self.logA+le[t+1][None,:]+lb[t+1][None,:]-ll
            xi += np.exp(np.clip(lx,-745,0))
        return ll,gamma,xi
    def fit(self, sequences):
        self._initialize(sequences); self.history=[]
        for _ in range(self.n_iter):
            occ=np.zeros(self.n_states); sx=np.zeros_like(self.means); sx2=np.zeros_like(self.means); trans=np.zeros_like(self.transmat); ll=0.
            for x in sequences:
                val,gamma,xi=self._fb(x); ll+=val
                occ+=gamma.sum(axis=0); sx+=gamma.T@x; sx2+=gamma.T@(x*x); trans+=xi
            self.history.append(float(ll))
            self.means=sx/np.maximum(occ[:,None],1e-8)
            self.vars=np.maximum(sx2/np.maximum(occ[:,None],1e-8)-self.means**2,self.min_covar)
            for i in range(self.n_states):
                allowed=np.flatnonzero(self.transmat[i]>0)
                row=trans[i,allowed]+1e-6
                self.transmat[i,:]=0.; self.transmat[i,allowed]=row/row.sum()
            self.logA=np.where(self.transmat>0,np.log(np.maximum(self.transmat,1e-300)),-np.inf)
            if len(self.history)>1 and abs(self.history[-1]-self.history[-2])<1e-3: break
        return self
    def score(self,x): return float(self._fb(x)[0])
    def decode(self,x):
        le=log_gaussian_diag(x,self.means,self.vars); T=len(x); S=self.n_states
        dp=np.full((T,S),-np.inf); ptr=np.zeros((T,S),int)
        dp[0]=np.log(np.maximum(self.startprob,1e-300))+le[0]
        for t in range(1,T):
            z=dp[t-1][:,None]+self.logA; ptr[t]=np.argmax(z,axis=0); dp[t]=le[t]+np.max(z,axis=0)
        state=int(np.argmax(dp[-1])); path=np.zeros(T,int); path[-1]=state
        for t in range(T-1,0,-1): path[t-1]=ptr[t,path[t]]
        return float(np.max(dp[-1])),path

def dtw_distance(a,b):
    # Euclidean MFCC path with standard insertion/deletion/diagonal moves.
    d=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2); n,m=d.shape
    acc=np.full((n+1,m+1),np.inf); acc[0,0]=0.
    for i in range(1,n+1):
        for j in range(1,m+1): acc[i,j]=d[i-1,j-1]+min(acc[i-1,j],acc[i,j-1],acc[i-1,j-1])
    return float(acc[n,m]/(n+m))

def parse_lexicon(path):
    """Parse WORD PHONE_1 ... PHONE_K lines into a pronunciation dictionary."""
    lex={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        fields=line.split()
        if len(fields)<2: raise ValueError(f'Lexicon line needs at least one phone: {line}')
        lex[fields[0]]=fields[1:]
    return lex

def bigram_prob(prev, cur, counts2, counts1, vocab, k=0.5):
    return (counts2.get((prev,cur),0)+k)/(counts1.get(prev,0)+k*vocab)

def plot_cm(cm, labels, path, title):
    fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,ax=ax)
    ax.set(xlabel='Predicted',ylabel='True',title=title); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def main():
    feats={}; statics={}; split={}
    for label in LABELS:
        split[label]={'train':[str(wav_path(label,i).relative_to(DATA)) for i in TRAIN_IDS], 'test':[str(wav_path(label,i).relative_to(DATA)) for i in TEST_IDS]}
        for i in TRAIN_IDS+TEST_IDS:
            p=wav_path(label,i)
            if not p.exists(): raise FileNotFoundError(p)
            feats[(label,i)],statics[(label,i)]=features(p,True)[0],features(p,False)[0]
    with open(OUT/'split.json','w',encoding='utf-8') as f: json.dump(split,f,ensure_ascii=False,indent=2)
    # Requirement A: MFCC, delta, delta-delta heatmaps for two words.
    for label,i in [('bat',1),('xuong',1)]:
        _,c=features(wav_path(label,i),False); f,_=features(wav_path(label,i),True)
        fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True)
        for ax,mat,title in zip(axs,[f[:,:13].T,f[:,13:26].T,f[:,26:39].T],['MFCC','Delta MFCC','Delta-delta MFCC']):
            sns.heatmap(mat,ax=ax,cmap='mako',cbar=True); ax.set_ylabel(title)
        axs[-1].set_xlabel('Frame'); fig.suptitle(f'Dynamic MFCC features: {label}'); fig.tight_layout(); fig.savefig(FIG/f'features_{label}.png',dpi=150); plt.close(fig)
    # B/C: HMMs and model size experiment.
    hmms={}; train_time={}; convergence={}; trans=[]
    for label in LABELS:
        seq=[feats[(label,i)] for i in TRAIN_IDS]
        start=time.perf_counter(); model=LeftRightGaussianHMM(5).fit(seq); train_time[label]=time.perf_counter()-start
        hmms[label]=model; convergence[label]={'iterations':len(model.history),'final_train_loglik':model.history[-1]}
        trans.append({'label':label,'transmat':model.transmat_.tolist() if hasattr(model,'transmat_') else model.transmat.tolist()})
        np.savez_compressed(MODELS_DIR/f'{label}_hmm5.npz',startprob=model.startprob,transmat=model.transmat,means=model.means,variances=model.vars,history=model.history)
    # Primary scores, predictions, Viterbi paths.
    score_rows=[]; preds=[]; vit_rows=[]; test_time=[]
    for label in LABELS:
        for i in TEST_IDS:
            x=feats[(label,i)]; scores={w:m.score(x) for w,m in hmms.items()}
            order=sorted(scores,key=scores.get,reverse=True); pred=order[0]; preds.append((label,pred))
            score_rows.append({'file':f'{label}_{i:02d}.wav','true':label,'pred':pred,**{f'score_{w}':scores[w] for w in LABELS},'top1':order[0],'top2':order[1],'margin':scores[order[0]]-scores[order[1]]})
            begin=time.perf_counter(); _,path=hmms[label].decode(x); test_time.append(time.perf_counter()-begin)
            vit_rows.append({'label':label,'file':f'{label}_{i:02d}.wav','path':path.tolist(),'counts':np.bincount(path,minlength=5).tolist()})
            fig,ax=plt.subplots(figsize=(8,2.5)); ax.step(np.arange(len(path)),path,where='post'); ax.set(yticks=range(5),xlabel='Frame',ylabel='HMM state',title=f'Viterbi alignment {label}_{i:02d}'); fig.tight_layout(); fig.savefig(FIG/f'viterbi_{label}_{i:02d}.png',dpi=150); plt.close(fig)
    scores_df=pd.DataFrame(score_rows); scores_df.to_csv(OUT/'hmm_test_scores.csv',index=False)
    cm=confusion_matrix([a for a,b in preds],[b for a,b in preds],labels=LABELS); plot_cm(cm,LABELS,FIG/'hmm_confusion.png','5-state Gaussian HMM')
    # DTW baseline, same training/test data, static 13D MFCC, three train templates/class.
    dtw_rows=[]; dtw_train={w:[statics[(w,i)] for i in TRAIN_IDS] for w in LABELS}
    for label in LABELS:
        for i in TEST_IDS:
            x=statics[(label,i)]; t=time.perf_counter(); ds={w:min(dtw_distance(x,z) for z in dtw_train[w]) for w in LABELS}; elapsed=time.perf_counter()-t
            dtw_rows.append({'file':f'{label}_{i:02d}.wav','true':label,'pred':min(ds,key=ds.get),**{f'distance_{w}':ds[w] for w in LABELS},'seconds':elapsed})
    dtw_df=pd.DataFrame(dtw_rows); dtw_df.to_csv(OUT/'dtw_test_scores.csv',index=False)
    dtw_cm=confusion_matrix(dtw_df.true,dtw_df.pred,labels=LABELS); plot_cm(dtw_cm,LABELS,FIG/'dtw_confusion.png','MFCC + DTW baseline')
    # E1: train/test accuracy n_states 3,5,7; 5 is represented above.
    e1=[]
    for ns in [3,5,7]:
        t0=time.perf_counter()
        models_ns=hmms if ns==5 else {w:LeftRightGaussianHMM(ns).fit([feats[(w,i)] for i in TRAIN_IDS]) for w in LABELS}
        yy=[]; pp=[]
        for label in LABELS:
            for i in TEST_IDS:
                yy.append(label); pp.append(max(models_ns,key=lambda w:models_ns[w].score(feats[(label,i)])))
        e1.append({'n_states':ns,'accuracy':float(accuracy_score(yy,pp)),'mean_iterations':float(np.mean([len(m.history) for m in models_ns.values()])),'train_seconds':time.perf_counter()-t0})
    pd.DataFrame(e1).to_csv(OUT/'state_experiment.csv',index=False)
    # A static-vs-dynamic feature test with 5-state HMM.
    static_hmms={w:LeftRightGaussianHMM(5).fit([statics[(w,i)] for i in TRAIN_IDS]) for w in LABELS}
    yy=[]; pp=[]
    for w in LABELS:
        for i in TEST_IDS: yy.append(w); pp.append(max(static_hmms,key=lambda k:static_hmms[k].score(statics[(w,i)])))
    # Robustness: add white noise at a controlled 20 dB SNR to test waveforms; no extra training.
    rng=np.random.default_rng(SEED); robust=[]
    for label in LABELS:
        for i in TEST_IDS:
            y,sr=librosa.load(wav_path(label,i),sr=SR,mono=True); y,_=librosa.effects.trim(y,top_db=30)
            sigma=np.sqrt(np.mean(y*y)/(10**(20/10))); noisy=y+rng.normal(0,sigma,len(y))
            # Feature routine's exact frontend, without creating temporary files.
            z=librosa.effects.preemphasis(noisy,coef=0.97)
            c=librosa.feature.mfcc(y=z,sr=sr,n_mfcc=13,n_fft=512,win_length=400,hop_length=160,window='hamming',n_mels=26,fmin=20,fmax=sr/2)
            f=np.vstack([c,librosa.feature.delta(c,order=1,mode='nearest'),librosa.feature.delta(c,order=2,mode='nearest')]).T
            f=(f-f.mean(0))/(f.std(0)+1e-8); sc={w:m.score(f) for w,m in hmms.items()}
            robust.append({'file':f'{label}_{i:02d}.wav','true':label,'pred':max(sc,key=sc.get),'snr_db':20})
    pd.DataFrame(robust).to_csv(OUT/'noise_robustness.csv',index=False)
    # F: phone inventory is deliberately marked illustrative; instructors must replace it with the class inventory.
    lexicon={'bat':['b','at'],'dung':['d','ung'],'len':['l','en'],'xuong':['x','uong'],'trai':['tr','ai']}
    with open(HERE/'lexicon.txt','w',encoding='utf-8') as f:
        for w,phones in lexicon.items(): f.write(w+' '+' '.join(phones)+'\n')
    # G: compact teaching corpus, counts and add-k smoothing; explicit context example and lambda sweep.
    corpus=[['bat','len'],['bat','xuong'],['dung','len'],['dung','xuong'],['len','trai'],['xuong','trai'],['trai','dung'],['len','dung'],['xuong','dung'],['bat','dung'],['dung','trai'],['trai','len']]
    unigram={w:0 for w in LABELS}; c1={w:0 for w in LABELS}; c2={}
    for sent in corpus:
        for w in sent: unigram[w]+=1
        for a,b in zip(sent,sent[1:]):
            c1[a]+=1; c2[(a,b)]=c2.get((a,b),0)+1
    with open(HERE/'command_corpus.txt','w',encoding='utf-8') as f:
        for sent in corpus: f.write(' '.join(sent)+'\n')
    with open(OUT/'bigram_counts.json','w',encoding='utf-8') as f: json.dump({'unigram':unigram,'bigram_context_counts':c1,'bigram':{f'{a} {b}':v for (a,b),v in c2.items()}},f,ensure_ascii=False,indent=2)
    # Pick closest top-1/top-2 acoustic trial and context prior from the toy corpus.
    rrow=scores_df.sort_values('margin').iloc[0]; prev='dung'; fvec=feats[(rrow.true,int(rrow.file.split('_')[1][:2]))]
    lmrows=[]
    for lam in [0,.1,.3,1.,20.,100.]:
        combined={w:hmms[w].score(fvec)/len(fvec)+lam*np.log(bigram_prob(prev,w,c2,c1,len(LABELS))) for w in LABELS}
        lmrows.append({'file':rrow.file,'previous_word':prev,'lambda':lam,'acoustic_top1':rrow.top1,'reranked':max(combined,key=combined.get),'top_candidates':json.dumps(sorted(combined,key=combined.get,reverse=True)[:2])})
    pd.DataFrame(lmrows).to_csv(OUT/'lm_reranking.csv',index=False)
    assert parse_lexicon(HERE/'lexicon.txt') == lexicon
    # Score separation and high-quantile threshold suggestion (small sample, exploratory).
    per=[]
    for row in score_rows: per.append(row['score_'+row['true']]/len(feats[(row['true'],int(row['file'].split('_')[1][:2]))]))
    metrics={'labels':LABELS,'split':split,'hmm_accuracy':float(accuracy_score([a for a,b in preds],[b for a,b in preds])),
      'dtw_accuracy':float(accuracy_score(dtw_df.true,dtw_df.pred)),'noise_20db_accuracy':float(accuracy_score([x['true'] for x in robust],[x['pred'] for x in robust])),
      'static_mfcc_hmm_accuracy':float(accuracy_score(yy,pp)) if False else None,
      'mean_hmm_training_seconds':float(np.mean(list(train_time.values()))),'mean_hmm_test_decode_seconds':float(np.mean(test_time)),
      'mean_dtw_test_seconds':float(dtw_df.seconds.mean()),'hmm_confusion':cm.tolist(),'dtw_confusion':dtw_cm.tolist(),
      'hmm_train':convergence,'state_experiment':e1,'test_count':len(preds),'per_frame_correct_score_quantiles':np.quantile(per,[0,.1,.5,.9,1]).tolist(),
      'top_margin_mean':float(scores_df.margin.mean()),'full_covariance':'not fitted: 3 training utterances/class are insufficient for a stable 39-D covariance estimate'}
    # static accuracy explicitly reconstructed
    sy=[]; sp=[]
    for w in LABELS:
        for i in TEST_IDS: sy.append(w); sp.append(max(static_hmms,key=lambda k:static_hmms[k].score(statics[(w,i)])))
    metrics['static_mfcc_hmm_accuracy']=float(accuracy_score(sy,sp))
    with open(OUT/'metrics.json','w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    pd.DataFrame([{'model':'HMM 39D MFCC+delta+delta2','accuracy':metrics['hmm_accuracy'],'train_templates_per_class':3,'mean_train_seconds':metrics['mean_hmm_training_seconds'],'test_seconds_per_trial':metrics['mean_hmm_test_decode_seconds'],'top_confusion':top_confusion(cm)},
      {'model':'DTW 13D MFCC','accuracy':metrics['dtw_accuracy'],'train_templates_per_class':3,'mean_train_seconds':np.nan,'test_seconds_per_trial':metrics['mean_dtw_test_seconds'],'top_confusion':top_confusion(dtw_cm)}]).to_csv(OUT/'hmm_vs_dtw.csv',index=False)
    pd.DataFrame(vit_rows).to_json(OUT/'viterbi_paths.json',orient='records',force_ascii=False,indent=2)
    print(json.dumps({k:metrics[k] for k in ['hmm_accuracy','dtw_accuracy','noise_20db_accuracy','static_mfcc_hmm_accuracy','state_experiment','mean_hmm_training_seconds','mean_hmm_test_decode_seconds','mean_dtw_test_seconds']},indent=2))

def top_confusion(cm):
    a=cm.copy(); np.fill_diagonal(a,0)
    if a.max()==0:return 'none'
    i,j=np.unravel_index(np.argmax(a),a.shape); return f'{LABELS[i]}->{LABELS[j]} ({a[i,j]})'

if __name__=='__main__': main()
