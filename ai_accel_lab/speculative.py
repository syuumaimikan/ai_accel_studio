from __future__ import annotations
import time
import pandas as pd
import torch
from torch import nn

class TinyCausalLM(nn.Module):
    """Small causal Transformer used only to demonstrate batched speculative verification."""
    def __init__(self,vocab=128,d_model=128,nhead=4,layers=2,max_len=512,seed=1):
        super().__init__(); torch.manual_seed(seed); self.vocab=vocab
        self.tok=nn.Embedding(vocab,d_model); self.pos=nn.Embedding(max_len,d_model)
        enc=nn.TransformerEncoderLayer(d_model,nhead,d_model*4,batch_first=True,activation='gelu')
        self.net=nn.TransformerEncoder(enc,layers); self.head=nn.Linear(d_model,vocab,bias=False)
    def forward(self,ids):
        b,t=ids.shape; p=torch.arange(t,device=ids.device); x=self.tok(ids)+self.pos(p)[None,:,:]
        mask=torch.triu(torch.ones(t,t,device=ids.device,dtype=torch.bool),diagonal=1)
        return self.head(self.net(x,mask=mask))

@torch.no_grad()
def greedy(target,prefix,n_new):
    seq=prefix.clone(); calls=0
    for _ in range(n_new):
        logits=target(seq); calls+=1; nxt=logits[:,-1].argmax(-1,keepdim=True); seq=torch.cat([seq,nxt],1)
    return seq,calls

@torch.no_grad()
def speculative(target,draft,prefix,n_new,k=4):
    seq=prefix.clone(); generated=0; tcalls=0; dcalls=0; accepted=0
    while generated<n_new:
        props=[]; dseq=seq
        for _ in range(min(k,n_new-generated)):
            dl=draft(dseq); dcalls+=1; p=dl[:,-1].argmax(-1,keepdim=True); props.append(p); dseq=torch.cat([dseq,p],1)
        proposal=torch.cat(props,1); candidate=torch.cat([seq,proposal],1)
        tl=target(candidate); tcalls+=1
        # logits at seq_last predicts prop1; logits at prop1 predicts prop2, ...
        start=seq.shape[1]-1
        pred=tl[:,start:start+proposal.shape[1]].argmax(-1)
        mismatch=None
        for j in range(proposal.shape[1]):
            if not torch.equal(pred[:,j:j+1],proposal[:,j:j+1]): mismatch=j; break
        if mismatch is None:
            take=min(proposal.shape[1],n_new-generated); seq=torch.cat([seq,proposal[:,:take]],1); generated+=take; accepted+=take
        else:
            if mismatch>0:
                seq=torch.cat([seq,proposal[:,:mismatch]],1); generated+=mismatch; accepted+=mismatch
            if generated<n_new:
                seq=torch.cat([seq,pred[:,mismatch:mismatch+1]],1); generated+=1
    return seq,{'target_calls':tcalls,'draft_calls':dcalls,'accepted':accepted,'accept_rate':accepted/max(1,n_new)}

def run_speculative(device_arg='auto',tokens=32,proposal_len=4):
    from .utils import resolve_device,sync,seed_all
    seed_all(); device=resolve_device(device_arg)
    target=TinyCausalLM(d_model=128,layers=2,seed=1).to(device).eval()
    # Same width/seed, but only half the Transformer depth. This keeps the draft cheaper while
    # making its distribution substantially closer to the target than an unrelated random model.
    draft=TinyCausalLM(d_model=128,nhead=4,layers=1,seed=1).to(device).eval()
    prefix=torch.tensor([[1,2,3]],device=device)
    sync(device); t=time.perf_counter(); gb,gc=greedy(target,prefix,tokens); sync(device); bms=(time.perf_counter()-t)*1000
    sync(device); t=time.perf_counter(); sp,st=speculative(target,draft,prefix,tokens,proposal_len); sync(device); sms=(time.perf_counter()-t)*1000
    return pd.Series({'device':str(device),'tokens':tokens,'baseline_ms':bms,'speculative_ms':sms,'wallclock_speedup':bms/sms,'baseline_target_calls':gc,'spec_target_calls':st['target_calls'],'draft_calls':st['draft_calls'],'accept_rate':st['accept_rate'],'identical_to_target_greedy':bool(torch.equal(gb,sp))})
