import sys, glob
from tensorboard.backend.event_processing import event_accumulator
d=sys.argv[1]
ev=sorted(glob.glob(d+"/events.out.tfevents*"))
ea=event_accumulator.EventAccumulator(ev[-1], size_guidance={event_accumulator.SCALARS:0}); ea.Reload()
tags=ea.Tags()['scalars']
want=[t for t in tags if any(k in t.lower() for k in ['learning_rate','noise_std','mean_std','kl','entropy'])]
print("matched tags:", want)
for t in want:
    s=ea.Scalars(t)
    vals=[round(x.value,6) for x in s]
    print(f"\n{t}: n={len(s)} first={vals[0]} last={vals[-1]} min={min(vals)} max={max(vals)}")
    # sample every ~len/8
    step=max(1,len(s)//8)
    print("  steps:", [(s[i].step, round(s[i].value,5)) for i in range(0,len(s),step)])
