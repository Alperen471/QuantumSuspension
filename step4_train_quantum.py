"""Adım 4: Quantum QFM eğitim — her run ayrı subprocess"""
import sys, json, time, subprocess
from pathlib import Path

sys.path.insert(0, ".")

CHECKPOINT = "results/step4_checkpoint.json"
RESULTS    = "results/ablation_main"
MODELS     = [
    ("quantum", 5,  "none",    "rl_q5_qfm",      200000),
    ("quantum", 10, "none",    "rl_q10_qfm",     200000),
    ("quantum", 20, "none",    "rl_q20_qfm",     200000),
    ("quantum", 5,  "bounded", "rl_q5_qfm_fuzzy",200000),
]
SEEDS = [42, 43, 44]
ROADS = ["random", "sinusoidal", "bump", "pothole"]

def load_cp():
    p = Path(CHECKPOINT)
    return json.load(open(p)) if p.exists() else {"done": []}

def save_cp(cp):
    Path(CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cp, open(CHECKPOINT,"w"), indent=2)

def run_one(enc, nq, fuzzy, mname, road, seed, steps):
    script = f"""
import sys; sys.path.insert(0,'.')
import torch, gc, json
from pathlib import Path
from rl.train import train, TrainingConfig
Path("{RESULTS}/models").mkdir(parents=True, exist_ok=True)
cfg = TrainingConfig(
    model_name="{mname}_{road}_{seed}", algorithm="sac",
    encoder_type="{enc}", n_qubits={nq}, variational=True, frozen=True,
    fuzzy_mode="{fuzzy}", road_type="{road}", seed={seed},
    total_steps={steps}, batch_size=256,
    eval_freq={max(10000,steps//20)}, eval_episodes=3, save_freq={steps},
    device="cuda", stats_path="dataset/statistics.json",
    save_dir="{RESULTS}/models", log_dir="{RESULTS}/logs",
)
r = train(cfg, verbose=True)
m = r.final_metrics
json.dump({{"model":"{mname}","road":"{road}","seed":{seed},
    "rms":float(m.get("rms_body_acc_mean",0)),
    "iso":float(m.get("iso_weighted_rms_mean",0)),
    "sd":float(m.get("rms_susp_defl_mean",0)),
    "cf":float(m.get("rms_ctrl_force_mean",0)),
    "min":r.total_time_s/60}},
    open("{RESULTS}/{mname}_{road}_{seed}.json","w"),indent=2)
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
"""
    p = subprocess.run([sys.executable,"-c",script],
                       cwd=".", timeout=7200)
    return p.returncode == 0

cp   = load_cp()
done = set(cp["done"])
runs = [(e,n,f,m,r,s,st) for e,n,f,m,st in MODELS
        for r in ROADS for s in SEEDS]
total = len(runs)

print(f"\n=== ADIM 4: QUANTUM EĞİTİM ({total} run) ===")
print(f"Tamamlanan: {len(done)}/{total}\n")

for i,(enc,nq,fuzzy,mname,road,seed,steps) in enumerate(runs):
    key = f"{mname}_{road}_{seed}"
    if key in done:
        print(f"[{i+1:2d}/{total}] {key} ATLA"); continue
    print(f"[{i+1:2d}/{total}] {key} başlıyor...", flush=True)
    try:
        ok = run_one(enc,nq,fuzzy,mname,road,seed,steps)
        if ok:
            done.add(key); cp["done"]=list(done); save_cp(cp)
            j = Path(f"{RESULTS}/{mname}_{road}_{seed}.json")
            d = json.load(open(j)) if j.exists() else {}
            print(f"[{i+1:2d}/{total}] ✓ rms={d.get('rms',0):.4f} iso={d.get('iso',0):.4f}")
    except Exception as e:
        print(f"[{i+1:2d}/{total}] ✗ {e}")

print("\n✓ ADIM 4 TAMAMLANDI — step5 çalıştırabilirsiniz\n")
