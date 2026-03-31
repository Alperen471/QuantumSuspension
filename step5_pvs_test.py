"""Adım 5: PVS gerçek veri testi"""
import subprocess, sys
result = subprocess.run([
    sys.executable, "-m", "experiments.run_pvs_evaluation",
    "--pvs_dir",     "dataset/data_test",
    "--results_dir", "results/ablation_main",
    "--model_name",  "rl_q5_qfm_fuzzy",
    "--output",      "results/pvs_results.json",
    "--n_pvs",       "9",
], cwd=".")
if result.returncode == 0:
    print("\n✓ ADIM 5 TAMAMLANDI — step6 çalıştırabilirsiniz\n")
else:
    print("\n✗ PVS testi başarısız\n")
