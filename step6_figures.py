"""Adım 6: Tüm figürler ve tablolar"""
import subprocess, sys
result = subprocess.run([
    sys.executable, "generate_paper_outputs.py",
    "--results_dir", "results/ablation_main",
    "--pvs_results", "results/pvs_results.json",
], cwd=".")
if result.returncode == 0:
    print("\n✓ ADIM 6 TAMAMLANDI — tüm çıktılar hazır!\n")
else:
    print("\n✗ Figür üretimi başarısız\n")
