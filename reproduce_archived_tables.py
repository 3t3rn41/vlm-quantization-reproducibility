"""Regenerate the archived aggregate numeric tables from released inputs."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "revise/Analysis_2/Major_4/reproduction_v20261102"
INPUTS = (
    "05_primary_metrics/078_primary_metrics_v1_3_corrected.json",
    "06_risk_ablation/079_risk_signal_ablation_v1_3_corrected.json",
    "07_candidate_mass/080_candidate_mass_metrics_v1_3_recovered.json",
    "08_queue_utility/081_queue_metrics_v1_3_corrected.json",
    "08_queue_utility/081_queue_metrics_v1_3_corrected_meta.json",
    "09_rbo/082_rbo_v1_3.json",
    "10_bootstrap/083_bootstrap_v1_3.json",
    "11_cost/084_memory_summary_v1_3.json",
    "12_coverage/085_quantization_coverage_summary_v1_3.json",
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="vlm_reproduction_") as tmp:
        base = Path(tmp)
        scripts = base / "inference_scripts/pipeline"
        scripts.mkdir(parents=True)
        for name in ("086_render_tables.py",):
            shutil.copy2(ARCHIVE / "inference_scripts/pipeline" / name, scripts / name)
        source = ARCHIVE / "artifacts/eswa_revision_v1"
        for rel in INPUTS:
            target = base / "artifacts/eswa_revision_v1" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, target)
        for name in ("086_render_tables.py",):
            subprocess.run([sys.executable, str(scripts / name)], check=True, cwd=base)
        artifacts = base / "artifacts/eswa_revision_v1"
        table_audit = json.loads((artifacts / "13_tables/audit/086_table_validity_audit.json").read_text())
        assert table_audit["status"] == "PASS", table_audit
        tables = len(list((artifacts / "13_tables/tables").glob("Table_*.csv")))
        assert tables == 8, tables
        print(json.dumps({"status": "PASS", "tables": tables}))


if __name__ == "__main__":
    main()
