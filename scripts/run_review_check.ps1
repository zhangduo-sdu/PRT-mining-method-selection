param(
    [string]$Python = "python"
)

& $Python scripts/run_integrity_checks.py
& $Python src/run_experiment.py `
    --data data/synthetic_benchmark.csv `
    --config configs/prt_config.yaml `
    --outdir outputs/reviewer_check `
    --epochs 5
