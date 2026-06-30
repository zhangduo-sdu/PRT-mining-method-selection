param(
    [string]$Python = "python"
)

& $Python src/model_summary.py --config configs/prt_config.yaml
