# Cheatsheet

**See ports busy with non-docker processes**
- Install lsof
```bash
sudo apt-get update
sudo apt-get install lsof
```
- Example: port 6006
```bash
lsof -iTCP:6006 -sTCP:LISTEN -n -P
```

- Kill the process
```bash
kill <PID>
```

Start MLflow tracking server either using this command (SQLite backend + local artifact store):
```bash
mlflow server \
  --backend-store-uri sqlite:////workplace/SkiNet/mlflow.db \
  --default-artifact-root file:///workplace/SkiNet/mlruns \
  --host 0.0.0.0 \
  --port 5000
```

or running a setup script:
```bash
chmod +x start_mlflow.sh
./start_mlflow.sh
```

Open MLflow UI in browser on port 5000. If using Lightning studio, ssh to it via
```bash
ssh -N -L 5000:localhost:5000 ssh_connection_string_from_your_studio@ssh.lightning.ai
```


## Lightning + MLflow notes

With `TRAIN_CONFIG.use_mlflow_logger: true`, runs now capture:
- Training/validation metrics (`train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`) and test metrics (`test_loss`, `test_accuracy`, `average_test_accuracy`) when `run_test_after_fit: true`.
- Fit + optimizer params such as batch size, max epochs, optimizer name, learning rate, and epsilon.
- Model summary artifact at fit start (`model/model_summary.txt`) when `mlflow_log_model_summary: true`.
- Lightning model artifact at fit end via `mlflow_log_model`.
- Early stopping params/state when `use_early_stopping: true` (for example `patience`, `min_delta`, `best_score`, `wait_count`, `stopped_epoch`, and `triggered`).
- Best checkpoint artifact under `checkpoints/best` when early stopping triggers and a best model path is available.
