"""Launcher for training — writes progress to a log file with flushing."""
import subprocess
import sys
import os

log_file = 'training_log.txt'

with open(log_file, 'w') as log:
    log.write("Starting training...\n")
    log.flush()

    proc = subprocess.Popen(
        [sys.executable, '-u', 'train.py', '--config', 'config_2048.yaml'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    for line in proc.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()

    proc.wait()
    log.write(f"\nProcess exited with code {proc.returncode}\n")
    log.flush()

print(f"\nTraining finished. Log saved to {log_file}")
