import subprocess
import sys
import os
import time

def run_step(script_name):
    print(f"\n{'='*60}")
    print(f" STARTING: {script_name}")
    print(f"{'='*60}", flush=True)
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n[SUCCESS] {script_name} finished in {duration:.2f}s", flush=True)
    else:
        print(f"\n[FAILED] {script_name} exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)

pipeline = [
    "corpus_validator.py",
    "matrix_embeddings.py",
    "w2v_training_logic.py",
    "w2v_evaluation_suite.py",
    "tagging_data_generator.py",
    "bilstm_tagger_train.py",
    "bilstm_tagger_eval.py",
    "topic_data_processor.py",
    "transformer_topic_classifier.py"
]

if __name__ == "__main__":
    print("NLP Master Runner Initiated...")
    for script in pipeline:
        run_step(script)
    print("\nALL PIPELINE STAGES COMPLETED SUCCESSFULLY.")
