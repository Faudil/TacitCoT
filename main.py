import sys
import subprocess

def main():
    args = sys.argv[1:]
    
    if "--mode" in args:
        idx = args.index("--mode")
        mode_val = args[idx + 1] if idx + 1 < len(args) else None
        
        # Strip mode arguments and delegate
        clean_args = args[:idx] + args[idx + 2:]
        
        if mode_val == "train":
            cmd = [sys.executable, "train.py"] + clean_args
        elif mode_val == "train_rl":
            cmd = [sys.executable, "train_rl.py"] + clean_args
        else:
            cmd = [sys.executable, "manual_test.py"] + clean_args
    else:
        cmd = [sys.executable, "manual_test.py"] + args

    subprocess.run(cmd)

if __name__ == "__main__":
    main()