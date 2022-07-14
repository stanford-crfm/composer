import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-name", help="artifact to download")
    parser.add_argument("--config", help="artifact config")
    parser.add_argument("--output-path", help="where to store")
    parser.add_argument("--no-convert", action="store_true", help="apply conversion script?")
    args = parser.parse_args()

    # download
    work_dir = "tmp-wandb-download"
    print("Creating tmp work dir and final output path ...")
    subprocess.call(f"mkdir -p {args.output_path}", shell=True)
    subprocess.call(f"mkdir {work_dir}", shell=True)
    print("Downloading wandb artifact ...")
    subprocess.call(f"wandb artifact get --root {work_dir} {args.artifact_name}", shell=True)
    print("Expanding artifact ...")
    subprocess.call(f"cd {work_dir} ; tar -xf *tar", shell=True)
    if not args.no_convert:
        print("Converting to HF format ...")
        subprocess.call(f"python convert_deepspeed_to_hf_model.py --input-checkpoint {work_dir}/deepspeed/mp_rank_00_model_states.pt --config {args.config} --output-dir {args.output_path}", shell=True)
    print("Removing tmp work dir ...")
    subprocess.call(f"rm -rf {work_dir}", shell=True)
    print("Done.")
        
   

if __name__ == "__main__":
    main()
