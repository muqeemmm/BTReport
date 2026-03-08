import os
import argparse
import subprocess
from pathlib import Path



def check_env_variables():
    if "OLLAMA_SIF" not in os.environ:
        raise RuntimeError("Set OLLAMA_SIF. Syntax: export OLLAMA_SIF=/path/to/ollama.sif")
    if "OLLAMA_MODELS" not in os.environ:
        raise RuntimeError("Set OLLAMA_MODELS. Syntax: export OLLAMA_MODELS=/path/to/ollama_models ")
    if "OLLAMA_HOST" not in os.environ:
        raise RuntimeError("Set OLLAMA_HOST. Syntax: export OLLAMA_HOST=http://127.0.0.1:50505")

import os
import subprocess
from pathlib import Path

def start_ollama(gpus="0"):
    check_env_variables()
    sif = os.environ["OLLAMA_SIF"]
    models = os.environ["OLLAMA_MODELS"]
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:50505")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["APPTAINERENV_OLLAMA_MODELS"] = models

    # DYNAMIC CERTIFICATE DISCOVERY 
    host_cert_locations = [
        "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL / CentOS / Hyak
        "/etc/ssl/certs/ca-certificates.crt", # Ubuntu / Debian
        "/etc/ssl/ca-bundle.pem",             # OpenSUSE
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem" 
    ]
    
    active_host_cert = next((p for p in host_cert_locations if os.path.exists(p)), None)
    container_cert_path = "/etc/ssl/certs/ca-certificates.crt"

    # CONSTRUCT BINDS 
    binds = [f"{Path(models)}:{Path(models)}"]
    if active_host_cert:
        binds.append(f"{active_host_cert}:{container_cert_path}")

    # LOGGING OUTPUT 
    print("Ollama Server Launch Configuration ")
    print(f"  Container SIF:  {sif}")
    print(f"  Model Storage:  {models}")
    print(f"  Ollama Host:    {ollama_host}")
    print(f"  GPU(s) active:  {gpus}")
    if active_host_cert:
        print(f"  Cert Mapping:   [HOST] {active_host_cert} -> [CONTAINER] {container_cert_path}")
    else:
        print("  Cert Mapping:   NONE (System CA bundle not found on host)")
    print("-")

    bind_args = []
    for b in binds:
        bind_args.extend(["-B", b])

    try:
        subprocess.run(
            [
                "apptainer",
                "exec",
                "--nv",
                "-e",
                "--env", f"OLLAMA_HOST={ollama_host}",
                *bind_args,
                sif,
                "ollama",
                "serve",
            ],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"\nError: Ollama server exited with status {e.returncode}")
    except KeyboardInterrupt:
        print("\nShutting down Ollama server...")


def check_ollama_server():
    "Check if Ollama server is running."
    try:
        # _, host = ENV.split("=", 1)
        host = os.environ["OLLAMA_HOST"]
        subprocess.run(
            ["curl", "--noproxy", "*", "-sf", f"{host}/api/tags"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        print(f"Ollama server found at {host}")
        return True
    except Exception:
        raise RuntimeError(f"Ollama server at {host} not reachable")
def pull_llm(model):
    check_env_variables()
    check_ollama_server() # This ensures the healthy server is already running
    
    sif = os.environ["OLLAMA_SIF"]
    models = os.environ["OLLAMA_MODELS"]
    
    env = os.environ.copy()
    env["APPTAINERENV_OLLAMA_MODELS"] = models

    subprocess.run(
        [
            "apptainer",
            "exec",
            "-e", 
            "--env", f"OLLAMA_HOST={os.environ['OLLAMA_HOST']}",
            "-B", f"{models}:{models}",
            sif,
            "ollama",
            "pull",
            model,
        ],
        check=True,
        env=env,
    )



def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start-ollama")
    p_start.add_argument("--gpus", default="0")

    p_pull = sub.add_parser("pull-llm")
    p_pull.add_argument("model")

    args = p.parse_args()

    if args.cmd == "start-ollama":
        start_ollama(args.gpus)
    elif args.cmd == "pull-llm":
        pull_llm(args.model)

    else:
        raise ValueError('Command not valid. Choose one of: ["start-ollama", "pull-llm"]')


if __name__ == "__main__":
    main()
