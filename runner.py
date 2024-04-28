from datetime import datetime, timezone
from enum import Enum
import glob
import os
import argparse
import sys
import numpy as np
import subprocess as sp

def get_system():
    import platform
    match platform.system():
        case "Linux":
            if platform.platform() == 'Linux-5.15.0-105-generic-x86_64-with-glibc2.35':
                return "ubuntu"
            else:
                return "ucloud"
        case _:
            raise Exception(f"Unsupported system {platform.system()}")

class Distribution(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"

class Program(Enum):
    CUSPATIAL = "cuspatial"
    GEO_RT_INDEX = "geo-rt-index"

class Benchmark(Enum):
    DS_SCALING = "dataset_scaling"

parser = argparse.ArgumentParser(description='Cuspatial runner')
parser.add_argument('-s', type=int, help='Seed for numpy.random',required=False, default=0)
parser.add_argument("-d", type=str, help='Distribution', required=True,choices=[e.value for e in Distribution])
parser.add_argument("-b", type=str, help="Benchmark to run", required=True,choices=[e.value for e in Benchmark])
parser.add_argument("-p", type=str, help="Program to benchmark", required=True,choices=[e.value for e in Program])
args = parser.parse_args()

SEED = args.s
BENCHMARK = Benchmark(args.b)
DIST = Distribution(args.d)
PROG = Program(args.p)

N = 5
INPUT_DATA_DIR = os.path.join("/home/aske/dev/geo-rt-index/data" if get_system() == "ubuntu" else "/home/ucloud/geo-rt-index/data", DIST.value)
FILES = glob.glob(os.path.join(INPUT_DATA_DIR, "*.parquet"))
OUTPUT_DATA_DIR  = "/home/aske/dev/geo-rt-index/data/runs" if get_system() == "ubuntu" else "/home/ucloud/geo-rt-index/data/runs"
SMI_CMD = [
    "nvidia-smi",
    "--query-gpu",
    "clocks.gr,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.free,memory.reserved,memory.used",
    "-lms",
    "200",
    "--format=csv,nounits"
]
CUSPATIAL_CMD = ["python3", "cuspatial_runner.py", "-n", f"{N}", "-q", "0", "0", "0.5", "0.5"]
# GEO_RT_CMD = ["./build/release/geo-rt-index", "-n", f"{N}", "-q", "0", "0", "0.5", "0.5"]

def get_session_str():
    return f"s{SEED}_b{BENCHMARK.value}_d{DIST.value}_p{PROG.value}_n{N}"

np.random.seed(SEED)

datasets = np.random.choice(FILES, 8, True).tolist()

TIME = datetime.now(timezone.utc).isoformat()
SESSION_OUTPUT_DIR = os.path.join(OUTPUT_DATA_DIR, get_session_str(), TIME)
os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)

file_count = 1
while file_count <= len(datasets):
    try:
        prog_out=open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_prog.txt"), "x")
        # smi_out= open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_smi.txt"), "x")
        files = datasets[:file_count]
        cmd = CUSPATIAL_CMD if PROG == Program.CUSPATIAL else []
        local_cmd = cmd + files
        print("cmd:", " ".join(local_cmd))
        prog_process = sp.Popen(local_cmd, stdout=prog_out, stderr=sys.stderr)
        # smi_process = sp.Popen(SMI_CMD, stdout=smi_out, stderr=sys.stderr)
        prog_process.wait()
        # smi_process.kill()
        file_count *= 2
    finally:
        prog_out.close()
        # smi_out.close()
