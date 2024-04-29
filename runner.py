from datetime import datetime, timezone
from enum import Enum
import glob
import os
import argparse
import sys
from more_itertools import flatten
import numpy as np
import subprocess as sp
from pathlib import Path

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
    DS_TIME_CHECK_EACH = "ds_check_each"

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
CUSPATIAL_CMD = ["python3", "cuspatial_runner.py", "-n", f"{N}"]
# GEO_RT_CMD = ["./build/release/geo-rt-index", "-n", f"{N}", "-q", "0", "0", "0.5", "0.5"]

def get_session_str():
    return f"s{SEED}_b{BENCHMARK.value}_d{DIST.value}_p{PROG.value}_n{N}"

np.random.seed(SEED)
TIME = datetime.now(timezone.utc).isoformat()
SESSION_OUTPUT_DIR = os.path.join(OUTPUT_DATA_DIR, get_session_str(), TIME)
os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)

def mk_query(selectivity: float, lo = 0, hi = 1):
    # raise NotImplementedError()
    assert(0 < selectivity and selectivity <= 1)
    match DIST:
        case Distribution.UNIFORM:
            LIMIT = 1000
            # used ChatGPT for this loop
            for _ in range(LIMIT):
                area = selectivity
                # Generate random dimensions
                width = np.random.uniform(1, area)  # Limit the width to be less than or equal to the area
                height = area / width
                
                # Check if the dimensions produce the desired area
                if abs(width * height - area) < 1e-9:
                    # return width, height
                    x1 = np.random.uniform(lo, hi - width)
                    y1 = np.random.uniform(lo, hi - height)
                    x2 = x1 + width
                    y2 = y1 + height
                    return x1, y1, x2, y2 # minx, miny, maxx maxy semantics
            raise Exception("Limit reached generating query for uniform distribution")
        case _: raise NotImplementedError(DIST.value)

def mk_queries(selectivity: float, n: int, lo = 0, hi = 1):
    for _ in range(n):
        yield mk_query(selectivity, lo, hi)

match BENCHMARK:
    case Benchmark.DS_SCALING:
        datasets = np.random.choice(FILES, 8, True).tolist()
        file_count = 1
        while file_count <= len(datasets):
            try:
                prog_out=open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_prog.txt"), "x")
                # smi_out= open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_smi.txt"), "x")
                files = datasets[:file_count]
                cmd = CUSPATIAL_CMD if PROG == Program.CUSPATIAL else []
                local_cmd = cmd + ["-q", "0", "0", "0.5", "0.5"] + files
                local_cmd_str = " ".join(local_cmd)
                prog_out.write(local_cmd_str)
                prog_out.write('\n')
                prog_out.flush()
                # print("cmd:", local_cmd_str)
                prog_process = sp.Popen(local_cmd, stdout=prog_out, stderr=sys.stderr)
                # smi_process = sp.Popen(SMI_CMD, stdout=smi_out, stderr=sys.stderr)
                prog_process.wait()
                # smi_process.kill()
                file_count *= 2
            finally:
                prog_out.close()
                # smi_out.close()
        
    case Benchmark.DS_TIME_CHECK_EACH:
        # Check each data set has approximately the same run time
        LO = -1
        HI = 1
        QUERIES = list(flatten(map(lambda t: [f"-q", f"{t[0]}", f"{t[1]}", f"{t[2]}", f"{t[3]}"], [
            *mk_queries(0.01, 5, LO, HI), 
            *mk_queries(0.02, 5, LO, HI),
            *mk_queries(0.05, 5, LO, HI),
            *mk_queries(0.10, 5, LO, HI),
            *mk_queries(0.20, 5, LO, HI)
        ])))
    
        # QUERIES = [
        #     # 1% selectivity on corners
        #     # ["-q", f"0",     f"0",    f"0.1",  f"0.1"], # ul
        #     # ["-q", f"0.9",   f"0.9",  f"1",    f"1"], # lr
        #     # ["-q", f"0.9",   f"0",    f"1",    f"0.1"], # ur
        #     # ["-q", f"0",     f"0.9",  f"0.1",  f"1"], # ll
        #     # 2% selectivity on corners
        #     ["-q", f"{LO}",          f"{LO}",             f"{LO + 0.14142}",   f"{LO + 0.14142}"], # ul
        #     ["-q", f"{HI - 0.14142}",f"{HI - 0.14142}",   f"{HI}",             f"{HI}"], # lr
        #     ["-q", f"{HI - 0.14142}",f"{LO}",             f"{HI}",             f"{LO + 0.14142}"], # ur
        #     ["-q", f"{LO}",          f"{HI - 0.14142}",   f"{LO + 0.14142}",   f"{HI}"], # ll
        #     # 5% selectivity on corners
        #     ["-q", f"{LO}",          f"{LO}",             f"{LO + 0.22361}",   f"{LO + 0.22361}"], # ul
        #     ["-q", f"{HI - 0.22361}",f"{HI - 0.22361}",   f"{HI}",             f"{HI}"], # lr
        #     ["-q", f"{HI - 0.22361}",f"{LO}",             f"{HI}",             f"{LO + 0.22361}"], # ur
        #     ["-q", f"{LO}",          f"{HI - 0.22361}",   f"{LO + 0.22361}",   f"{HI}"], # ll
        #     # # 10% selectivity on corners
        #     # ["-q", f"{LO}",          f"{LO}",             f"{LO + 0.31623}",   f"{LO + 0.31623}"], # ul
        #     # ["-q", f"{HI - 0.31623}",f"{HI - 0.31623}",   f"{HI}",             f"{HI}"], # lr
        #     # ["-q", f"{HI - 0.31623}",f"{LO}",             f"{HI}",             f"{LO + 0.31623}"], # ur
        #     # ["-q", f"{LO}",          f"{HI - 0.31623}",   f"{LO + 0.31623}",   f"{HI}"], # ll
        #     # # 20% selectivity on corners
        #     # ["-q", f"{LO}",          f"{LO}",             f"{LO + 0.44721}",   f"{LO + 0.44721}"], # ul
        #     # ["-q", f"{HI - 0.44721}",f"{HI - 0.44721}",   f"{HI}",             f"{HI}"], # lr
        #     # ["-q", f"{HI - 0.44721}",f"{LO}",             f"{HI}",             f"{LO + 0.44721}"], # ur
        #     # ["-q", f"{LO}",          f"{HI - 0.44721}",   f"{LO + 0.44721}",   f"{HI}"], # ll
        # ]
        try:
            prog_out=open(os.path.join(SESSION_OUTPUT_DIR, f"lo{LO}hi{HI}_prog.txt"), "x")
            # smi_out= open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_smi.txt"), "x")
            cmd = CUSPATIAL_CMD if PROG == Program.CUSPATIAL else []
            for file in FILES:
                local_cmd = cmd + QUERIES + [file]
                local_cmd_str = " ".join(local_cmd)
                print(local_cmd_str)
                prog_out.write(local_cmd_str)
                prog_out.write('\n')
                prog_out.flush()
                # print("cmd:", local_cmd_str)
                prog_process = sp.Popen(local_cmd, stdout=prog_out, stderr=sys.stderr)
                # smi_process = sp.Popen(SMI_CMD, stdout=smi_out, stderr=sys.stderr)
                prog_process.wait()
            # smi_process.kill()
        finally:
            prog_out.close()
            # smi_out.close()

    case _: raise NotImplementedError(f"Unimplemented benchmark: {BENCHMARK.value}")

