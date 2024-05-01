from datetime import datetime, timezone
from enum import Enum
import glob
import os
import argparse
import sys
from typing import Iterable
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
    QUERY_SCALING = "query_scaling"
    AABB_LAYERING_SCALING = "aabb_layering"
    RAYS_PER_THREAD_SCALING = "rays_per_thread"

class NotSupportedException(RuntimeError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


parser = argparse.ArgumentParser(description='Cuspatial runner')
parser.add_argument('-s', type=int, help='Seed for numpy.random',required=False, default=None)
parser.add_argument("-d", type=str, help='Distribution', required=True,choices=[e.value for e in Distribution])
parser.add_argument("-b", type=str, help="Benchmark to run", required=True,choices=[e.value for e in Benchmark])
parser.add_argument("-p", type=str, help="Program to benchmark", required=True,choices=[e.value for e in Program])
parser.add_argument("--dry-run", help='Print commands that would be used to start a program but do not run program', default=False, action='store_true')

args = parser.parse_args()

SEED = np.random.randint(1_000_000) if args.s is None else args.s
BENCHMARK = Benchmark(args.b)
DIST = Distribution(args.d)
PROG = Program(args.p)
DRY_RUN = args.dry_run

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

def get_cuspatial_cmd(n = N):
    CUSPATIAL_CMD = ["python3", "cuspatial_runner.py", "-n", f"{n}"]
    return CUSPATIAL_CMD

def get_geo_rt_cmd(n = N, r = 1, l = 0, m = 1):
    GEO_RT_CMD = ["./build/release/geo-rt-index", "-n", f"{N}", "-r", f"{r}", "-l", f"{l}", "-m", f"{m}"]
    return GEO_RT_CMD

def get_session_str():
    return f"s{SEED}_b{BENCHMARK.value}_d{DIST.value}_p{PROG.value}_n{N}"

np.random.seed(SEED)
TIME = datetime.now(timezone.utc).isoformat()
SESSION_OUTPUT_DIR = os.path.join(OUTPUT_DATA_DIR, get_session_str(), TIME)
if not DRY_RUN: 
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

def mk_query_strings(queries: Iterable):
    return list(flatten(map(lambda t: [f"-q", f"{t[0]}", f"{t[1]}", f"{t[2]}", f"{t[3]}"], queries)))

LO = -1
HI = 1
QUERIES = mk_query_strings([
    *mk_queries(0.01, 5, LO, HI), 
    *mk_queries(0.02, 5, LO, HI),
    *mk_queries(0.05, 5, LO, HI),
    *mk_queries(0.10, 5, LO, HI),
    *mk_queries(0.20, 5, LO, HI)
])

match BENCHMARK:
    case Benchmark.DS_SCALING:
        datasets = np.random.choice(FILES, 8, False).tolist()
        file_count = 1
        while file_count <= len(datasets):
            try:
                files = datasets[:file_count]
                cmd = get_cuspatial_cmd() if PROG == Program.CUSPATIAL else get_geo_rt_cmd()
                local_cmd = cmd + QUERIES + files
                local_cmd_str = " ".join(local_cmd)
                file_count *= 2
                if DRY_RUN:
                    print(local_cmd_str)
                    continue
                prog_out= open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_prog.txt"), "x")
                # smi_out= open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_smi.txt"), "x")
                prog_out.write(local_cmd_str)
                prog_out.write('\n')
                prog_out.flush()
                # print("cmd:", local_cmd_str)
                prog_process = sp.Popen(local_cmd, stdout=prog_out, stderr=sys.stderr)
                # smi_process = sp.Popen(SMI_CMD, stdout=smi_out, stderr=sys.stderr)
                prog_process.wait()
                # smi_process.kill()
            finally:
                if not DRY_RUN:
                    prog_out.close()
                # smi_out.close()
    case Benchmark.QUERY_SCALING:
        raise NotImplementedError(Benchmark.QUERY_SCALING)    
    
    case Benchmark.DS_TIME_CHECK_EACH:
        # Check each data set has approximately the same run time
        try:
            prog_out = None
            if not DRY_RUN:
                prog_out=open(os.path.join(SESSION_OUTPUT_DIR, f"lo{LO}hi{HI}_prog.txt"), "x")
            # smi_out= open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_smi.txt"), "x")
            cmd = get_cuspatial_cmd() if PROG == Program.CUSPATIAL else get_geo_rt_cmd()
            for file in FILES:
                local_cmd = cmd + QUERIES + [file]
                local_cmd_str = " ".join(local_cmd)
                if DRY_RUN:
                    print(local_cmd_str)
                    continue
                prog_out.write(local_cmd_str)
                prog_out.write('\n')
                prog_out.flush()
                # print("cmd:", local_cmd_str)
                prog_process = sp.Popen(local_cmd, stdout=prog_out, stderr=sys.stderr)
                # smi_process = sp.Popen(SMI_CMD, stdout=smi_out, stderr=sys.stderr)
                prog_process.wait()
            # smi_process.kill()
        finally:
            if not DRY_RUN:
                prog_out.close()
            # smi_out.close()

    case Benchmark.AABB_LAYERING_SCALING:
        raise NotImplementedError(Benchmark.AABB_LAYERING_SCALING)
        if PROG == Program.CUSPATIAL:
            raise NotSupportedException(f"Program {PROG} does not support benchmark {BENCHMARK}")
        
        layering_types = [0, 1, 2]

        pass
    case Benchmark.RAYS_PER_THREAD_SCALING:
        raise NotImplementedError(Benchmark.RAYS_PER_THREAD_SCALING)
        if PROG == Program.CUSPATIAL:
            raise NotSupportedException(f"Program {PROG} does not support benchmark {BENCHMARK}")
        
        rp = [2 ** x for x in range(10)]
        pass
    case _: raise NotImplementedError(f"Unimplemented benchmark: {BENCHMARK.value}")

