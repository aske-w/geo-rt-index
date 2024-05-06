from datetime import datetime, timezone
from enum import Enum
import glob
import json
import os
import argparse
import sys
from typing import Iterable
from more_itertools import flatten
import numpy as np
import subprocess as sp
from pathlib import Path
import uuid

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
parser.add_argument("--lo", type=int, help="Low", required=True)
parser.add_argument("--hi", type=int, help="High", required=True)
parser.add_argument("--file-stem-suffix", type=str, required=False, default="*")
parser.add_argument("--dry-run", help='Print commands that would be used to start a program but do not run program', default=False, action='store_true')

args = parser.parse_args()

SEED = np.random.randint(1_000_000) if args.s is None else args.s
print("Using seed ", SEED)
np.random.seed(SEED)
BENCHMARK = Benchmark(args.b)
DIST = Distribution(args.d)
PROG = Program(args.p)
DRY_RUN = args.dry_run
SUFFIX = args.file_stem_suffix
LO = args.lo
HI = args.hi

N = 5
INPUT_DATA_DIR = os.path.join("/home/aske/dev/geo-rt-index/data" if get_system() == "ubuntu" else "/home/ucloud/geo-rt-index/data", DIST.value)
PARQUET_FILES = glob.glob(os.path.join(INPUT_DATA_DIR, f"{SUFFIX}.parquet"))
PICKLE_FILES = None if PROG is Program.GEO_RT_INDEX else glob.glob(os.path.join(INPUT_DATA_DIR, f"{SUFFIX}.xy.pickle"))
PARQUET_FILES.sort(key=lambda t: Path(t).stem) # file name without path or extensions
if PICKLE_FILES is not None:
    PICKLE_FILES.sort(key=lambda t: Path(t).stem)

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
    return f"b{BENCHMARK.value}_d{DIST.value}_p{PROG.value}_n{N}_s{SEED}"

SESSION_OUTPUT_DIR = os.path.join(OUTPUT_DATA_DIR, get_session_str())
while os.path.exists(SESSION_OUTPUT_DIR):
    TIME = datetime.now(timezone.utc).isoformat()
    SESSION_OUTPUT_DIR = os.path.join(OUTPUT_DATA_DIR, get_session_str(), TIME)

if not DRY_RUN: 
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)

# def mk_query(selectivity: float, lo = 0, hi = 1):
#     # raise NotImplementedError()
#     assert(0 < selectivity and selectivity <= 1)
#     match DIST:
#         case Distribution.UNIFORM:
#             LIMIT = 1000
#             # used ChatGPT for this loop
#             for _ in range(LIMIT):
#                 area = selectivity
#                 # Generate random dimensions
#                 width = np.random.uniform(1, area)  # Limit the width to be less than or equal to the area
#                 height = area / width
                
#                 # Check if the dimensions produce the desired area
#                 if abs(width * height - area) < 1e-9:
#                     # return width, height
#                     x1 = np.random.uniform(lo, hi - width)
#                     y1 = np.random.uniform(lo, hi - height)
#                     x2 = x1 + width
#                     y2 = y1 + height
#                     return x1, y1, x2, y2 # minx, miny, maxx maxy semantics
#             raise Exception("Limit reached generating query for uniform distribution")
#         case _: raise NotImplementedError(DIST.value)

# def mk_queries(selectivity: float, n: int, lo = 0, hi = 1):
#     for _ in range(n):
#         yield mk_query(selectivity, lo, hi)

def mk_query_strings(queries: Iterable):
    return list(flatten(map(lambda t: [f"-q", f"{t[0]}", f"{t[1]}", f"{t[2]}", f"{t[3]}"], queries)))

# QUERIES = mk_query_strings([
#     *mk_queries(0.01, 5, LO, HI),
#     *mk_queries(0.02, 5, LO, HI),
#     *mk_queries(0.05, 5, LO, HI),
#     *mk_queries(0.10, 5, LO, HI),
#     *mk_queries(0.20, 5, LO, HI)
# ])

QUERIES = {
    0.01: None,
    0.02: None,
    0.05: None,
    0.10: None,
    0.20: None,
}

for s in [1,2,5,10,20]:
    with open(f"./data/queries/{DIST}/{s}.json") as fp:
        deserialized: list[dict] = json.load(fp)
        QUERIES[s/100] = [(bbox["minx"],bbox["miny"], bbox["maxx"], bbox["maxy"]) for bbox in deserialized]

_files = PICKLE_FILES if PROG == Program.CUSPATIAL and PICKLE_FILES is not None else PARQUET_FILES

match BENCHMARK:
    case Benchmark.DS_SCALING:
        counts = [1,2,4,6] if PROG == Program.CUSPATIAL else [1,2,4,6,8,16,32]
        # shuffle_mapping = [x for x in range(len(_files))]
        # np.random.shuffle(shuffle_mapping)
        # shuffled_files = []
        # for mapping in shuffle_mapping:
        #     shuffled_files.append(_files[mapping])

        queries = mk_query_strings(QUERIES[0.01][:4] + QUERIES[0.02][:4] + QUERIES[0.05][:4] + QUERIES[0.10][:4] + QUERIES[0.20][:4])

        for file_count in counts:
            try:
                prog_out = None if DRY_RUN else open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_prog.txt"), "a")
                cmd = get_cuspatial_cmd() if PROG == Program.CUSPATIAL else get_geo_rt_cmd()
                files = _files[:file_count]
                local_cmd = cmd  + ["--id", uuid.uuid4().hex] + queries + files
                local_cmd_str = " ".join(local_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue
                # smi_out= open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_smi.txt"), "x")
                prog_out.write(local_cmd_str)
                prog_out.write('\n')
                prog_out.flush()
                # print("cmd:", local_cmd_str)
                prog_process = sp.Popen(local_cmd, stdout=prog_out, stderr=prog_out)
                # smi_process = sp.Popen(SMI_CMD, stdout=smi_out, stderr=prog_out)
                assert (prog_process.wait() == 0)
                prog_out.flush()
                # smi_process.kill()
            finally:
                if not DRY_RUN:
                    assert(prog_out is not None)
                    prog_out.flush()
                    prog_out.close()
                    # smi_out.close()
    case Benchmark.QUERY_SCALING:
        # if PROG != Program.GEO_RT_INDEX:
        #     raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")

        FIXED_FILES = np.random.choice(_files, 1).tolist()[0]
        SCALE_LOG = 8
        for selectivity in [0.01, 0.02, 0.05, 0.10, 0.20]:
            prog_out = None
            try:
                if not DRY_RUN:
                    prog_out = open(os.path.join(SESSION_OUTPUT_DIR, f"query_scaling_selectivity{selectivity}_prog.txt"), "x")

                for limit in map(lambda power: 2**power, range(SCALE_LOG)):
                    # while len(queries) < limit:
                    queries = mk_query_strings(QUERIES[0.01][:limit] + QUERIES[0.02][:limit] + QUERIES[0.05][:limit] + QUERIES[0.10][:limit] + QUERIES[0.20][:limit])

                    query_scaling_cmd = get_geo_rt_cmd()  + ["--id", uuid.uuid4().hex] + mk_query_strings(queries) + [FIXED_FILES]
                    local_cmd_str = " ".join(query_scaling_cmd)
                    print(local_cmd_str)
                    if DRY_RUN:
                        continue # skip to next log
                    prog_out.write(f"Running with {limit} queries\n")
                    prog_out.write(f"{local_cmd_str}\n")
                    prog_out.flush()
                    prog_process = sp.Popen(query_scaling_cmd, stdout=prog_out, stderr=prog_out)
                    assert (prog_process.wait() == 0)
                    prog_out.flush()
            finally:
                if not DRY_RUN:
                    prog_out.flush()
                    prog_out.close()



    case Benchmark.DS_TIME_CHECK_EACH:
        # Check each data set has approximately the same run time
        try:
            prog_out = None
            if not DRY_RUN:
                prog_out=open(os.path.join(SESSION_OUTPUT_DIR, f"lo{LO}hi{HI}_prog.txt"), "x")
            # smi_out= open(os.path.join(SESSION_OUTPUT_DIR, f"fc{file_count}_smi.txt"), "x")
            cmd = get_cuspatial_cmd() if PROG == Program.CUSPATIAL else get_geo_rt_cmd()
            for file in _files:
                local_cmd = cmd  + ["--id", uuid.uuid4().hex] + mk_query_strings(QUERIES[0.01][:4] + QUERIES[0.02][:4] + QUERIES[0.05][:4] + QUERIES[0.10][:4] + QUERIES[0.20][:4]) + [file]
                local_cmd_str = " ".join(local_cmd)
                if DRY_RUN:
                    print(local_cmd_str)
                    continue
                prog_out.write(local_cmd_str)
                prog_out.write('\n')
                prog_out.flush()
                # print("cmd:", local_cmd_str)
                prog_process = sp.Popen(local_cmd, stdout=prog_out, stderr=prog_out)
                # smi_process = sp.Popen(SMI_CMD, stdout=smi_out, stderr=prog_out)
                assert (prog_process.wait() == 0)
                prog_out.flush()
            # smi_process.kill()
        finally:
            if not DRY_RUN:
                assert(prog_out is not None)
                prog_out.flush()
                prog_out.close()
            # smi_out.close()

    case Benchmark.AABB_LAYERING_SCALING:
        if PROG != Program.GEO_RT_INDEX:
            raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")

        LAYERING_TYPES = [0, 1, 2]
        FIXED_FILES = np.random.choice(_files, 1).tolist()[0]
        SELECTIVITY = 0.20

        SCALE_LOG = 10 # 512
        fixed_queries = mk_query_strings(QUERIES[0.01][:4] + QUERIES[0.02][:4] + QUERIES[0.05][:4] + QUERIES[0.10][:4] + QUERIES[0.20][:4])
        
        CMD_SUFFIX = mk_query_strings(fixed_queries) + [FIXED_FILES]
        for layer_type in LAYERING_TYPES:
            prog_out = None
            try:
                if not DRY_RUN:
                    PATH = os.path.join(SESSION_OUTPUT_DIR, f"aabb_layering_type{layer_type}_prog.txt")
                    prog_out = open(PATH, "x")

                query_scaling_cmd = get_geo_rt_cmd(l=layer_type) + ["--id", uuid.uuid4().hex] + CMD_SUFFIX
                local_cmd_str = " ".join(query_scaling_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue # skip to next log
                prog_out.write(f"Running with layer type {layer_type} queries\n")
                prog_out.write(f"{local_cmd_str}\n")
                prog_out.flush()
                prog_process = sp.Popen(query_scaling_cmd, stdout=prog_out, stderr=prog_out)
                assert (prog_process.wait() == 0)
                prog_out.flush()
            finally:
                if not DRY_RUN:
                    assert prog_out is not None
                    prog_out.flush()
                    prog_out.close()

    case Benchmark.RAYS_PER_THREAD_SCALING:
        if PROG != Program.GEO_RT_INDEX:
            raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")
        
        rays_per_threads = [2 ** x for x in range(10)]
        fc = [1,2,4]
        FIXED_FILES = np.random.choice(_files, fc[-1], False)
        np.random.shuffle(FIXED_FILES)
        SELECTIVITY = 0.20

        NUM_QUERIES = 64 # 512
        fixed_queries = queries = mk_query_strings(QUERIES[0.01][:4] + QUERIES[0.02][:4] + QUERIES[0.05][:4] + QUERIES[0.10][:4] + QUERIES[0.20][:4])

        for count in fc:
            files = FIXED_FILES[:count].tolist()
            CMD_SUFFIX = mk_query_strings(fixed_queries) + files
            prog_out = None
            try:
                if not DRY_RUN:
                    PATH = os.path.join(SESSION_OUTPUT_DIR, f"rays_per_thread_numfiles{count}_prog.txt")
                    prog_out = open(PATH, "x")
                for rays_per_thread in rays_per_threads:
                    query_scaling_cmd = get_geo_rt_cmd(r=rays_per_thread) + ["--id", uuid.uuid4().hex] + CMD_SUFFIX
                    local_cmd_str = " ".join(query_scaling_cmd)
                    print(local_cmd_str)
                    if DRY_RUN:
                        continue # skip to next log
                    prog_out.write(f"Running with {rays_per_thread} rays per thread\n")
                    prog_out.write(f"{local_cmd_str}\n")
                    prog_out.flush()
                    prog_process = sp.Popen(query_scaling_cmd, stdout=prog_out, stderr=prog_out)
                    assert (prog_process.wait() == 0)
                    prog_out.flush()
            finally:
                if not DRY_RUN:
                    assert prog_out is not None
                    prog_out.flush()
                    prog_out.close()


    case _: raise NotImplementedError(f"Unimplemented benchmark: {BENCHMARK.value}")

