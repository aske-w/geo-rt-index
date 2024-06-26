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

import partial_overlap_gen 

def get_system():
    import platform
    match platform.system():
        case "Linux":
            if platform.platform() == 'Linux-5.15.0-107-generic-x86_64-with-glibc2.35':
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
    QUERY_SCALING_SINGLE_QUERY = "query_scaling_single_query"
    AABB_LAYERING_SCALING = "aabb_layering"
    RAYS_PER_THREAD_SCALING = "rays_per_thread"
    POINT_SORTING = "point_sorting"
    MODIFIER = "modifier"
    COMPACTION = "compaction"
    RAY_LENGTH = "ray_length"
    AABB_Z_VALUE = "aabb_z_value"
    OVERLAP = "overlap"

class NotSupportedException(RuntimeError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


parser = argparse.ArgumentParser(description='Cuspatial runner')
parser.add_argument("-d", type=str, help='Distribution', required=True,choices=[e.value for e in Distribution])
parser.add_argument("-b", type=str, help="Benchmark to run", required=True,choices=[e.value for e in Benchmark])
parser.add_argument("-p", type=str, help="Program to benchmark", required=True,choices=[e.value for e in Program])
parser.add_argument("--lo", type=int, help="Low", required=True)
parser.add_argument("--hi", type=int, help="High", required=True)
parser.add_argument("--kernel-only", action='store_true', default=False)
# parser.add_argument("--file-stem-suffix", type=str, required=False, default="*")
parser.add_argument("--dry-run", help='Print commands that would be used to start a program but do not run program', default=False, action='store_true')

args = parser.parse_args()

BENCHMARK = Benchmark(args.b)
DIST = Distribution(args.d)
PROG = Program(args.p)
DRY_RUN = args.dry_run
LO = args.lo
HI = args.hi
PREFIX = f"*_r{LO}{HI}"
KERNEL_ONLY= args.kernel_only

N = 5
INPUT_DATA_DIR = os.path.join("/home/aske/dev/geo-rt-index/data" if get_system() == "ubuntu" else "/home/ucloud/geo-rt-index/data", DIST.value)
PARQUET_FILES = glob.glob(os.path.join(INPUT_DATA_DIR, f"{PREFIX}.parquet"))
assert(len(PARQUET_FILES) == 32)
PICKLE_FILES = None if PROG is Program.GEO_RT_INDEX else glob.glob(os.path.join(INPUT_DATA_DIR, f"{PREFIX}.xy.pickle"))
PARQUET_FILES.sort(key=lambda t: Path(t).stem) # file name without path or extensions
if PICKLE_FILES is not None:
    PICKLE_FILES.sort(key=lambda t: Path(t).stem)
    assert(len(PARQUET_FILES) == len(PICKLE_FILES))

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
    CUSPATIAL_CMD = [
        "python3", 
        "cuspatial_runner.py", 
        "-n", f"{n}",
        "--benchmark", f"{BENCHMARK.value}"
        ]
    return CUSPATIAL_CMD

def get_geo_rt_cmd(n = N, r = 1, l = 0, m = 1, sort=0, compaction=False, aabb_z_value=None, ray_length=None):
    geo_rt_cmd = [
        "./build/release/geo-rt-index", 
        "-n", f"{N}", 
        "-r", f"{r}", 
        "-l", f"{l}", 
        "-m", f"{m}", 
        "--sort", f"{sort}",
        "--benchmark", f"{BENCHMARK.value}"
    ]
    if compaction:
        geo_rt_cmd.append("--compaction")

    if aabb_z_value is not None:
        geo_rt_cmd += ["--aabb-z-value", f"{aabb_z_value}"]

    if ray_length is not None:
        geo_rt_cmd += ["--ray-length", f"{ray_length}"]

    if KERNEL_ONLY:
        geo_rt_cmd += ["--kernel-only"]

    return geo_rt_cmd

def get_session_str():
    return f"b{BENCHMARK.value}_d{DIST.value}_p{PROG.value}_n{N}_r{LO}{HI}"

def get_time():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

SESSION_OUTPUT_DIR = os.path.join(OUTPUT_DATA_DIR, get_session_str())
while os.path.exists(SESSION_OUTPUT_DIR):
    TIME = get_time()
    SESSION_OUTPUT_DIR = os.path.join(OUTPUT_DATA_DIR, get_session_str(), TIME)

if not DRY_RUN: 
    os.makedirs(SESSION_OUTPUT_DIR, exist_ok=True)

def mk_query_strings(queries: Iterable):
    def conv(t):
        return [f"-q", f"{t[0]}", f"{t[1]}", f"{t[2]}", f"{t[3]}"]
    return list(flatten(map(conv,queries)))

QUERIES = {
    0.01: None,
    0.02: None,
    0.05: None,
    0.10: None,
    0.20: None,
}

for s in [1,2,5,10,20]:
    with open(f"./queries/{DIST.value}/r{LO}{HI}/{s}.json") as fp:
        deserialized: list[dict] = json.load(fp)
        QUERIES[s/100] = [(bbox["minx"],bbox["miny"], bbox["maxx"], bbox["maxy"]) for bbox in deserialized]

BASELINE_QUERIES = mk_query_strings(QUERIES[0.01][:4] + QUERIES[0.02][:4] + QUERIES[0.05][:4] + QUERIES[0.10][:4] + QUERIES[0.20][:4])
_files = PICKLE_FILES if PROG == Program.CUSPATIAL and PICKLE_FILES is not None else PARQUET_FILES
BASELINE_FILES = _files[:2]
assert(len(BASELINE_FILES) == 2)
assert(len(BASELINE_QUERIES) == 5 * 4 * 5)

match BENCHMARK:
    case Benchmark.DS_SCALING:
        counts = [1,2,4,6] if PROG == Program.CUSPATIAL else [1,2,4,8,16,32]

        try:
            prog_out = None if DRY_RUN else open(os.path.join(SESSION_OUTPUT_DIR, f"dataset_scaling_prog.txt" if not KERNEL_ONLY else "dataset_scaling_kernel.txt"), "a")
            for file_count in counts:
                cmd = get_cuspatial_cmd() if PROG == Program.CUSPATIAL else get_geo_rt_cmd()
                files = _files[:file_count]
                assert(len(files) == file_count)
                local_cmd = cmd  + ["--id", uuid.uuid4().hex] + BASELINE_QUERIES + files
                local_cmd_str = " ".join(local_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue
                prog_out.write(f"{get_time()}\n")
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

        SCALE_LOG = 7
        prog_out = None
        try:
            if not DRY_RUN:
                prog_out = open(os.path.join(SESSION_OUTPUT_DIR, f"query_scaling_prog.txt" if not KERNEL_ONLY else "query_scaling_kernel.txt"), "x")

            for limit in map(lambda power: 2**power, range(SCALE_LOG)):
                # while len(queries) < limit:
                queries = mk_query_strings(QUERIES[0.01][:limit] + QUERIES[0.02][:limit] + QUERIES[0.05][:limit] + QUERIES[0.10][:limit] + QUERIES[0.20][:limit])
                assert(len(queries) == limit * 5 * 5)
                query_scaling_cmd = get_geo_rt_cmd() if PROG == Program.GEO_RT_INDEX else get_cuspatial_cmd()  
                query_scaling_cmd += ["--id", uuid.uuid4().hex] + queries + BASELINE_FILES
                local_cmd_str = " ".join(query_scaling_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue # skip to next log
                prog_out.write(f"{get_time()}\n")
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

    case Benchmark.QUERY_SCALING_SINGLE_QUERY:
        if PROG != Program.GEO_RT_INDEX:
            raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")
        SCALE_LOG = 7
        prog_out = None
        for s in [0.01, 0.02, 0.05, 0.10, 0.20]:
            try:
                if not DRY_RUN:
                    prog_out = open(os.path.join(SESSION_OUTPUT_DIR, f"query_scaling_single_s{s}_prog.txt"), "x")

                for limit in map(lambda power: (2**power) * 5, range(SCALE_LOG)):
                    # while len(queries) < limit:
                    _queries = mk_query_strings(QUERIES[s][:1])
                    queries = []
                    for _ in range(limit):
                        queries += _queries
                    assert(len(queries) == limit * 5)
                    query_scaling_cmd = get_geo_rt_cmd()
                    query_scaling_cmd += ["--id", uuid.uuid4().hex] + queries + BASELINE_FILES
                    local_cmd_str = " ".join(query_scaling_cmd)
                    print(local_cmd_str)
                    if DRY_RUN:
                        continue # skip to next log
                    prog_out.write(f"{get_time()}\n")
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
                local_cmd = cmd  + ["--id", uuid.uuid4().hex] + BASELINE_QUERIES + [file]
                local_cmd_str = " ".join(local_cmd)
                if DRY_RUN:
                    print(local_cmd_str)
                    continue
                prog_out.write(f"{get_time()}\n")
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
        
        CMD_SUFFIX = BASELINE_QUERIES + BASELINE_FILES
        prog_out = None
        try:
            if not DRY_RUN:
                PATH = os.path.join(SESSION_OUTPUT_DIR, f"aabb_layering_prog.txt")
                prog_out = open(PATH, "x")
            for layer_type in LAYERING_TYPES:

                query_scaling_cmd = get_geo_rt_cmd(l=layer_type) + ["--id", uuid.uuid4().hex] + CMD_SUFFIX
                local_cmd_str = " ".join(query_scaling_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue # skip to next log
                prog_out.write(f"{get_time()}\n")
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

        prog_out = None
        if not DRY_RUN:
            PATH = os.path.join(SESSION_OUTPUT_DIR, f"rays_per_thread_prog.txt")
            prog_out = open(PATH, "x")

        try:
            files = BASELINE_FILES
            CMD_SUFFIX = BASELINE_QUERIES + files
            for rays_per_thread in rays_per_threads:
                query_scaling_cmd = get_geo_rt_cmd(r=rays_per_thread) + ["--id", uuid.uuid4().hex] + CMD_SUFFIX
                local_cmd_str = " ".join(query_scaling_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue # skip to next log
                prog_out.write(f"{get_time()}\n")
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

    case Benchmark.POINT_SORTING:
        if PROG != Program.GEO_RT_INDEX:
            raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")

        sorts = [
            0,  # none
            1,  # X
            2,  # Y
            3   # Z order
        ]

        CMD_SUFFIX = BASELINE_QUERIES + BASELINE_FILES
        for sort_type in sorts:
            prog_out = None
            try:
                if not DRY_RUN:
                    PATH = os.path.join(SESSION_OUTPUT_DIR, f"sort_type_type{sort_type}_prog.txt")
                    prog_out = open(PATH, "x")

                query_scaling_cmd = get_geo_rt_cmd(sort=sort_type) + ["--id", uuid.uuid4().hex] + CMD_SUFFIX
                local_cmd_str = " ".join(query_scaling_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue  # skip to next log
                prog_out.write(f"{get_time()}\n")
                prog_out.write(f"Running with layer type {sort_type} queries\n")
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


    case Benchmark.MODIFIER:
        if PROG != Program.GEO_RT_INDEX:
            raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")

        modifiers = [
            1e-10,
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            1e1,
            1e2,
            1e3,
            1e4,
            1e5,
            1e6,
            1e7,
            1e8,
            1e9,
        ]

        CMD_SUFFIX = BASELINE_QUERIES + BASELINE_FILES
        prog_out = None
        try:
            if not DRY_RUN:
                PATH = os.path.join(SESSION_OUTPUT_DIR, f"modifier_prog.txt")
                prog_out = open(PATH, "x")
            for modifier in modifiers:

                query_scaling_cmd = get_geo_rt_cmd(m=modifier) + ["--id", uuid.uuid4().hex] + CMD_SUFFIX
                local_cmd_str = " ".join(query_scaling_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue  # skip to next log
                prog_out.write(f"{get_time()}\n")
                prog_out.write(f"Running with modifier value {modifier}\n")
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

    case Benchmark.COMPACTION:
        pass


    case Benchmark.RAY_LENGTH:
        
        if PROG != Program.GEO_RT_INDEX:
            raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")

        lengths = [
            # 1e-10,
            # 1e-9,
            # 1e-8,
            # 1e-7,
            # 1e-6,
            # 1e-5,
            # 1e-4,
            1e-3,
            1e-2,
            1e-1,
            1,
            1e1,
            1e2,
            1e3,
            # 1e4,
            # 1e5,
            # 1e6,
            # 1e7,
            # 1e8,
            # 1e9,
        ]

        CMD_SUFFIX = BASELINE_QUERIES + BASELINE_FILES
        prog_out = None
        try:
            if not DRY_RUN:
                PATH = os.path.join(SESSION_OUTPUT_DIR, f"aabb_z_value_prog.txt")
                prog_out = open(PATH, "x")
            for ray_length in lengths:

                ray_length_cmd = get_geo_rt_cmd(ray_length=ray_length) + ["--id", uuid.uuid4().hex] + CMD_SUFFIX
                local_cmd_str = " ".join(ray_length_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue  # skip to next log
                prog_out.write(f"{get_time()}\n")
                prog_out.write(f"Running with ray_length {ray_length}\n")
                prog_out.write(f"{local_cmd_str}\n")
                prog_out.flush()
                prog_process = sp.Popen(ray_length_cmd, stdout=prog_out, stderr=prog_out)
                prog_process.wait()
                # assert (prog_process.wait() == 0)
                prog_out.flush()
        finally:
            if not DRY_RUN:
                assert prog_out is not None
                prog_out.flush()
                prog_out.close()


    case Benchmark.AABB_Z_VALUE:
        if PROG != Program.GEO_RT_INDEX:
            raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")

        zs = [
            # 1e-10,
            # 1e-9,
            # 1e-8,
            # 1e-7,
            # 1e-6,
            # 1e-5,
            # 1e-4,
            # 1e-3,
            # 1e-2,
            # 1e-1,
            # 1,
            1e1 - 1,
            1e2 - 1,
            1e3 - 1,
            1e4 - 1,
            1e5 - 1,
            1e6 - 1,
            1e7 - 1,
            1e8 - 1,
            1e9 - 1,
        ]

        CMD_SUFFIX = BASELINE_QUERIES + BASELINE_FILES
        prog_out = None
        try:
            if not DRY_RUN:
                PATH = os.path.join(SESSION_OUTPUT_DIR, f"aabb_z_value_prog.txt")
                prog_out = open(PATH, "x")
            for z in zs:

                ray_length_cmd = get_geo_rt_cmd(aabb_z_value=z) + ["--id", uuid.uuid4().hex] + CMD_SUFFIX
                local_cmd_str = " ".join(ray_length_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue  # skip to next log
                prog_out.write(f"{get_time()}\n")
                prog_out.write(f"Running with aabb z value {z}\n")
                prog_out.write(f"{local_cmd_str}\n")
                prog_out.flush()
                prog_process = sp.Popen(ray_length_cmd, stdout=prog_out, stderr=prog_out)
                assert (prog_process.wait() == 0)
                prog_out.flush()
        finally:
            if not DRY_RUN:
                assert prog_out is not None
                prog_out.flush()
                prog_out.close()
        
    case Benchmark.OVERLAP:
        if PROG != Program.GEO_RT_INDEX:
            raise NotSupportedException(f"{Benchmark.QUERY_SCALING} not supported with program {PROG}")
        if LO != 0 and HI != 1:
            raise Exception(f"LO must be 0 and HI must be 1, but were {LO} and {HI}")
        if DIST != Distribution.UNIFORM:
            raise Exception("Distribution must be uniform")

        overlaps = [
            0.50,
            0.25,
            0.10,
            0.05,
            0.025,
            0.0125,
            0.00625,
            0.0
        ]

        prog_out = None
        try:
            if not DRY_RUN:
                PATH = os.path.join(SESSION_OUTPUT_DIR, f"overlap.txt")
                prog_out = open(PATH, "x")

            for overlap in overlaps:
                queries = list(flatten(map(lambda bbox: bbox.to_query(), partial_overlap_gen.get(overlap, 20))))
                ray_length_cmd = get_geo_rt_cmd() + ["--id", uuid.uuid4().hex] + queries + BASELINE_FILES
                local_cmd_str = " ".join(ray_length_cmd)
                print(local_cmd_str)
                if DRY_RUN:
                    continue  # skip to next log
                prog_out.write(f"{get_time()}\n")
                prog_out.write(f"Running with overlap {overlap}\n")
                prog_out.write(f"{local_cmd_str}\n")
                prog_out.flush()
                prog_process = sp.Popen(ray_length_cmd, stdout=prog_out, stderr=prog_out)
                assert (prog_process.wait() == 0)
                prog_out.flush()
        finally:
            if not DRY_RUN:
                assert prog_out is not None
                prog_out.flush()
                prog_out.close()

    case _: raise NotImplementedError(f"Unimplemented benchmark: {BENCHMARK.value}")

