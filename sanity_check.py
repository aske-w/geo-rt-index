import duckdb
import glob
import json
import numpy as np
import tqdm
from pytypes import BBox

duckdb.connect(":memory:", config={"allow_unsigned_extensions": True})
duckdb.install_extension("spatial")
duckdb.load_extension("spatial")

def mk_query(minx, miny, maxx, maxy,file):
    return f"""select (
        select count(*) 
        from '{file}' 
        where st_x(st_geomfromwkb(geometry)) 
        between {minx} and {maxx}
        and st_y(st_geomfromwkb(geometry)) between {miny} and {maxy}
    ) / count(*) * 100 as hit_rate from '{file}';"""

query_files = {
    1: "./data/queries/normal/1.json",
    2: "./data/queries/normal/2.json",
    5: "./data/queries/normal/5.json",
    10: "./data/queries/normal/10.json",
    20: "./data/queries/normal/20.json"
}
data_files = "/Volumes/Untitled/data/normal/*_r01.parquet"


# print(duckdb.sql(f"create table data as select st_geomfromwkb(geometry) as geom from '{data_files}'"))
for s, f in query_files.items():
    with open(f, "r") as fp:
        deserialized:list = json.load(fp)
        results = []
        for bbox in tqdm.tqdm(deserialized):
            # print(bbox)
            # print("a")
            result = duckdb.sql(mk_query(bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"], data_files))
            print(result)
            # print("b")
            # results.append(result.to_df()["hit_rate"].iloc[0])

        # print("data_file", data_file, "s", s, "f", f)
        # print(np.max(results), np.min(results), np.average(results))

# duckdb.sql("drop table data")
