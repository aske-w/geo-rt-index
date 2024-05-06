import duckdb
import glob
import json
import numpy as np
import tqdm
from pytypes import BBox

duckdb.connect(":memory:", config={"allow_unsigned_extensions": True})
duckdb.install_extension("spatial")
duckdb.load_extension("spatial")

LO = -1
HI = 1

def mk_query(minx, miny, maxx, maxy,file):
    return f"""select (
        select count(*) 
        from '{file}' 
        where st_x(st_geomfromwkb(geometry)) 
        between {minx} and {maxx}
        and st_y(st_geomfromwkb(geometry)) between {miny} and {maxy}
    ) / count(*) * 100 as hit_rate from '{file}';"""

query_files = {
    1: f"./data/queries/normal/r{LO}{HI}/1.json",
    2: f"./data/queries/normal/r{LO}{HI}/2.json",
    5: f"./data/queries/normal/r{LO}{HI}/5.json",
    10: f"./data/queries/normal/r{LO}{HI}/10.json",
    20: f"./data/queries/normal/r{LO}{HI}/20.json"
}
data_files = f"/Volumes/Untitled/data/normal/*_r{LO}{HI}.parquet"


# print(duckdb.sql(f"create table data as select st_geomfromwkb(geometry) as geom from '{data_files}'"))
for df in glob.glob(data_files):
    for s, f in query_files.items():
        with open(f, "r") as fp:
            deserialized:list = json.load(fp)
            results = []
            for bbox in tqdm.tqdm(np.random.choice(deserialized, 32, False)):
                # print(bbox)
                # print("a")
                result = duckdb.sql(mk_query(bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"], df))
                print(result)
                # print("b")
                # results.append(result.to_df()["hit_rate"].iloc[0])

            # print("data_file", data_file, "s", s, "f", f)
            # print(np.max(results), np.min(results), np.average(results))

    # duckdb.sql("drop table data")
