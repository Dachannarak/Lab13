"""
parquet_vs_iceberg.py

Demonstrates the flow from the diagram:
  1. load  — read source data into Spark
  2. init  — write/initialize a parquet file (full snapshot)
  3. delete record — remove specific records from another parquet file
                     (read → filter → overwrite, the classic parquet pattern)

Requirements:
    pip install pyspark
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import os
import tempfile

# --------------------------------------------------------------------------- 
# 0. Create Spark session
# ---------------------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("parquet_vs_iceberg")
    .master("local[1]")  # Changed to local[1] to avoid Windows socket connection issues
    # Enable Hive support only if you have a metastore; remove otherwise
    # .enableHiveSupport()
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# --------------------------------------------------------------------------- 
# Paths  (adjusted to your environment / S3 / HDFS paths as needed)
# ---------------------------------------------------------------------------

BASE_DIR          = os.path.join(tempfile.gettempdir(), "parquet_demo")  # Changed to Windows temp dir for compatibility
SOURCE_PARQUET    = os.path.join(BASE_DIR, "source_data.parquet")   # step 2 target
MUTABLE_PARQUET   = os.path.join(BASE_DIR, "mutable_data.parquet")  # step 3 target

# --------------------------------------------------------------------------- 
# Helper: sample data
# ---------------------------------------------------------------------------

SCHEMA = StructType([
    StructField("id",         IntegerType(), False),
    StructField("name",       StringType(),  True),
    StructField("department", StringType(),  True),
    StructField("salary",     DoubleType(),  True),
])

SAMPLE_ROWS = [
    (1,  "Alice",   "Engineering", 95000.0),
    (2,  "Bob",     "Marketing",   72000.0),
    (3,  "Carol",   "Engineering", 88000.0),
    (4,  "Dave",    "HR",          65000.0),
    (5,  "Eve",     "Engineering", 102000.0),
    (6,  "Frank",   "Marketing",   71000.0),
    (7,  "Grace",   "HR",          67000.0),
    (8,  "Heidi",   "Engineering", 91000.0),
    (9,  "Ivan",    "Finance",     84000.0),
    (10, "Judy",    "Finance",     79000.0),
]

# IDs we want to delete in step 3
IDS_TO_DELETE = [2, 5, 9]


# =========================================================================== 
# STEP 1 — load: read source data into Spark (in-memory DataFrame)
# =========================================================================== 

def step1_load() -> "pyspark.sql.DataFrame":
    """
    In a real pipeline this would be:
        spark.read.csv(...)  / .json(...)  / .jdbc(...)  etc.
    Here we create a small in-memory dataset to keep the demo self-contained.
    """
    print("\n" + "="*60)
    print("STEP 1 — load: reading source data into Spark")
    print("="*60)

    df = spark.createDataFrame(SAMPLE_ROWS, schema=SCHEMA)

    print(f"  Loaded {df.count()} rows")
    df.show(truncate=False)
    return df


# =========================================================================== 
# STEP 2 — init: write the full dataset to a parquet file (SOURCE_PARQUET)
# =========================================================================== 

def step2_init(df: "pyspark.sql.DataFrame") -> None:
    """
    Write the DataFrame to parquet — this is the 'initialise' step.
    mode='overwrite' creates the file fresh each run.
    Use partitionBy() for large datasets to improve read performance.
    """
    print("\n" + "="*60)
    print(f"STEP 2 — init: writing full snapshot → {SOURCE_PARQUET}")
    print("="*60)

    (
        df.write
        .mode("overwrite")
        .partitionBy("department")          # optional but common practice
        .parquet(SOURCE_PARQUET)
    )

    # Verify
    written = spark.read.parquet(SOURCE_PARQUET)
    print(f"  Written {written.count()} rows  |  partitions: {written.rdd.getNumPartitions()}")
    written.show(truncate=False)

    # Also seed the mutable parquet (same data — will be mutated in step 3)
    (
        df.write
        .mode("overwrite")
        .parquet(MUTABLE_PARQUET)
    )
    print(f"  Seeded mutable parquet  → {MUTABLE_PARQUET}")


# =========================================================================== 
# STEP 3 — delete record: remove rows from MUTABLE_PARQUET
#
# Parquet is immutable — there is no in-place delete.
# The standard pattern is:  read → filter out unwanted rows → overwrite.
# =========================================================================== 

def step3_delete_records(ids_to_delete: list) -> None:
    """
    'Delete' rows from a parquet file by reading the whole file,
    filtering out the target IDs, and writing back.

    For large tables consider:
      - Delta Lake  (ACID deletes)
      - Apache Iceberg (row-level deletes / merge-on-read)
      - Apache Hudi   (upsert / delete support)
    """
    print("\n" + "="*60)
    print(f"STEP 3 — delete record: removing ids {ids_to_delete}")
    print(f"         target parquet → {MUTABLE_PARQUET}")
    print("="*60)

    # 3a. Read existing parquet
    df_existing = spark.read.parquet(MUTABLE_PARQUET)
    before_count = df_existing.count()
    print(f"  Before delete: {before_count} rows")

    # 3b. Filter out the rows to delete
    df_filtered = df_existing.filter(~F.col("id").isin(ids_to_delete))
    after_count = df_filtered.count()
    print(f"  After  delete: {after_count} rows  ({before_count - after_count} removed)")

    # 3c. Overwrite — write to a temp path, then rename to avoid reading
    #     from the same path we're writing to (can cause issues on some FS)
    tmp_path = MUTABLE_PARQUET + "_tmp"
    (
        df_filtered
        .coalesce(1)                        # single file for demo clarity
        .write
        .mode("overwrite")
        .parquet(tmp_path)
    )

    # Atomically replace original with updated version
    import shutil
    if os.path.exists(MUTABLE_PARQUET):
        shutil.rmtree(MUTABLE_PARQUET)
    os.rename(tmp_path, MUTABLE_PARQUET)

    # 3d. Verify
    df_final = spark.read.parquet(MUTABLE_PARQUET)
    print("\n  Final state of mutable parquet:")
    df_final.orderBy("id").show(truncate=False)


# =========================================================================== 
# BONUS — why Iceberg / Delta / Hudi?
# =========================================================================== 

def explain_limitations() -> None:
    print("\n" + "="*60)
    print("WHY plain Parquet has limits for mutable workloads")
    print("="*60)
    print("""
  Plain parquet limitations for updates/deletes
  ───────────────────────────────────────────────
  • No ACID transactions  → concurrent writers can corrupt data
  • No row-level deletes  → must rewrite the entire file/partition
  • No schema evolution   → adding columns needs careful migration
  • No time-travel        → can't query previous versions

  Table formats that solve this (while still storing data as parquet):
  ┌──────────────┬────────────────────────────────────────────────────┐
  │ Format       │ Key feature                                        │
  ├──────────────┼────────────────────────────────────────────────────┤
  │ Delta Lake   │ ACID, time-travel, Z-order clustering              │
  │ Apache Iceberg│ Row-level deletes, hidden partitioning, snapshots │
  │ Apache Hudi  │ Upsert/delete, incremental pulls, CDC support      │
  └──────────────┴────────────────────────────────────────────────────┘

  The diagram shows the core problem:
    • parquet (left)  = source / init  → append-friendly
    • parquet (right) = mutable copy   → requires full rewrite on delete
  """)


# =========================================================================== 
# Main
# =========================================================================== 

if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    # ── 1. load ──────────────────────────────────────────────────────────────
    raw_df = step1_load()

    # ── 2. init ──────────────────────────────────────────────────────────────
    step2_init(raw_df)

    # ── 3. delete record ─────────────────────────────────────────────────────
    step3_delete_records(IDS_TO_DELETE)

    # ── Bonus explanation ────────────────────────────────────────────────────
    explain_limitations()

    spark.stop()
    print("\nDone.")