"""
basic_iceberg.py
===================
Basic PySpark + Apache Iceberg operations:
  - Create a table
  - Insert / append data
  - Read data
  - Upsert (MERGE INTO)
  - List snapshots
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from pyspark.sql.functions import current_date, lit
import datetime
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ---------------------------------------------------------------------------
# 1. Create SparkSession with Iceberg catalog
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Iceberg Basic Demo")
    .config("spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.local.type", "hadoop")
    .config("spark.sql.catalog.local.warehouse",
            "file:///C:/Users/PC-1802/Desktop/Lab13/warehouse")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("SparkSession created with Iceberg catalog\n")
# ---------------------------------------------------------------------------
# 2. Dataset: E-commerce orders
# ---------------------------------------------------------------------------
schema = StructType([
    StructField("order_id",    IntegerType(), False),
    StructField("customer_id", StringType(),  False),
    StructField("product",     StringType(),  True),
    StructField("quantity",    IntegerType(), True),
    StructField("price",       DoubleType(),  True),
    StructField("status",      StringType(),  True),
    StructField("order_date",  StringType(),  True),   # kept as string for simplicity
])

initial_orders = [
    (1001, "C001", "Laptop",     1, 999.99,  "pending",   "2024-01-10"),
    (1002, "C002", "Headphones", 2, 149.99,  "shipped",   "2024-01-11"),
    (1003, "C003", "Keyboard",   1,  79.99,  "delivered", "2024-01-12"),
    (1004, "C001", "Mouse",      1,  29.99,  "pending",   "2024-01-13"),
    (1005, "C004", "Monitor",    2, 399.99,  "processing","2024-01-14"),
]

df_orders = spark.createDataFrame(initial_orders, schema)

# ---------------------------------------------------------------------------
# 3. Create Iceberg table & write initial data
# ---------------------------------------------------------------------------
# Drop if re-running
spark.sql("DROP TABLE IF EXISTS local.ecommerce.orders")
spark.sql("CREATE NAMESPACE IF NOT EXISTS local.ecommerce")

(
    df_orders.writeTo("local.ecommerce.orders")
    .partitionedBy("status")          # Iceberg hidden partitioning — no need to add a column
    .tableProperty("write.format.default", "parquet")   # underlying file format
    .create()
)

print("✅ Iceberg table created (snapshot #1 — initial load)")
spark.sql("SELECT * FROM local.ecommerce.orders ORDER BY order_id").show()

# ---------------------------------------------------------------------------
# 4. Append new rows (creates a new snapshot automatically)
# ---------------------------------------------------------------------------
new_orders = [
    (1006, "C005", "Webcam",  1,  89.99, "pending",  "2024-01-15"),
    (1007, "C002", "USB Hub", 3,  24.99, "shipped",  "2024-01-15"),
]
df_new = spark.createDataFrame(new_orders, schema)
df_new.writeTo("local.ecommerce.orders").append()

print("✅ Appended 2 new orders (snapshot #2)")
print(f"   Total rows now: {spark.table('local.ecommerce.orders').count()}\n")

# ---------------------------------------------------------------------------
# 5. Row-level UPDATE via MERGE INTO  (impossible with plain Parquet!)
# ---------------------------------------------------------------------------
#   Update orders 1001 and 1004 from 'pending' → 'shipped'
spark.createDataFrame(
    [(1001, "shipped"), (1004, "shipped")],
    ["order_id", "new_status"]
).createOrReplaceTempView("status_updates")

spark.sql("""
    MERGE INTO local.ecommerce.orders AS t
    USING status_updates             AS s
    ON t.order_id = s.order_id
    WHEN MATCHED THEN
        UPDATE SET t.status = s.new_status
""")

print("✅ MERGE INTO completed — orders 1001 & 1004 updated to 'shipped' (snapshot #3)")
spark.sql("""
    SELECT order_id, customer_id, product, status
    FROM   local.ecommerce.orders
    WHERE  order_id IN (1001, 1004)
""").show()

# ---------------------------------------------------------------------------
# 6. List all snapshots
# ---------------------------------------------------------------------------
print("📸 Snapshot history:")
spark.sql("SELECT snapshot_id, committed_at, operation FROM local.ecommerce.orders.snapshots").show(truncate=False)

print("Done! See 02_time_travel.py to query old snapshots.\n")
spark.stop()