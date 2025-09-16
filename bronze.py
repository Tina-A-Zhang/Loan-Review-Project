import pandas as pd
import hashlib
import os

def bronze_layer_transformation_chunked(input_file_path, output_file_path, chunk_size=50000):
    """
    Processes a large CSV file in chunks, adding a unique hash ID to each record
    and saves the output to a Parquet file.
    """
    STABLE_ID_COLS = [
        "loan_amnt", "funded_amnt", "int_rate", "installment", "grade",
        "home_ownership", "annual_inc", "dti", "purpose", "addr_state",
    ]

    import pyarrow as pa
    import pyarrow.parquet as pq
    from pandas.api.types import is_integer_dtype, is_float_dtype, is_bool_dtype

    writer = None
    first_columns = None
    first_schema = None
    first_dtypes = None
    try:
        for chunk in pd.read_csv(
            input_file_path,
            chunksize=chunk_size,
            low_memory=False,
            dtype={'desc': 'string'}  # 已知混合列先固定；若后续报新列，再补这里
        ):
            # 生成稳定 ID
            chunk['combined_key'] = chunk[STABLE_ID_COLS].astype(str).sum(axis=1)
            chunk['loan_application_id'] = chunk['combined_key'].apply(
                lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()
            )
            chunk.drop('combined_key', axis=1, inplace=True)

            # 首块：锁定列顺序与列类型
            if first_columns is None:
                first_columns = list(chunk.columns)
                first_dtypes = {c: chunk[c].dtype for c in first_columns}

            # 对齐列集合
            for col in first_columns:
                if col not in chunk.columns:
                    chunk[col] = pd.NA
            # 删除多余列并重排
            chunk = chunk[first_columns]

            # 将本分块的列类型对齐到首块的类型
            for col in first_columns:
                dt = first_dtypes[col]
                if is_integer_dtype(dt):
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('Int64')
                elif is_float_dtype(dt):
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                elif is_bool_dtype(dt):
                    chunk[col] = chunk[col].astype('boolean')
                else:
                    # 其余按字符串处理，避免 object/分类混淆
                    chunk[col] = chunk[col].astype('string')

            # 写入（统一 schema）
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if first_schema is None:
                first_schema = table.schema
                writer = pq.ParquetWriter(output_file_path, first_schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
