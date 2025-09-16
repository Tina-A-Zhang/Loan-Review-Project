import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib # to save the transformers for later use

def gold_layer_transformation_chunked(input_file_path, output_file_path, chunk_size=50000):
    """
    Processes the Silver layer data in chunks, performs feature engineering,
    and saves the final features to a new Parquet file.
    
    Args:
        input_file_path (str): The path to the Silver layer Parquet file.
        output_file_path (str): The path where the output Parquet file will be saved.
        chunk_size (int): The number of rows to process at a time.
    """
    
    # Define a list of features to process
    categorical_features = ["purpose", "home_ownership_norm", "verification_status"]
    numeric_features = [
        "loan_amnt", "installment", "annual_inc", "fico_range_high", 
        "fico_range_low", "int_rate", "dti", "open_acc", "pub_rec", 
        "revol_bal", "revol_util", "total_acc", "sub_grade_ord", 
        "term_months", "emp_length_months", "credit_age_months", 
        "issue_year", "issue_month",
    ]

    # Initialize transformers
    imputer = SimpleImputer(strategy='median')

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    import pyarrow as pa
    import pyarrow.parquet as pq

    header_written = False
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 先用第一批数据拟合转换器
    print("Fitting transformers on first chunk...")
    pf = pq.ParquetFile(input_file_path)
    first_batch = next(pf.iter_batches(batch_size=50000))
    fit_df = first_batch.to_pandas()

    numeric_chunk = fit_df[numeric_features]
    imputer.fit(numeric_chunk)

    categorical_chunk = fit_df[categorical_features]
    encoder.fit(categorical_chunk)

    # 保存转换器
    joblib.dump(imputer, 'imputer.joblib')
    joblib.dump(encoder, 'encoder.joblib')

    # 分批处理完整数据并写出
    print("Processing entire dataset...")
    writer = None
    try:
        for batch in pf.iter_batches(batch_size=50000):
            chunk = batch.to_pandas()

            # --- Label hygiene ---
            if "label" in chunk.columns:
                chunk['label'] = chunk['label'].astype('float')
            elif "is_bad_loan" in chunk.columns:
                chunk['label'] = chunk['is_bad_loan'].astype('float')
            else:
                chunk['label'] = np.nan

            # --- Impute Numeric features ---
            chunk[numeric_features] = imputer.transform(chunk[numeric_features])

            # --- One-Hot Encode Categorical features ---
            encoded_features = encoder.transform(chunk[categorical_features])
            encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

            # 重置索引确保对齐
            chunk = chunk.reset_index(drop=True)
            encoded_df = encoded_df.reset_index(drop=True)

            # 拼接最终特征
            final_features = pd.concat([chunk[numeric_features], encoded_df], axis=1)
            final_chunk = pd.concat([chunk[['loan_application_id', 'label']], final_features], axis=1)

            table = pa.Table.from_pandas(final_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_file_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    print(f"Gold layer processing complete. Output saved to {output_file_path}")
