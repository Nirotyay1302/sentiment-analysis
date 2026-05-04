import sys
import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the file uploader
content = content.replace('uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="dataset_uploader")', 'uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx", "xls"], key="dataset_uploader")')

# Replace the dataset reader
old_func = r'def read_csv_with_header_detection\(uploaded\):.*?return None, None'

new_func = '''def read_csv_with_header_detection(uploaded):
    """Read a CSV or Excel file and try to detect if the real header is on a later row."""
    try:
        is_excel = uploaded.name.lower().endswith(('.xlsx', '.xls'))
        
        # Read preview
        if is_excel:
            if hasattr(uploaded, 'seek'): uploaded.seek(0)
            preview = pd.read_excel(uploaded, header=None, nrows=10, dtype=str)
            if hasattr(uploaded, 'seek'): uploaded.seek(0)
        else:
            if hasattr(uploaded, 'read'):
                uploaded.seek(0)
                sample_text = uploaded.read().decode('utf-8', errors='ignore')
                import io
                preview_buf = io.StringIO(sample_text)
            else:
                preview_buf = uploaded
            preview = pd.read_csv(preview_buf, header=None, nrows=10, dtype=str, keep_default_na=False)

        header_row = None
        for i, row in preview.iterrows():
            row_vals = ' '.join([str(x).lower() for x in row.tolist()])
            if re.search(r'\\btext\\b', row_vals):
                header_row = i
                break

        if header_row is None:
            first0 = str(preview.iloc[0, 0]).lower()
            if 'social media' in first0 or 'sentiments' in first0 or 'social' in first0:
                header_row = 1

        # Helper to read full file safely
        def safe_read_full_file(file_obj, header_idx):
            kwargs = {'header': header_idx, 'dtype': str}
            if not is_excel:
                kwargs['keep_default_na'] = False
                
            if hasattr(file_obj, 'size') and file_obj.size > 50 * 1024 * 1024:
                kwargs['nrows'] = 50000
                st.warning("File is extremely large (>50MB). Only loading the first 50,000 rows to prevent Streamlit memory crashes.")
                
            if is_excel:
                if hasattr(file_obj, 'seek'): file_obj.seek(0)
                return pd.read_excel(file_obj, **kwargs)
                
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    if hasattr(file_obj, 'seek'): file_obj.seek(0)
                    return pd.read_csv(file_obj, encoding=enc, **kwargs)
                except UnicodeDecodeError:
                    continue
            if hasattr(file_obj, 'seek'): file_obj.seek(0)
            return pd.read_csv(file_obj, encoding='utf-8', encoding_errors='replace', **kwargs)

        df = safe_read_full_file(uploaded, header_row if header_row is not None else 0)

        # Excel can return NaNs for empty strings, so fill them for text processing
        if is_excel:
            df = df.fillna('')

        return df, header_row
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None'''

content = re.sub(old_func, new_func, content, flags=re.DOTALL)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)
