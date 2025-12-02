# Copyright (c) QuantCo and pydiverse contributors 2025-2025
# SPDX-License-Identifier: BSD-3-Clause
import base64
import hashlib
import types
import warnings

try:
    import polars as pl
except ImportError:
    pl = types.ModuleType("pl")
    pl.DataFrame = None

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.types as pat
except ImportError:
    pa = types.ModuleType("pa")
    pa.DataType = None
    pa.Table = None
    pa.Schema = None
    pat = types.ModuleType("pat")
    ipc = types.ModuleType("ipc")


def stable_hash(*args: str) -> str:
    """Compute a hash over a set of strings

    :param args: Some strings from which to compute the cache key
    :return: A sha256 base32 digest, trimmed to 20 char length
    """

    combined_hash = hashlib.sha256(b"PYDIVERSE")
    for arg in args:
        arg_bytes = str(arg).encode("utf8")
        arg_bytes_len = len(arg_bytes).to_bytes(length=8, byteorder="big")

        combined_hash.update(arg_bytes_len)
        combined_hash.update(arg_bytes)

    # Only take first 20 characters of base32 digest (100 bits). This
    # provides 50 bits of collision resistance, which is more than enough.
    # To illustrate: If you were to generate 1k hashes per second,
    # you still would have to wait over 800k years until you encounter
    # a collision.

    # NOTE: Can't use base64 because it contains lower and upper case
    #       letters; identifiers in pipedag are all lowercase
    hash_digest = combined_hash.digest()
    hash_str = base64.b32encode(hash_digest).decode("ascii").lower()
    return hash_str[:20]


def _hash_polars_dataframe(df: pl.DataFrame, use_init_repr=False) -> str:
    """
    Compute a stable hash for a polars DataFrame. This function is meant as a backup for
    the pyarrow hashing function in case it breaks.

    The function may fail for some known causes:
    * In old polars versions, hash_rows() does not work if there are list columns
      https://github.com/pola-rs/polars/issues/24121
    * Hashing categorical and enum columns may lead to unstable results depending on the internal dictionary encoding.
    * E.g.
    ```python
    import polars as pl
    df_c = pl.DataFrame(dict(x=["c", "b", "c"]), schema=dict(x=pl.Enum(["b", "c"])))
    _hash_polars_dataframe(df_c)
    ```
    and
    ```python
    import polars as pl
    df_a = pl.DataFrame(dict(x=["apple", "banana", "apple"]), schema=dict(x=pl.Enum(["apple", "banana"])))
    df_c = pl.DataFrame(dict(x=["c", "b", "c"]), schema=dict(x=pl.Enum(["b", "c"])))
    _hash_polars_dataframe(df_c)
    ```
    will give different results because the internal dictionary encoding is different. Same applies
    to the Enum type replaced by Categorical.
    See also https://github.com/pola-rs/polars/issues/14829
    """
    if not use_init_repr:
        try:
            schema_hash = stable_hash(repr(df.schema))
            if df.is_empty():
                content_hash = "empty"
            else:
                content_hash = str(
                    df.hash_rows()  # We get a Series of hashes, one for each row
                    # Since polars only hashes rows, we need to implode the Series into
                    # a single row to get a single hash
                    .implode()
                    .hash()
                    .item()
                )
            return "0" + stable_hash(schema_hash, content_hash)
        except Exception:
            warnings.warn(
                "Failed to compute hash for polars DataFrame in fast way. Falling back to to_init_repr() method.",
                stacklevel=1,
            )

    # fallback to to_init_repr string representation
    return "1" + stable_hash(df.to_init_repr(len(df)))


# ---------- type helpers ----------


def _without_dict_type(t: pa.DataType) -> pa.DataType:
    """Return a type like t but with all Dictionary types replaced by their value types (recursively)."""
    if pat.is_dictionary(t):
        return t.value_type
    if pat.is_struct(t):
        new_fields = [pa.field(f.name, _without_dict_type(f.type), nullable=f.nullable) for f in t]
        return pa.struct(new_fields)
    if pat.is_list(t):
        return pa.list_(_without_dict_type(t.value_type))
    if pat.is_large_list(t):
        return pa.large_list(_without_dict_type(t.value_type))
    if pat.is_fixed_size_list(t):
        return pa.list_(_without_dict_type(t.value_type), t.list_size)
    if pat.is_map(t):
        return pa.map_(
            _without_dict_type(t.key_type),
            _without_dict_type(t.item_type),
            keys_sorted=t.keys_sorted,
        )
    # (We leave unions and other rare complex types as-is; they typically don't carry dictionaries)
    return t


def _strip_field_metadata(schema: pa.Schema) -> pa.Schema:
    """Drop field-level metadata to avoid non-semantic differences."""
    fields = [pa.field(f.name, f.type, nullable=f.nullable) for f in schema]
    return pa.schema(fields, metadata=None)


def _dictionary_free_schema(schema: pa.Schema) -> pa.Schema:
    """Make a schema with identical field names/nullable flags, but no dictionary types (recursively)."""
    fields = [pa.field(f.name, _without_dict_type(f.type), nullable=f.nullable) for f in schema]
    return pa.schema(fields)  # no schema metadata


def _as_arrow_table(obj) -> pa.Table:
    """Accept pyarrow Table, Polars DF, pandas DF and return a pa.Table."""
    if isinstance(obj, pa.Table):
        return obj

    # Polars DataFrame
    if pl is not None and isinstance(obj, pl.DataFrame):
        return obj.to_arrow()

    # Pandas DataFrame
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return pa.Table.from_pandas(obj, preserve_index=True)

    raise TypeError(f"Unsupported input type: {type(obj)!r}")


def _hash_schema(obj) -> str:
    """Compute a stable hash for the schema of a pyarrow Table, Polars DataFrame, or pandas DataFrame."""
    if isinstance(obj, pa.Table):
        return stable_hash(repr(obj.schema))
    if pl is not None and isinstance(obj, pl.DataFrame):
        return stable_hash(repr(obj.schema))
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return stable_dataframe_hash(pa.Table.from_pandas(pd.DataFrame(obj.dtypes.astype("string[pyarrow]"))))
    raise TypeError(f"Unsupported input type: {type(obj)!r}")


def stable_dataframe_hash(
    df,
    *,
    strip_schema_metadata: bool = True,
    drop_field_metadata: bool = True,
) -> str:
    """
    Compute a stable, cross-library SHA-256 of a full table.

    Accepts: Polars DataFrame, PyArrow Table, pandas DataFrame.

    What makes it stable:
      • combine_chunks()
      • recursively decode dictionary/categorical encodings via cast to a dict-free schema
      • strip schema + field metadata
      • serialize as a single-batch Arrow IPC stream with pinned options and hash the bytes
    """
    # 1) Convert to Arrow and canonicalize chunking
    t = _as_arrow_table(df).combine_chunks()

    # 2) Drop non-semantic metadata (schema + field-level)
    if strip_schema_metadata:
        t = t.replace_schema_metadata(None)
    if drop_field_metadata:
        t = pa.Table.from_arrays(list(t.columns), schema=_strip_field_metadata(t.schema))

    # 3) Hash schema separately, since we replace dictionaries in the next step
    # If we don't have a pyarrow Table, we use the original df to compute the schema hash
    # because some dtype information may be lost in the conversion to pyarrow.
    # For example, after converting from pandas to pyarrow, we no longer know if a column
    # was arrow or numpy backed.
    schema_hash = _hash_schema(df) if not isinstance(df, pa.Table) else _hash_schema(t)

    # 4) Decode all dictionary encodings (including nested) by casting to a dict-free schema
    dict_free = _dictionary_free_schema(t.schema)
    if dict_free != t.schema:
        t = t.cast(dict_free)

    # 5) Deterministic IPC serialization (single batch, no compression, explicit metadata version)
    sink = pa.BufferOutputStream()
    write_opts = ipc.IpcWriteOptions(metadata_version=ipc.MetadataVersion.V5, compression=None)
    with ipc.RecordBatchStreamWriter(sink, t.schema, options=write_opts) as writer:
        writer.write_table(t, max_chunksize=t.num_rows)

    # 6) Hash subtleties of pandas DataFrame object type columns that are lost when converting
    # to pyarrow, e.g. columns mixing dates and datetimes, by hashing their CSV representation.
    # See https://github.com/apache/arrow/issues/41896 for an example of such a conversion issue.
    if not isinstance(df, (pa.Table, pl.DataFrame)):
        import pandas as pd

        assert isinstance(df, pd.DataFrame), f"Dataframe of type {type(df)} not supported"

        df_object = df.select_dtypes(include=["object"])
        csv = df_object.to_csv()
    else:
        csv = ""

    return stable_hash(schema_hash, hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest(), csv)
