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


def hash_polars_dataframe(df: pl.DataFrame, use_init_repr=False) -> str:
    if not use_init_repr:
        try:

            def unnest_all(df: pl.DataFrame) -> pl.DataFrame:
                while struct_cols_and_dtypes := [
                    (col, dtype)
                    for col, dtype in df.schema.items()
                    if dtype == pl.Struct
                ]:
                    df = df.with_columns(
                        pl.col(struct_col_name).struct.rename_fields(
                            [
                                stable_hash(field_name, struct_col_name)
                                for field_name in struct_dtype.fields
                            ]
                        )
                        for struct_col_name, struct_dtype in struct_cols_and_dtypes
                    ).unnest(
                        [
                            struct_col_name
                            for struct_col_name, _ in struct_cols_and_dtypes
                        ]
                    )
                return df

            schema_hash = stable_hash(repr(df.schema))
            if df.is_empty():
                content_hash = "empty"
            else:
                # Since we need to operate on all lists, we need to access them first
                # if they are within a struct.
                df = unnest_all(df)
                array_columns = [
                    col for col, dtype in df.schema.items() if dtype == pl.Array
                ]
                list_columns = [
                    col for col, dtype in df.schema.items() if dtype == pl.List
                ]
                lf = df.lazy()
                if array_columns:
                    lf = lf.with_columns(pl.col(array_columns).reshape([-1]).implode())
                lf = lf.with_columns(
                    # Necessary because hash() does not work on lists of strings.
                    # This can be removed when
                    # https://github.com/pola-rs/polars/issues/21523 is resolved
                    # in all supported versions of polars.
                    pl.selectors.by_dtype(pl.List(pl.String)).list.eval(
                        pl.element().hash()
                    )
                )
                if list_columns or array_columns:
                    # Necessary because hash_rows() does not work on lists.
                    # This can be removed when
                    # https://github.com/pola-rs/polars/issues/24121 is resolved
                    # in all supported versions of polars.
                    lf = lf.with_columns(pl.col(*list_columns, *array_columns).hash())

                content_hash = str(
                    lf.collect()
                    .hash_rows()  # We get a Series of hashes, one for each row
                    # Since polars only hashes rows, we need to implode the Series into
                    # a single row to get a single hash
                    .implode()
                    .hash()
                    .item()
                )
            return "0" + stable_hash(schema_hash, content_hash)
        except Exception:
            warnings.warn(
                "Failed to compute hash for polars DataFrame in fast way. "
                "Falling back to to_init_repr() method.",
                stacklevel=1,
            )

    # fallback to to_init_repr string representation
    return "1" + stable_hash(df.to_init_repr(len(df)))
