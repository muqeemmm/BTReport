# --------------------------------------------------------------------------- #
# json_csv_converter.py
#
# Two generalizable conversion utilities:
#   * json_to_csv : one or many JSON files  ->  single flat CSV
#   * csv_to_json : flat CSV                ->  one or many JSON files
#
# Design goals
# ------------
# 1. No hard-coded field names. If upstream code adds new keys to the JSONs,
#    they are picked up automatically (new columns appear in the CSV; new
#    nested paths are reconstructed on the reverse pass).
# 2. Round-trip safe for the structures seen in UCSF-PDGM metadata JSONs:
#    nested dicts, lists of scalars, lists of lists, numbers, strings,
#    booleans, and nulls.
# 3. No reliance on a fixed schema at read-time. Missing keys in some records
#    become NaN in the CSV and are dropped (not written as null) when
#    converting back to JSON, so the reconstructed record matches the
#    original if it did not have that key to begin with.
#
# Flattening rules
# ----------------
#   dict  -> recurse, join keys with `sep` (default ".")
#   list  -> json.dumps(...) (stored as a JSON-encoded string cell)
#   scalar (str/int/float/bool/None) -> stored as-is
#
# Un-flattening rules (csv_to_json)
# ---------------------------------
#   * Column names are split on `sep` and used to rebuild nested dicts.
#   * Each cell is coerced back to its native JSON type:
#       - empty -> None (explicit JSON null)
#       - strings that parse as valid JSON (lists, dicts, bools, null, or
#         numbers) are returned as the parsed value
#       - everything else is left as a string
#   * The CSV is read with `dtype=str, keep_default_na=False, na_values=[""]`
#     so pandas does NOT turn the literal tokens "None", "NA", "False",
#     "True", "42" etc. into NaN / bool / number before we can inspect them.
#
# Usage
# -----
# As a library:
#
#     from json_csv_converter import json_to_csv, csv_to_json
#
#     # 1) Single JSON file  ->  CSV (one row)
#     json_to_csv("metadata/UCSF-PDGM-535_metadata_no_clinical.json",
#                 "out/metadata.csv")
#
#     # 2) A whole directory of JSONs  ->  one merged CSV (one row per file)
#     json_to_csv("metadata/", "out/all_metadata.csv")
#
#     # 3) In-memory dicts or a mixed iterable of paths and dicts
#     json_to_csv([record_a, record_b, "extra.json"], "out/batch.csv")
#
#     # 4) CSV  ->  one JSON per row (filename taken from the `source_file`
#     #    column written in step 1/2)
#     csv_to_json("out/all_metadata.csv", "rebuilt_json/")
#
#     # 5) CSV  ->  a single combined JSON file (list of records, or a
#     #    single object if the CSV has exactly one row)
#     csv_to_json("out/metadata.csv", "out/metadata.json", single_file = True)
#
#     # 6) Drop fields whose CSV cell is empty (useful when a merged CSV has
#     #    sparse columns and you do not want every rebuilt record padded
#     #    with explicit nulls)
#     csv_to_json("out/all_metadata.csv", "rebuilt_json/", keep_nulls = False)
#
# From the command line:
#
#     # JSON -> CSV
#     python json_csv_converter.py json-to-csv  metadata/   out/all.csv
#     python json_csv_converter.py json-to-csv  case.json   out/case.csv  --sort
#
#     # CSV -> JSON  (one file per row)
#     python json_csv_converter.py csv-to-json  out/all.csv rebuilt_json/
#
#     # CSV -> JSON  (single combined file, and drop empty-cell fields)
#     python json_csv_converter.py csv-to-json  out/all.csv all.json \
#            --single-file --drop-nulls
#
# Adding new fields to upstream JSONs
# -----------------------------------
# No changes needed here. New top-level keys, new nested sub-keys, new list
# fields, and new list-of-dict fields are all picked up automatically on the
# next `json_to_csv` call (new columns) and rebuilt on the next `csv_to_json`
# call (new nested paths / re-parsed lists).
# --------------------------------------------------------------------------- #
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

import pandas as pd

PathLike = Union[str, Path]

DEFAULT_SEP       = "."
DEFAULT_ID_COLUMN = "source_file"


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _flatten_record(obj: Any, parent_key: str = "", sep: str = DEFAULT_SEP) -> Dict[str, Any]:
    """
    Recursively flatten a JSON-like object into a single-level dict.

    * Nested dicts are expanded using dotted keys.
    * Lists (including lists of lists, lists of dicts, empty lists) are
      preserved intact as JSON strings so the original structure survives
      a CSV round-trip.
    * Scalars are returned as-is.
    """
    items: Dict[str, Any] = {}

    if isinstance(obj, Mapping):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, Mapping):
                items.update(_flatten_record(v, new_key, sep = sep))
            elif isinstance(v, list):
                # Serialize lists as JSON strings -> lossless round-trip.
                items[new_key] = json.dumps(v, ensure_ascii = False)
            elif isinstance(v, bool):
                # JSON-encode bools ("true"/"false" lowercase) so they
                # round-trip as real bools. If we left them as Python
                # objects, pandas would render them as "True"/"False",
                # which is NOT valid JSON and would come back as strings.
                items[new_key] = json.dumps(v)
            else:
                items[new_key] = v
        return items

    # Top-level object is not a dict: store it under the parent key (or "value")
    key = parent_key or "value"
    if isinstance(obj, list):
        items[key] = json.dumps(obj, ensure_ascii = False)
    elif isinstance(obj, bool):
        items[key] = json.dumps(obj)
    else:
        items[key] = obj
    return items


def _is_missing(value: Any) -> bool:
    """True for NaN, None, or empty string (pandas' default missing markers)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value == "":
        return True
    return False


def _coerce_cell(value: Any) -> Any:
    """
    Convert a single CSV cell back to its native JSON-compatible type.

    Rules:
      * Missing (NaN/None/"") -> None
      * Numbers (int/float/bool) stay as numbers/bools.
      * Strings that parse as JSON and resolve to list/dict/bool/None/number
        are returned as the parsed value.
      * Everything else is returned unchanged.
    """
    if _is_missing(value):
        return None

    # pandas may give us numpy scalar types; normalise.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # Preserve ints where possible (pandas read_csv may upcast to float
        # when a column has any missing rows).
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        # Attempt JSON parse for lists/dicts/bools/null/numbers.
        if stripped[0] in "[{\"-0123456789" or stripped in ("true", "false", "null"):
            try:
                parsed = json.loads(stripped)
                return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return value

    return value


def _unflatten_record(
    flat       : Mapping[str, Any],
    sep        : str  = DEFAULT_SEP,
    keep_nulls : bool = True,
) -> Dict[str, Any]:
    """
    Rebuild a nested dict from a flat {"a.b.c": value, ...} mapping.

    Parameters
    ----------
    keep_nulls : bool, default True
        If True, empty cells are emitted as explicit JSON nulls. This makes
        single-record round-trips exact when the original JSON contained
        explicit null values. Set to False to drop missing cells entirely
        (useful when a merged CSV has many sparsely populated columns and
        you don't want every reconstructed record padded with nulls).
    """
    out: Dict[str, Any] = {}
    for key, raw in flat.items():
        if _is_missing(raw):
            if not keep_nulls:
                continue
            value = None
        else:
            value = _coerce_cell(raw)

        parts = str(key).split(sep)
        cursor: Dict[str, Any] = out
        for part in parts[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, dict):
                existing = {}
                cursor[part] = existing
            cursor = existing
        cursor[parts[-1]] = value
    return out


def _iter_json_inputs(
    json_input: Union[PathLike, Sequence[Union[PathLike, Mapping[str, Any]]], Mapping[str, Any]],
) -> Iterable[tuple[str, Dict[str, Any]]]:
    """
    Normalise the `json_to_csv` input into an iterator of (identifier, record)
    pairs. Accepts any of:

      * a path to a .json file
      * a path to a directory containing .json files
      * an iterable of paths and/or already-loaded dicts
      * a single dict
    """
    # Single dict -> one anonymous record.
    if isinstance(json_input, Mapping):
        yield ("record_0", dict(json_input))
        return

    # Single path (file or directory).
    if isinstance(json_input, (str, Path)):
        path = Path(json_input)
        if path.is_dir():
            files = sorted(p for p in path.glob("*.json") if p.is_file())
            if not files:
                raise FileNotFoundError(f"No .json files found in directory: {path}")
            for f in files:
                with f.open("r", encoding = "utf-8") as fh:
                    yield (f.stem, json.load(fh))
            return
        if path.is_file():
            with path.open("r", encoding = "utf-8") as fh:
                yield (path.stem, json.load(fh))
            return
        raise FileNotFoundError(f"JSON input path does not exist: {path}")

    # Iterable of paths and/or dicts.
    for i, item in enumerate(json_input):
        if isinstance(item, Mapping):
            yield (f"record_{i}", dict(item))
        else:
            p = Path(item)
            with p.open("r", encoding = "utf-8") as fh:
                yield (p.stem, json.load(fh))


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #
def json_to_csv(
    json_input : Union[PathLike, Sequence[Union[PathLike, Mapping[str, Any]]], Mapping[str, Any]],
    csv_output : PathLike,
    *,
    sep           : str  = DEFAULT_SEP,
    id_column     : str  = DEFAULT_ID_COLUMN,
    include_id    : bool = True,
    sort_columns  : bool = False,
) -> pd.DataFrame:
    """
    Convert one or more JSON records into a single flat CSV file.

    Parameters
    ----------
    json_input : path | directory | iterable of paths/dicts | dict
        Source JSON(s). See `_iter_json_inputs` for all accepted forms.
    csv_output : path
        Destination .csv path.
    sep : str, default "."
        Separator used to join nested dict keys into column names.
    id_column : str, default "source_file"
        Name of the column that stores the source file stem (for traceability).
    include_id : bool, default True
        If False, the id column is omitted.
    sort_columns : bool, default False
        If True, columns are sorted alphabetically (id column stays first).

    Returns
    -------
    pd.DataFrame
        The DataFrame that was written to disk.
    """
    rows: List[Dict[str, Any]] = []
    for source_id, record in _iter_json_inputs(json_input):
        flat = _flatten_record(record, sep = sep)
        if include_id:
            # Put id first by inserting it into a fresh dict.
            row = {id_column: source_id, **flat}
        else:
            row = flat
        rows.append(row)

    if not rows:
        raise ValueError("No JSON records were produced from the input.")

    df = pd.DataFrame(rows)

    if sort_columns:
        cols = sorted(c for c in df.columns if c != id_column)
        if include_id and id_column in df.columns:
            cols = [id_column] + cols
        df = df[cols]
    elif include_id and id_column in df.columns:
        # Ensure id column is first even if pandas reordered things.
        cols = [id_column] + [c for c in df.columns if c != id_column]
        df = df[cols]

    csv_output = Path(csv_output)
    csv_output.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(csv_output, index = False)
    return df


def csv_to_json(
    csv_input    : PathLike,
    json_output  : PathLike,
    *,
    sep                 : str  = DEFAULT_SEP,
    id_column           : str  = DEFAULT_ID_COLUMN,
    drop_id_from_record : bool = True,
    single_file         : bool = False,
    indent              : int  = 2,
    keep_nulls          : bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert a flat CSV (as produced by `json_to_csv`) back into JSON records.

    Parameters
    ----------
    csv_input : path
        Source .csv file.
    json_output : path
        * If `single_file=True`: destination .json file containing a list of
          records (or a single object if the CSV has only one row).
        * Else: destination directory into which one .json per row is written.
    sep : str, default "."
        Separator used to split flattened column names back into nested keys.
        Must match whatever was used during `json_to_csv`.
    id_column : str, default "source_file"
        Column used as the per-row filename when writing one file per row.
    drop_id_from_record : bool, default True
        If True, the id column is not embedded inside each JSON record.
    single_file : bool, default False
        If True, all records are written to a single JSON file.
    indent : int, default 2
        Indentation used by `json.dump`.
    keep_nulls : bool, default True
        If True, empty cells are preserved as explicit JSON nulls. Turn off
        if you would rather omit absent fields from each record.

    Returns
    -------
    list[dict]
        The reconstructed records.
    """
    csv_input   = Path(csv_input)
    json_output = Path(json_output)

    # Read every column as a raw string and treat ONLY truly empty cells as
    # missing. Without this, pandas would:
    #   * auto-convert "True"/"False" strings into Python bool,
    #   * upcast numeric columns (changing ints with any NaN into floats),
    #   * silently turn literal tokens like "None", "NA", "NULL", "NaN"
    #     into NaN -- losing a real value from the source JSON.
    # We defer all type inference to `_coerce_cell`, which uses json.loads
    # so only proper JSON literals are re-typed.
    df = pd.read_csv(
        csv_input,
        dtype            = str,
        keep_default_na  = False,
        na_values        = [""],
    )

    records: List[Dict[str, Any]] = []
    filenames: List[str] = []

    for i, row in df.iterrows():
        row_dict = row.to_dict()

        if id_column in row_dict:
            source_id = row_dict[id_column]
            if _is_missing(source_id) or str(source_id).strip() == "":
                source_id = f"record_{i}"
            else:
                source_id = str(source_id)
            if drop_id_from_record:
                row_dict.pop(id_column, None)
        else:
            source_id = f"record_{i}"

        record = _unflatten_record(row_dict, sep = sep, keep_nulls = keep_nulls)
        records.append(record)
        filenames.append(source_id)

    if single_file:
        json_output.parent.mkdir(parents = True, exist_ok = True)
        payload: Any = records[0] if len(records) == 1 else records
        with json_output.open("w", encoding = "utf-8") as fh:
            json.dump(payload, fh, indent = indent, ensure_ascii = False)
    else:
        json_output.mkdir(parents = True, exist_ok = True)
        for name, record in zip(filenames, records):
            safe_name = name if name.lower().endswith(".json") else f"{name}.json"
            with (json_output / safe_name).open("w", encoding = "utf-8") as fh:
                json.dump(record, fh, indent = indent, ensure_ascii = False)

    return records


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = "Convert between JSON records and a flat CSV file.",
    )
    sub = parser.add_subparsers(dest = "command", required = True)

    # json -> csv
    p_j2c = sub.add_parser("json-to-csv",
                           help = "Flatten one JSON file, a list of JSON files, "
                                  "or all .json files in a directory into a CSV.")
    p_j2c.add_argument("json_input",  type = Path,
                       help = "Path to a .json file OR a directory of .json files.")
    p_j2c.add_argument("csv_output",  type = Path, help = "Output .csv path.")
    p_j2c.add_argument("--sep",       default = DEFAULT_SEP,
                       help = "Separator for nested keys (default: '.').")
    p_j2c.add_argument("--id-column", default = DEFAULT_ID_COLUMN,
                       help = "Column name that stores the source file stem.")
    p_j2c.add_argument("--no-id",     action = "store_true",
                       help = "Do not write the source-file id column.")
    p_j2c.add_argument("--sort",      action = "store_true",
                       help = "Sort columns alphabetically.")

    # csv -> json
    p_c2j = sub.add_parser("csv-to-json",
                           help = "Expand a flat CSV back into JSON records.")
    p_c2j.add_argument("csv_input",   type = Path, help = "Input .csv path.")
    p_c2j.add_argument("json_output", type = Path,
                       help = "Output directory (one .json per row) OR, with "
                              "--single-file, a single .json path.")
    p_c2j.add_argument("--sep",       default = DEFAULT_SEP,
                       help = "Separator for nested keys (default: '.').")
    p_c2j.add_argument("--id-column", default = DEFAULT_ID_COLUMN,
                       help = "Column name that stores the source file stem.")
    p_c2j.add_argument("--keep-id",   action = "store_true",
                       help = "Keep the id column inside each JSON record.")
    p_c2j.add_argument("--single-file", action = "store_true",
                       help = "Write all records to a single JSON file.")
    p_c2j.add_argument("--indent",    type = int, default = 2,
                       help = "Indent level for JSON output (default: 2).")
    p_c2j.add_argument("--drop-nulls", action = "store_true",
                       help = "Omit fields whose CSV cell is empty, rather "
                              "than writing them as explicit nulls.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_cli().parse_args(argv)

    if args.command == "json-to-csv":
        df = json_to_csv(
            args.json_input,
            args.csv_output,
            sep          = args.sep,
            id_column    = args.id_column,
            include_id   = not args.no_id,
            sort_columns = args.sort,
        )
        print(f"Wrote {len(df)} row(s) and {len(df.columns)} column(s) "
              f"to {args.csv_output}")
        return

    if args.command == "csv-to-json":
        records = csv_to_json(
            args.csv_input,
            args.json_output,
            sep                 = args.sep,
            id_column           = args.id_column,
            drop_id_from_record = not args.keep_id,
            single_file         = args.single_file,
            indent              = args.indent,
            keep_nulls          = not args.drop_nulls,
        )
        target = args.json_output
        print(f"Wrote {len(records)} record(s) to {target}")


if __name__ == "__main__":
    main()
