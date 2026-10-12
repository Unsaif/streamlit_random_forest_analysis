"""Stage S0: typing, imputation, encoding and label encoding."""

import numpy as np
import pandas as pd
import pytest

from fastforest.preprocess import (
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY_CODE,
    Preprocessor,
    encode_labels,
    split_columns,
)


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0],
            "int": [1, 2, 3, 4],
            "cat": ["a", "b", None, "a"],
            "flag": [True, False, True, False],
            "low": [0, 1, 0, 1],
        }
    )


def test_split_columns_keeps_low_cardinality_numeric_continuous_by_default():
    continuous, categorical = split_columns(frame())
    assert continuous == ["num", "int", "low"]
    assert categorical == ["cat", "flag"]


def test_split_columns_low_cardinality_rule_is_opt_in():
    continuous, categorical = split_columns(frame(), categorical_low_cardinality=True, max_card=2)
    assert continuous == ["num", "int"]
    assert categorical == ["cat", "flag", "low"]


def test_nans_imputed_with_training_median():
    out = Preprocessor().fit_transform(frame())
    assert out["num"].tolist() == [1.0, 3.0, 3.0, 4.0]
    assert not out.isna().any().any()


def test_column_names_and_order_preserved():
    out = Preprocessor().fit_transform(frame())
    assert out.columns.tolist() == frame().columns.tolist()
    assert out.index.equals(frame().index)


def test_categorical_encoded_in_sorted_order_with_missing_category():
    pre = Preprocessor().fit(frame())
    out = pre.transform(frame())
    assert pre.categories_["cat"] == [MISSING_CATEGORY, "a", "b"]
    assert out["cat"].tolist() == [1, 2, 0, 1]
    assert out["flag"].tolist() == [1, 0, 1, 0]  # "False" < "True"
    assert out["cat"].dtype.kind == "i"


def test_unseen_category_does_not_crash():
    pre = Preprocessor().fit(frame())
    new = frame()
    new.loc[0, "cat"] = "zzz"
    out = pre.transform(new)
    assert out.loc[0, "cat"] == UNKNOWN_CATEGORY_CODE


def test_missing_indicator_columns_are_opt_in():
    assert "num_na" not in Preprocessor().fit_transform(frame()).columns
    pre = Preprocessor(add_missing_indicator=True).fit(frame())
    out = pre.transform(frame())
    assert pre.feature_names_out_ == [*frame().columns, "num_na"]
    assert out["num_na"].tolist() == [0, 1, 0, 0]


def test_transform_subset_of_columns():
    pre = Preprocessor(add_missing_indicator=True).fit(frame())
    out = pre.transform(frame(), columns=["cat", "num_na"])
    assert out.columns.tolist() == ["cat", "num_na"]
    with pytest.raises(ValueError, match="unknown output columns"):
        pre.transform(frame(), columns=["nope"])


def test_integer_columns_keep_dtype():
    out = Preprocessor().fit_transform(frame())
    assert out["int"].dtype.kind == "i"


def test_nullable_dtypes_are_handled():
    df = pd.DataFrame(
        {
            "a": pd.array([1, None, 3], dtype="Int64"),
            "b": pd.array([True, None, False], dtype="boolean"),
        }
    )
    out = Preprocessor().fit_transform(df)
    assert out["a"].tolist() == [1.0, 2.0, 3.0]
    assert out["b"].tolist() == [2, 0, 1]  # categories: (missing), False, True


def test_all_missing_column_imputed_with_zero():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    out = Preprocessor().fit_transform(df)
    assert out["a"].tolist() == [0.0, 0.0]


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        Preprocessor().transform(frame())


def test_missing_input_column_raises():
    pre = Preprocessor().fit(frame())
    with pytest.raises(ValueError, match="missing from input"):
        pre.transform(frame().drop(columns=["num"]))


def test_duplicate_columns_raise():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate"):
        Preprocessor().fit(df)


def test_not_a_dataframe_raises():
    with pytest.raises(TypeError):
        Preprocessor().fit(np.zeros((3, 2)))


def test_encode_labels_sorted_classes():
    codes, classes = encode_labels(["b", "a", "c", "a"])
    assert classes == ["a", "b", "c"]
    assert codes.tolist() == [1, 0, 2, 0]


def test_encode_labels_numeric_sorted_numerically():
    codes, classes = encode_labels(pd.Series([10, 2, 2]))
    assert classes == ["2", "10"]
    assert codes.tolist() == [1, 0, 0]


def test_encode_labels_mixed_types_fall_back_to_strings():
    codes, classes = encode_labels([1, "a", 1])
    assert classes == ["1", "a"]
    assert codes.tolist() == [0, 1, 0]


def test_encode_labels_missing_raises():
    with pytest.raises(ValueError, match="missing"):
        encode_labels(["a", None])
