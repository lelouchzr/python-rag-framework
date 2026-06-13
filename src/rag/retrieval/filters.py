"""Metadata filtering utilities."""

from collections.abc import Sequence

from rag.documents.models import Metadata, MetadataValue

MetadataOperatorValue = MetadataValue | Sequence[MetadataValue]
MetadataCondition = dict[str, MetadataOperatorValue]
MetadataFilterValue = MetadataValue | MetadataCondition
MetadataFilter = dict[str, MetadataFilterValue]


def matches_metadata(metadata: Metadata, metadata_filter: MetadataFilter) -> bool:
    """Return whether metadata satisfies an exact or operator-based filter."""

    for key, condition in metadata_filter.items():
        actual = metadata.get(key)
        if isinstance(condition, dict):
            if not _matches_condition(actual, condition, key in metadata):
                return False
        elif actual != condition:
            return False
    return True


def _matches_condition(
    actual: MetadataValue,
    condition: MetadataCondition,
    exists: bool,
) -> bool:
    for operator, expected in condition.items():
        if operator == "$eq" and actual != expected:
            return False
        if operator == "$ne" and actual == expected:
            return False
        if operator == "$in" and actual not in _as_sequence(expected):
            return False
        if operator == "$nin" and actual in _as_sequence(expected):
            return False
        if operator == "$contains" and not _contains(actual, expected):
            return False
        if operator == "$exists" and exists is not bool(expected):
            return False
        if operator in {"$gt", "$gte", "$lt", "$lte"} and not _compare(
            actual,
            expected,
            operator,
        ):
            return False
        if operator not in {
            "$eq",
            "$ne",
            "$in",
            "$nin",
            "$contains",
            "$exists",
            "$gt",
            "$gte",
            "$lt",
            "$lte",
        }:
            return False
    return True


def _as_sequence(value: MetadataOperatorValue) -> Sequence[MetadataValue]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return (value,)
    return value


def _contains(actual: MetadataValue, expected: MetadataOperatorValue) -> bool:
    if actual is None:
        return False
    if isinstance(actual, str):
        return str(expected) in actual
    return actual == expected


def _compare(
    actual: MetadataValue,
    expected: MetadataOperatorValue,
    operator: str,
) -> bool:
    if not isinstance(actual, int | float) or isinstance(actual, bool):
        return False
    if not isinstance(expected, int | float) or isinstance(expected, bool):
        return False

    if operator == "$gt":
        return actual > expected
    if operator == "$gte":
        return actual >= expected
    if operator == "$lt":
        return actual < expected
    return actual <= expected
