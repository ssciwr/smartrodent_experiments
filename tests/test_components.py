"""Building configured components from a name plus parameters."""

import builtins
import fractions
import math
import numbers

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smartrodent.components import ComponentSpec


def test_a_bare_name_is_built_from_the_default_namespace():
    """Callers name a component of their own namespace, e.g. an Optuna sampler class."""
    spec = ComponentSpec(name="Fraction", parameters={"numerator": 1, "denominator": 3})
    assert spec.build(fractions, numbers.Rational) == fractions.Fraction(1, 3)


def test_an_import_string_selects_a_component_from_any_module():
    """`module:attr` plugs in custom code regardless of the caller's namespace."""
    spec = ComponentSpec(name="decimal:Decimal", parameters={"value": "1.5"})
    assert str(spec.build(fractions, None)) == "1.5"


def test_functions_are_components_too():
    """A configured function is called with its parameters, e.g. a metric or a rule."""
    spec = ComponentSpec(name="json:dumps", parameters={"sort_keys": True})
    assert spec.build(builtins, None, {"b": 1, "a": 2}) == '{"a": 2, "b": 1}'


def test_pipeline_arguments_are_passed_alongside_the_parameters():
    """Positional and keyword arguments supplied by the caller join the configured ones."""
    spec = ComponentSpec(name="Fraction", parameters={"denominator": 3})
    assert spec.build(fractions, None, 2) == fractions.Fraction(2, 3)


_keys = st.text("abcd", min_size=1, max_size=2)


@given(
    parameters=st.dictionaries(_keys, st.integers(), max_size=4),
    injected=st.dictionaries(_keys, st.integers(), max_size=4),
)
def test_supplied_arguments_never_silently_override_parameters(parameters, injected):
    """A key set by both the configuration and the pipeline is an error, never a merge."""
    spec = ComponentSpec(name="dict", parameters=parameters)
    collisions = set(parameters) & set(injected)
    if collisions:
        with pytest.raises(ValueError, match=min(collisions)):
            spec.build(builtins, None, **injected)
    else:
        assert spec.build(builtins, None, **injected) == {**parameters, **injected}


@pytest.mark.parametrize(
    "name,parameters,namespace,base",
    [
        ("Fancy", {}, fractions, None),  # unknown in the namespace
        ("no_such_module:Thing", {}, fractions, None),  # unknown module
        ("fractions:Missing", {}, fractions, None),  # unknown attribute
        ("Fraction", {}, fractions, str),  # not of the required kind
        ("Rational", {}, numbers, numbers.Rational),  # abstract
        ("pi", {}, math, None),  # not callable
        ("Fraction", {"bogus": 1}, fractions, None),  # rejected by the constructor
    ],
)
def test_components_that_cannot_be_built_are_value_errors(
    name, parameters, namespace, base
):
    """Configuration mistakes surface as one error type, before any work starts."""
    with pytest.raises(ValueError):
        ComponentSpec(name=name, parameters=parameters).build(namespace, base)
