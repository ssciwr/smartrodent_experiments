"""Build configured components, e.g. Optuna samplers or dataset creators, from a name plus parameters."""

import importlib
import inspect
from collections.abc import Callable
from dataclasses import field
from types import ModuleType
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from smartrodent.training_validation import Nonblank


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ComponentSpec:
    """A class or function chosen by name, and the keyword arguments it is built with.

    Args:
        name: An attribute of the caller's default namespace, or ``module:attr`` for
            any importable component (configuration files are trusted code).
        parameters: Keyword arguments; empty means the component's own defaults.
    """

    name: Nonblank
    parameters: dict[str, Any] = field(default_factory=dict)

    def resolve(
        self, namespace: ModuleType, base: type | None = None, /
    ) -> Callable[..., Any]:
        """Resolve the configured class or function without calling it.

        Args:
            namespace: Module in which a bare name is looked up.
            base: Required base class, or None for any callable.

        Returns:
            The configured callable.

        Raises:
            ValueError: The name is unknown, or does not identify an allowed callable.
        """
        module_name, separator, attribute = self.name.rpartition(":")
        if separator:
            try:
                module = importlib.import_module(module_name)
            except ImportError as error:
                raise ValueError(
                    f"Cannot import module '{module_name}' of component '{self.name}'"
                ) from error
        else:
            module = namespace
        if not hasattr(module, attribute):
            raise ValueError(f"'{attribute}' is not defined in {module.__name__}")
        component = getattr(module, attribute)
        if base is not None and not (
            isinstance(component, type)
            and issubclass(component, base)
            and not inspect.isabstract(component)
        ):
            raise ValueError(f"'{self.name}' is not a concrete {base.__name__}")
        elif not callable(component):
            raise ValueError(f"'{self.name}' is not callable")
        return component

    def build(
        self, namespace: ModuleType, base: type | None = None, /, *args, **supplied
    ) -> Any:
        """Call the component with the caller's arguments and the configured parameters.

        Args:
            namespace: Module in which a bare name is looked up.
            base: Required base class, or None for any callable.
            *args: Positional arguments supplied by the caller.
            **supplied: Keyword arguments supplied by the caller; the configuration
                may not set them too.

        Returns:
            The built component (an instance, or a function's return value).

        Raises:
            ValueError: The component is unknown, not a concrete subclass of ``base``,
                sets a supplied argument, or rejects its arguments. A ``TypeError`` or
                ``ValueError`` raised by the call itself, including inside a function
                component, is reported as rejected arguments.
        """
        component = self.resolve(namespace, base)
        collisions = sorted(set(self.parameters) & set(supplied))
        if collisions:
            raise ValueError(
                f"Parameters of '{self.name}' may not set {collisions}; "
                "the pipeline supplies them"
            )
        try:
            return component(*args, **self.parameters, **supplied)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid parameters for '{self.name}': {error}"
            ) from error
