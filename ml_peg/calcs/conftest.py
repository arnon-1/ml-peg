"""Configure pytest for calculations."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import CallInfo, Config, Item, Parser

from ml_peg import models
from ml_peg.calcs.utils import completion


def pytest_addoption(parser: Parser) -> None:
    """
    Add custom CLI inputs to pytest.

    Parameters
    ----------
    parser
        Pytest parser object.
    """
    parser.addoption(
        "--run-mock",
        action="store_true",
        default=False,
        help="Include mock model in tests",
    )
    parser.addoption(
        "--mock-only",
        action="store_true",
        default=False,
        help="Only run mock model, ignoring other models",
    )
    parser.addoption(
        "--force-calcs",
        action="store_true",
        default=False,
        help="Run calculations even if they previously completed",
    )


def pytest_configure(config: Config) -> None:
    """
    Configure pytest to custom CLI inputs.

    Parameters
    ----------
    config
        Pytest configuration object.
    """
    # Set current models from CLI input
    models.run_mock = config.getoption("--run-mock")
    models.mock_only = config.getoption("--mock-only")


def _item_mlip(item: Item) -> tuple[str, Any] | None:
    """
    Get the model a test item is parametrized with, if any.

    Parameters
    ----------
    item
        Pytest test item.

    Returns
    -------
    tuple[str, Any] | None
        The item's (model_name, model) "mlip" parameter, or None if the test
        is not parametrized over models.
    """
    callspec = getattr(item, "callspec", None)
    mlip = callspec.params.get("mlip") if callspec is not None else None
    if isinstance(mlip, tuple) and len(mlip) == 2 and isinstance(mlip[0], str):
        return mlip
    return None


def _module_models(item: Item) -> dict[str, Any] | None:
    """
    Get the module-level MODELS dict for a test item, if defined.

    Parameters
    ----------
    item
        Pytest test item.

    Returns
    -------
    dict[str, Any] | None
        The test module's MODELS dict, or None if not defined.
    """
    module_models = getattr(getattr(item, "module", None), "MODELS", None)
    return module_models if isinstance(module_models, dict) else None


def pytest_runtest_setup(item: Item) -> None:
    """
    Skip calculations that previously completed with identical inputs.

    Tests parametrized over models ("mlip") are skipped per model. For tests
    that loop over the module's MODELS dict instead, completed models are
    removed from the dict, so the loop only runs the remaining models, and
    the test is skipped entirely if none remain. Pruned models are restored
    in pytest_runtest_teardown.

    Parameters
    ----------
    item
        Pytest test item.
    """
    mlip = _item_mlip(item)
    module_models = _module_models(item)
    if mlip is None and module_models is None:
        return

    completion.clear_data_files()
    force = item.config.getoption("--force-calcs")
    calc_dir = item.path.parent
    out_path = calc_dir / "outputs"

    if mlip is not None:
        name = mlip[0]
        fingerprint = completion.calc_fingerprint(calc_dir, name)
        if not force and completion.is_complete(
            out_path, name, item.originalname, fingerprint
        ):
            pytest.skip(f"'{name}' previously completed. Use --force-calcs to re-run.")
        return

    item._pruned_models = {}
    if force:
        return

    for name in list(module_models):
        fingerprint = completion.calc_fingerprint(calc_dir, name)
        if completion.is_complete(out_path, name, item.name, fingerprint):
            print(f"[skip] {item.name}: '{name}' previously completed")
            item._pruned_models[name] = module_models.pop(name)

    if item._pruned_models and not module_models:
        pytest.skip("All models previously completed. Use --force-calcs to re-run.")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Item, call: CallInfo):
    """
    Record whether the test call phase passed.

    Needed as pytest does not expose the test outcome to
    pytest_runtest_teardown, which must only mark calculations as completed
    if the test passed.

    Parameters
    ----------
    item
        Pytest test item.
    call
        Result of the test phase that just ran.

    Yields
    ------
    Result
        Hook wrapper result holding the test report.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.passed:
        item._calcs_passed = True


def pytest_runtest_teardown(item: Item) -> None:
    """
    Mark completed calculations and restore pruned models.

    Parameters
    ----------
    item
        Pytest test item.
    """
    mlip = _item_mlip(item)
    pruned = getattr(item, "_pruned_models", None)
    module_models = _module_models(item)

    if getattr(item, "_calcs_passed", False):
        if mlip is not None:
            test_name, names = item.originalname, [mlip[0]]
        elif pruned is not None:
            test_name, names = item.name, list(module_models)
        else:
            test_name, names = item.name, []

        calc_dir = item.path.parent
        data_files = completion.used_data_files()
        for name in names:
            completion.mark_complete(
                calc_dir / "outputs",
                name,
                test_name,
                completion.calc_fingerprint(calc_dir, name),
                data_files,
            )

    if pruned is not None and module_models is not None:
        module_models.update(pruned)
