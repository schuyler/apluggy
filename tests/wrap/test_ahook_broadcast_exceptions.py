"""Exception handling for broadcast (firstresult=False) hooks.

With or without a wrapper, a broadcast runs every implementation to
completion, then raises the first exception in execution order. Exceptions
from the other implementations are attached to it as notes rather than
dropped (Python 3.11+, where BaseException.add_note exists).
"""

import asyncio
import sys

import pytest

import apluggy as pluggy
from apluggy import PluginManager

_PROJECT = "test_ahook_broadcast_exceptions"

hookspec = pluggy.HookspecMarker(_PROJECT)
hookimpl = pluggy.HookimplMarker(_PROJECT)

needs_notes = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="BaseException.add_note is 3.11+"
)


class Spec:
    """Hookspec namespace: one broadcast hook."""

    @hookspec
    async def ahook(self, value: int) -> int:
        ...  # pragma: no cover


class Wrapper:
    """A pass-through wrapper, to route the call through _async_multicall."""

    @hookimpl(wrapper=True)
    def ahook(self, value: int):
        return (yield)


def make_pm(*plugins: object) -> PluginManager:
    """Build a plugin manager with ``Spec`` and the given plugins, in order."""
    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    for plugin in plugins:
        pm.register(plugin)
    return pm


# Both paths must behave the same, so every test runs with and without a
# wrapper.
paths = pytest.mark.parametrize("wrapped", [False, True], ids=["plain", "wrapped"])


@paths
async def test_raise_waits_for_siblings(wrapped: bool) -> None:
    """The exception reaches the caller only after every sibling finished."""
    finished: list[str] = []

    class Slow:
        @hookimpl
        async def ahook(self, value: int) -> int:
            await asyncio.sleep(0.05)
            finished.append("slow")
            return value

    class Raises:
        @hookimpl
        async def ahook(self, value: int) -> int:
            raise ValueError("boom")

    plugins: list[object] = [Slow(), Raises()]
    if wrapped:
        plugins.append(Wrapper())
    pm = make_pm(*plugins)

    with pytest.raises(ValueError, match="boom"):
        await pm.ahook.ahook(value=1)
    assert finished == ["slow"]


@paths
async def test_first_in_execution_order_wins(wrapped: bool) -> None:
    """The raised exception is the first in execution order, not in time.

    Execution order is last-registered first, so ``Early`` (registered last)
    runs first in order but raises after ``Late`` does.
    """

    class Late:
        @hookimpl
        async def ahook(self, value: int) -> int:
            raise KeyError("late in order, early in time")

    class Early:
        @hookimpl
        async def ahook(self, value: int) -> int:
            await asyncio.sleep(0.05)
            raise ValueError("early in order, late in time")

    plugins: list[object] = [Late(), Early()]
    if wrapped:
        plugins.append(Wrapper())
    pm = make_pm(*plugins)

    with pytest.raises(ValueError, match="early in order"):
        await pm.ahook.ahook(value=1)


@needs_notes
@paths
async def test_other_exceptions_become_notes(wrapped: bool) -> None:
    """A second exception is not dropped: it is a note on the raised one,
    naming the hook and the other exception's type and message."""

    class Second:
        @hookimpl
        async def ahook(self, value: int) -> int:
            raise KeyError("second")

    class First:
        @hookimpl
        async def ahook(self, value: int) -> int:
            raise ValueError("first")

    plugins: list[object] = [Second(), First()]
    if wrapped:
        plugins.append(Wrapper())
    pm = make_pm(*plugins)

    with pytest.raises(ValueError, match="first") as info:
        await pm.ahook.ahook(value=1)

    notes = getattr(info.value, "__notes__", [])
    assert len(notes) == 1, notes
    assert "ahook" in notes[0]
    assert "KeyError" in notes[0]
    assert "second" in notes[0]


@paths
async def test_results_unchanged_without_exceptions(wrapped: bool) -> None:
    """No exception: the result list in execution order, as before."""

    class A:
        @hookimpl
        async def ahook(self, value: int) -> int:
            return value + 1

    class B:
        @hookimpl
        async def ahook(self, value: int) -> int:
            return value + 2

    plugins: list[object] = [A(), B()]
    if wrapped:
        plugins.append(Wrapper())
    pm = make_pm(*plugins)

    assert await pm.ahook.ahook(value=10) == [12, 11]
