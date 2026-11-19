"""Tests for ``AHook(return_exceptions=True)`` / ``pm.ahook_isolated``.

Each test maps to one acceptance criterion of requirements 1-7 in
``.claude/plans/ahook-isolated.md`` (requirement 8 is covered by the
cherry-picked tests, requirement 9 is the README). Timeouts are kept small and
every wait that could deadlock is wrapped in ``asyncio.wait_for``.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pluggy import HookCallError

import apluggy as pluggy
from apluggy import PluginManager
from apluggy.wrap.ext import AHook

_PROJECT = "test_ahook_isolated"

hookspec = pluggy.HookspecMarker(_PROJECT)
hookimpl = pluggy.HookimplMarker(_PROJECT)

# Hook-call remedy pattern: "pm.ahook" but not "pm.ahook_isolated".
_REMEDY = re.compile(r"pm\.ahook(?!_isolated)")


class Spec:
    """Hookspec namespace for the hooks that need a spec."""

    @hookspec(firstresult=True)
    async def first_hook(self) -> int | None: ...  # pragma: no cover


class _CustomBase(BaseException):
    """A BaseException subclass that is neither Exception nor KeyboardInterrupt."""


def _pm(*plugins: object, specs: bool = False) -> PluginManager:
    # Build a manager and register the plugins in the order given.
    pm = PluginManager(_PROJECT)
    if specs:
        pm.add_hookspecs(Spec)
    for plugin in plugins:
        pm.register(plugin)
    return pm


def _run_in_fresh_loop(make: Callable[[], Awaitable[Any]]) -> Any:
    # Run the awaitable in its own loop, so a KeyboardInterrupt that escapes the
    # loop surfaces as this test's own exception instead of wrecking the session.
    # No inner wait_for: on 3.10/3.11 it would add the very Task boundary these
    # tests probe; pytest-timeout bounds the test instead.
    async def run() -> Any:
        return await make()

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# Requirement 1: (name, Result) pairs in hook order
# ---------------------------------------------------------------------------


async def test_pairs_in_hook_order_with_names_values_and_exception() -> None:
    """tryfirst, plain, trylast give three pairs in that order, named by pm."""

    class TryFirst:
        @hookimpl(tryfirst=True)
        async def ping(self) -> str:
            # Slowest, so completion order differs from hook order.
            await asyncio.sleep(0.02)
            return "first"

    class Plain:
        @hookimpl
        async def ping(self) -> str:
            raise ValueError("plain failed")

    class TryLast:
        @hookimpl(trylast=True)
        async def ping(self) -> str:
            return "last"

    tryfirst, plain, trylast = TryFirst(), Plain(), TryLast()
    # Register out of hook order so registration order cannot explain the result.
    pm = _pm(trylast, plain, tryfirst)

    pairs = await asyncio.wait_for(pm.ahook_isolated.ping(), 2)

    assert [name for name, _ in pairs] == [
        pm.get_name(tryfirst),
        pm.get_name(plain),
        pm.get_name(trylast),
    ]
    assert pairs[0][1].get_result() == "first"
    assert isinstance(pairs[1][1].exception, ValueError)
    assert pairs[2][1].get_result() == "last"


# ---------------------------------------------------------------------------
# Requirement 2: concurrent, to completion, attributed by dispatch
# ---------------------------------------------------------------------------


async def test_impls_run_concurrently() -> None:
    """Two impls that each wait for the other to start both finish."""
    started = {"a": asyncio.Event(), "b": asyncio.Event()}

    class A:
        @hookimpl
        async def meet(self) -> str:
            started["a"].set()
            await asyncio.wait_for(started["b"].wait(), 2)
            return "a"

    class B:
        @hookimpl
        async def meet(self) -> str:
            started["b"].set()
            await asyncio.wait_for(started["a"].wait(), 2)
            return "b"

    pm = _pm(A(), B())

    pairs = await asyncio.wait_for(pm.ahook_isolated.meet(), 5)

    assert sorted(r.get_result() for _, r in pairs) == ["a", "b"]


async def test_raising_impl_does_not_stop_slower_sibling() -> None:
    """The slower sibling has finished by the time the call returns."""
    finished: list[str] = []

    class Raiser:
        @hookimpl(tryfirst=True)
        async def work(self) -> None:
            raise RuntimeError("boom")

    class Slow:
        @hookimpl
        async def work(self) -> str:
            await asyncio.sleep(0.05)
            finished.append("slow")
            return "slow done"

    raiser, slow = Raiser(), Slow()
    pm = _pm(raiser, slow)

    pairs = await asyncio.wait_for(pm.ahook_isolated.work(), 2)

    # Checked after the call returned: the sibling completed before then.
    assert finished == ["slow"]
    assert isinstance(pairs[0][1].exception, RuntimeError)
    assert pairs[1][1].get_result() == "slow done"


async def test_impl_returning_none_gets_its_own_pair() -> None:
    """A None return is kept (pluggy drops it) and stays attributed correctly."""

    class ReturnsNone:
        @hookimpl(tryfirst=True)
        async def maybe(self) -> None:
            return None

    class ReturnsValue:
        @hookimpl
        async def maybe(self) -> int:
            return 7

    none_plugin, value_plugin = ReturnsNone(), ReturnsValue()
    pm = _pm(none_plugin, value_plugin)

    pairs = await asyncio.wait_for(pm.ahook_isolated.maybe(), 2)

    assert [name for name, _ in pairs] == [
        pm.get_name(none_plugin),
        pm.get_name(value_plugin),
    ]
    assert pairs[0][1].get_result() is None
    assert pairs[1][1].get_result() == 7


# ---------------------------------------------------------------------------
# Requirement 3: BaseException propagates after every sibling finished
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_propagates_after_slower_sibling_finished() -> None:
    """KeyboardInterrupt reaches the caller only once the sibling is done."""
    finished: list[str] = []

    class Interrupter:
        @hookimpl(tryfirst=True)
        async def stop(self) -> None:
            raise KeyboardInterrupt

    class Slow:
        @hookimpl
        async def stop(self) -> None:
            await asyncio.sleep(0.05)
            finished.append("slow")

    pm = _pm(Interrupter(), Slow())

    with pytest.raises(KeyboardInterrupt):
        _run_in_fresh_loop(lambda: pm.ahook_isolated.stop())

    assert finished == ["slow"]


def test_keyboard_interrupt_with_timeout_propagates_after_sibling_finished() -> None:
    """Same with a timeout set (wait_for may add a Task boundary)."""
    finished: list[str] = []

    class Interrupter:
        @hookimpl(tryfirst=True)
        async def stop(self) -> None:
            raise KeyboardInterrupt

    class Slow:
        @hookimpl
        async def stop(self) -> None:
            await asyncio.sleep(0.05)
            finished.append("slow")

    pm = _pm(Interrupter(), Slow())
    hook = AHook(pm, return_exceptions=True, timeout=0.2)

    with pytest.raises(KeyboardInterrupt):
        _run_in_fresh_loop(lambda: hook.stop())

    assert finished == ["slow"]


def test_custom_base_exception_propagates_after_sibling_finished() -> None:
    """A custom BaseException subclass is raised once the sibling is done."""
    finished: list[str] = []

    class Raiser:
        @hookimpl(tryfirst=True)
        async def stop(self) -> None:
            raise _CustomBase("custom")

    class Slow:
        @hookimpl
        async def stop(self) -> None:
            await asyncio.sleep(0.05)
            finished.append("slow")

    pm = _pm(Raiser(), Slow())

    with pytest.raises(_CustomBase):
        _run_in_fresh_loop(lambda: pm.ahook_isolated.stop())

    assert finished == ["slow"]


# ---------------------------------------------------------------------------
# Requirement 4: per-implementation timeout
# ---------------------------------------------------------------------------


async def test_timed_out_impl_is_cancelled_and_sibling_unaffected() -> None:
    """The slow impl gets TimeoutError and sees CancelledError; fast is kept."""
    seen: list[str] = []

    class Slow:
        @hookimpl(tryfirst=True)
        async def race(self) -> None:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                seen.append("cancelled")
                raise

    class Fast:
        @hookimpl
        async def race(self) -> str:
            return "fast value"

    pm = _pm(Slow(), Fast())
    hook = AHook(pm, return_exceptions=True, timeout=0.05)

    pairs = await asyncio.wait_for(hook.race(), 2)

    assert isinstance(pairs[0][1].exception, asyncio.TimeoutError)
    assert seen == ["cancelled"]
    assert pairs[1][1].get_result() == "fast value"


def test_timeout_without_return_exceptions_is_rejected() -> None:
    """Construction names both the rejected and the required argument."""
    pm = _pm()

    with pytest.raises(ValueError) as excinfo:
        AHook(pm, timeout=0.05)

    message = str(excinfo.value)
    assert "timeout" in message
    assert "return_exceptions" in message


# ---------------------------------------------------------------------------
# Requirement 5: wrappers and firstresult are refused before anything runs
# ---------------------------------------------------------------------------


async def test_firstresult_hook_is_refused_naming_hook_and_remedy() -> None:
    """A firstresult spec raises HookCallError; no impl ran."""
    ran: list[str] = []

    class Impl:
        @hookimpl
        async def first_hook(self) -> int:
            ran.append("impl")
            return 1

    pm = _pm(Impl(), specs=True)

    with pytest.raises(HookCallError) as excinfo:
        await asyncio.wait_for(pm.ahook_isolated.first_hook(), 2)

    message = str(excinfo.value)
    assert "first_hook" in message
    assert _REMEDY.search(message)
    assert ran == []


async def test_wrapper_hook_is_refused_naming_hook_plugin_and_remedy() -> None:
    """A wrapper=True impl raises HookCallError naming its plugin; no impl ran."""
    ran: list[str] = []

    class Impl:
        @hookimpl
        async def wrapped(self) -> int:
            ran.append("impl")
            return 1

    class Wrapper:
        @hookimpl(wrapper=True)
        def wrapped(self):
            return (yield)

    pm = _pm(Impl())
    pm.register(Wrapper(), name="the_wrapper_plugin")

    with pytest.raises(HookCallError) as excinfo:
        await asyncio.wait_for(pm.ahook_isolated.wrapped(), 2)

    message = str(excinfo.value)
    assert "wrapped" in message
    assert "the_wrapper_plugin" in message
    assert _REMEDY.search(message)
    assert ran == []


async def test_hookwrapper_hook_is_refused_naming_hook_plugin_and_remedy() -> None:
    """A hookwrapper=True impl raises HookCallError naming its plugin."""
    ran: list[str] = []

    class Impl:
        @hookimpl
        async def old_wrapped(self) -> int:
            ran.append("impl")
            return 1

    class OldWrapper:
        @hookimpl(hookwrapper=True)
        def old_wrapped(self):
            yield

    pm = _pm(Impl())
    pm.register(OldWrapper(), name="the_old_wrapper_plugin")

    with pytest.raises(HookCallError) as excinfo:
        await asyncio.wait_for(pm.ahook_isolated.old_wrapped(), 2)

    message = str(excinfo.value)
    assert "old_wrapped" in message
    assert "the_old_wrapper_plugin" in message
    assert _REMEDY.search(message)
    assert ran == []


# ---------------------------------------------------------------------------
# Requirement 6: keyword arguments only; a missing argument fails up front
# ---------------------------------------------------------------------------


async def test_missing_argument_raises_before_any_impl_runs() -> None:
    """HookCallError names the missing argument; no impl ran."""
    ran: list[str] = []

    class NeedsChannel:
        @hookimpl
        async def need(self, channel: int) -> int:
            ran.append("needs_channel")
            return channel

    class NeedsNothing:
        @hookimpl(tryfirst=True)
        async def need(self) -> int:
            ran.append("needs_nothing")
            return 0

    pm = _pm(NeedsChannel(), NeedsNothing())

    with pytest.raises(HookCallError, match="channel"):
        await asyncio.wait_for(pm.ahook_isolated.need(chanel=1), 2)

    assert ran == []


# ---------------------------------------------------------------------------
# Requirement 7: a sync impl is not tolerated
# ---------------------------------------------------------------------------


async def test_sync_impl_gets_type_error_and_async_sibling_is_kept() -> None:
    """The sync impl's Result carries a TypeError; the async value survives."""

    class Sync:
        @hookimpl(tryfirst=True)
        def mixed(self) -> int:
            return 1

    class Async:
        @hookimpl
        async def mixed(self) -> int:
            return 2

    pm = _pm(Sync(), Async())

    pairs = await asyncio.wait_for(pm.ahook_isolated.mixed(), 2)

    assert isinstance(pairs[0][1].exception, TypeError)
    assert pairs[1][1].get_result() == 2
