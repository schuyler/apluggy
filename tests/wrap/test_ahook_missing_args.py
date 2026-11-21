"""Tests for ``pm.ahook`` calls that omit an argument (issue #4).

A call that omits an argument an implementation needs must raise
``HookCallError: hook call must provide argument '<name>'`` on every dispatch
path, not a bare ``KeyError``. When the hook has a spec, pluggy's ``UserWarning``
about spec arguments missing from the call is emitted as well.
"""

import gc
import warnings

import pytest
from pluggy import HookCallError

import apluggy as pluggy
from apluggy import PluginManager

_PROJECT = "test_ahook_missing_args"

hookspec = pluggy.HookspecMarker(_PROJECT)
hookimpl = pluggy.HookimplMarker(_PROJECT)

_ERROR_MATCH = "hook call must provide argument 'channel'"
_WARNING_MATCH = "which are declared in the hookspec cannot be found in this hook call"


class Spec:
    @hookspec(firstresult=True)
    async def first_plain(self, channel: int) -> int | None: ...  # pragma: no cover

    @hookspec(firstresult=True)
    async def first_wrapped(self, channel: int) -> int | None: ...  # pragma: no cover

    @hookspec
    async def all_wrapped(self, channel: int) -> int: ...  # pragma: no cover

    @hookspec(firstresult=True)
    async def extra_first(self, channel: int, extra: int) -> int | None: ...  # pragma: no cover

    @hookspec
    async def extra_wrapped(self, channel: int, extra: int) -> int: ...  # pragma: no cover


async def test_firstresult_missing_arg_raises_hook_call_error() -> None:
    class Plugin:
        @hookimpl
        async def first_plain(self, channel: int) -> int:
            return channel

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    pm.register(Plugin())

    with pytest.warns(UserWarning, match=_WARNING_MATCH):
        with pytest.raises(HookCallError, match=_ERROR_MATCH):
            await pm.ahook.first_plain(chanel=1)


async def test_wrapper_firstresult_missing_arg_raises_hook_call_error() -> None:
    class Plugin:
        @hookimpl
        async def first_wrapped(self, channel: int) -> int:
            return channel

    class Wrapper:
        @hookimpl(wrapper=True)
        def first_wrapped(self):
            return (yield)

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    pm.register(Plugin())
    pm.register(Wrapper())

    with pytest.warns(UserWarning, match=_WARNING_MATCH):
        with pytest.raises(HookCallError, match=_ERROR_MATCH):
            await pm.ahook.first_wrapped(chanel=1)


async def test_wrapper_broadcast_missing_arg_raises_hook_call_error() -> None:
    class Plugin:
        @hookimpl
        async def all_wrapped(self, channel: int) -> int:
            return channel

    class Wrapper:
        @hookimpl(wrapper=True)
        def all_wrapped(self):
            return (yield)

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    pm.register(Plugin())
    pm.register(Wrapper())

    with pytest.warns(UserWarning, match=_WARNING_MATCH):
        with pytest.raises(HookCallError, match=_ERROR_MATCH):
            await pm.ahook.all_wrapped(chanel=1)


async def test_specless_wrapper_missing_arg_raises_without_warning() -> None:
    """No hookspec is registered, so there is nothing to warn against."""

    class Plugin:
        @hookimpl
        async def specless(self, channel: int) -> int:
            return channel

    class Wrapper:
        @hookimpl(wrapper=True)
        def specless(self):
            return (yield)

    pm = PluginManager(_PROJECT)
    pm.register(Plugin())
    pm.register(Wrapper())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(HookCallError, match=_ERROR_MATCH):
            await pm.ahook.specless(chanel=1)

    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


async def test_firstresult_unneeded_spec_arg_missing_warns_and_returns() -> None:
    """The spec declares ``extra`` but no impl needs it: the call still works."""

    class Plugin:
        @hookimpl
        async def extra_first(self, channel: int) -> int:
            return channel + 1

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    pm.register(Plugin())

    with pytest.warns(UserWarning, match=_WARNING_MATCH):
        result = await pm.ahook.extra_first(channel=1)

    assert result == 2


async def test_wrapper_unneeded_spec_arg_missing_warns_and_returns() -> None:
    class Plugin:
        @hookimpl
        async def extra_wrapped(self, channel: int) -> int:
            return channel + 1

    class Wrapper:
        @hookimpl(wrapper=True)
        def extra_wrapped(self):
            return (yield)

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    pm.register(Plugin())
    pm.register(Wrapper())

    with pytest.warns(UserWarning, match=_WARNING_MATCH):
        result = await pm.ahook.extra_wrapped(channel=1)

    assert result == [2]


async def test_broadcast_missing_arg_leaves_no_unawaited_coroutine() -> None:
    """The plain broadcast path (no wrappers, firstresult=False) built every
    implementation's coroutine through pluggy's ``_multicall`` and only then hit
    the missing argument, so the ones already built were never awaited and Python
    complained at collection time.

    One implementation can be called without the omitted argument and one cannot,
    which is what leaves a coroutine behind.
    """

    class Plugin:
        @hookimpl
        async def all_plain(self, channel: int) -> int:
            return channel

    class Needy:
        @hookimpl
        async def all_plain(self, channel: int, extra: int) -> int:
            return channel + extra

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    # LIFO: Plugin is called first, so its coroutine exists when Needy raises.
    pm.register(Needy())
    pm.register(Plugin())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(HookCallError, match="hook call must provide argument 'extra'"):
            await pm.ahook.all_plain(channel=1)
        gc.collect()

    leaked = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert leaked == [], [str(w.message) for w in leaked]


async def test_a_sync_impl_returning_none_is_skipped_like_multicall() -> None:
    """A non-firstresult impl returning None contributes no result (#10 review).

    pluggy's ``_multicall`` drops that None, so the broadcast must too: handing
    it to ``asyncio.gather`` raised "an asyncio.Future, a coroutine or an
    awaitable is required" on a path that worked before the pre-binding change.
    """

    class Spec:
        @hookspec
        async def all_plain(self, channel: int) -> int: ...  # pragma: no cover

    class SyncNone:
        @hookimpl
        def all_plain(self, channel: int):
            return None

    class AsyncX:
        @hookimpl
        async def all_plain(self, channel: int):
            return channel

    class AsyncSum:
        @hookimpl
        async def all_plain(self, channel: int, extra: int = 5):
            return channel + extra

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    pm.register(SyncNone(), name="syncnone")
    pm.register(AsyncX(), name="asyncx")
    pm.register(AsyncSum(), name="asyncsum")

    # Registration order reversed, as pluggy dispatches: sum, x, sync-none.
    assert await pm.ahook.all_plain(channel=1) == [6, 1]


async def test_the_broadcast_applies_the_spec_defaults() -> None:
    """An omitted argument takes the *hookspec's* default, as HookCaller does.

    Behaviour inherited from `_multicall`; the pre-binding path lost it and
    raised HookCallError for an implementation that has no default of its own.
    """

    class Spec:
        @hookspec
        async def defaults(self, channel: int, mode: str = "fast") -> tuple: ...  # pragma: no cover

    class NoOwnDefault:
        @hookimpl
        async def defaults(self, channel: int, mode):
            return ("impl", channel, mode)

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    pm.register(NoOwnDefault(), name="nodefault")

    assert await pm.ahook.defaults(channel=1) == [("impl", 1, "fast")]
    assert await pm.ahook.defaults(channel=1, mode="slow") == [("impl", 1, "slow")]


async def test_an_async_return_of_none_keeps_its_slot() -> None:
    """Only a call-time None is dropped; an awaited None is a result (#10 review).

    pluggy filters on the coroutine object, which is never None, so every async
    implementation contributes a slot.
    """

    class Spec:
        @hookspec
        async def slots(self, x: int) -> int: ...  # pragma: no cover

    class Ten:
        @hookimpl
        async def slots(self, x: int):
            return x * 10

    class NoneResult:
        @hookimpl
        async def slots(self, x: int):
            return None

    class One:
        @hookimpl
        async def slots(self, x: int):
            return x

    pm = PluginManager(_PROJECT)
    pm.add_hookspecs(Spec)
    pm.register(Ten(), name="ten")
    pm.register(NoneResult(), name="none")
    pm.register(One(), name="one")

    # Reversed registration order, as pluggy dispatches.
    assert await pm.ahook.slots(x=1) == [1, None, 10]
