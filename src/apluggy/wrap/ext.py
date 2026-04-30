import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Coroutine

from pluggy import HookCaller
from pluggy import PluginManager as PluginManager_

from apluggy.stack import AGenCtxMngr, GenCtxMngr, async_stack_gen_ctxs, stack_gen_ctxs

if TYPE_CHECKING:
    from pluggy._hooks import HookImpl


async def _call_firstresult(
    hook: HookCaller,
    caller_kwargs: dict[str, Any],
) -> Any:
    """Await hookimpls in pluggy order, returning the first non-None result.

    Pluggy's _multicall returns a single coroutine for firstresult=True hooks,
    which is not iterable and cannot be passed to asyncio.gather. This function
    bypasses _multicall entirely and drives dispatch manually.
    """
    # Retrieve implementations in pluggy's storage order, then iterate in
    # reverse to match pluggy's _multicall execution order:
    # tryfirst first, then plain impls in registration order, trylast last.
    hookimpls: list[HookImpl] = hook.get_hookimpls()

    for hook_impl in reversed(hookimpls):
        # Build kwargs from argnames only, matching pluggy's _multicall behavior.
        # kwargnames is intentionally excluded — pluggy does not pass keyword-only
        # parameters through hook dispatch.
        impl_kwargs = {k: caller_kwargs[k] for k in hook_impl.argnames}

        # Invoke the hookimpl function to get a coroutine, then await it.
        result = await hook_impl.function(**impl_kwargs)

        # First non-None result stops the chain (firstresult semantics).
        if result is not None:
            return result

    # All implementations returned None; return None to the caller.
    return None


class AHook:
    def __init__(self, pm: PluginManager_) -> None:
        self.pm = pm

    def __getattr__(self, name: str) -> Callable[..., Coroutine[Any, Any, Any]]:
        async def call(*args: Any, **kwargs: Any) -> Any:
            # Resolve the named HookCaller from pluggy's hook namespace.
            hook: HookCaller = getattr(self.pm.hook, name)

            # Check whether this hookspec uses firstresult semantics.
            # A missing spec means no hookspec was registered; treat as False.
            firstresult: bool = bool(
                hook.spec and hook.spec.opts.get("firstresult", False)
            )

            if firstresult:
                # firstresult=True: bypass _multicall; iterate impls sequentially.
                # _multicall returns a single coroutine for these hooks, which is
                # not iterable and would crash asyncio.gather.
                # Positional args are intentionally not forwarded here — pluggy
                # hookspecs use keyword arguments exclusively, and the gather path
                # (below) only passes *args for backward compatibility with
                # HookCaller.__call__.
                return await _call_firstresult(hook, kwargs)
            else:
                # firstresult=False (default): collect unawaited coroutines from
                # pluggy's _multicall and run them concurrently.
                coros: list[asyncio.Future] = hook(*args, **kwargs)
                return await asyncio.gather(*coros)

        return call


class With:
    def __init__(self, pm: PluginManager_, reverse: bool = False) -> None:
        self.pm = pm
        self.reverse = reverse

    def __getattr__(self, name: str) -> Callable[..., GenCtxMngr[list]]:
        hook: HookCaller = getattr(self.pm.hook, name)

        def call(*args: Any, **kwargs: Any) -> GenCtxMngr[list]:
            ctxs = hook(*args, **kwargs)
            if self.reverse:
                ctxs = list(reversed(ctxs))
            return stack_gen_ctxs(ctxs)

        return call


class AWith:
    def __init__(self, pm: PluginManager_, reverse: bool = False) -> None:
        self.pm = pm
        self.reverse = reverse

    def __getattr__(self, name: str) -> Callable[..., AGenCtxMngr]:
        hook: HookCaller = getattr(self.pm.hook, name)

        def call(*args: Any, **kwargs: Any) -> AGenCtxMngr[list]:
            ctxs = hook(*args, **kwargs)
            if self.reverse:
                ctxs = list(reversed(ctxs))
            return async_stack_gen_ctxs(ctxs)

        return call
