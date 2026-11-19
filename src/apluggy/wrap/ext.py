import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Coroutine

from pluggy import HookCaller, HookCallError, Result
from pluggy import PluginManager as PluginManager_

from apluggy.stack import AGenCtxMngr, GenCtxMngr, async_stack_gen_ctxs, stack_gen_ctxs

if TYPE_CHECKING:
    from pluggy._hooks import HookImpl


def _impl_kwargs(
    hook_impl: "HookImpl", caller_kwargs: dict[str, Any]
) -> dict[str, Any]:
    # Select only the args this hookimpl declares. A missing one is a caller
    # error, so raise what pluggy's _multicall raises rather than a bare KeyError.
    try:
        return {k: caller_kwargs[k] for k in hook_impl.argnames}
    except KeyError as e:
        raise HookCallError(f"hook call must provide argument {e.args[0]!r}") from e


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
        impl_kwargs = _impl_kwargs(hook_impl, caller_kwargs)

        # Invoke the hookimpl function to get a coroutine, then await it.
        result = await hook_impl.function(**impl_kwargs)

        # First non-None result stops the chain (firstresult semantics).
        if result is not None:
            return result

    # All implementations returned None; return None to the caller.
    return None


async def _async_multicall(
    hook_name: str,
    hookimpls: list["HookImpl"],
    caller_kwargs: dict[str, Any],
    firstresult: bool,
) -> Any:
    """Drive async hook dispatch for hooks that include wrapper=True hookimpls.

    Mirrors pluggy's _multicall (_callers.py:76-169) but handles async
    non-wrappers and sync generator wrappers. Wrappers are set up first
    (pre-yield code runs during setup), then non-wrappers are awaited
    (concurrently for firstresult=False, as without wrappers; in order for
    firstresult=True, stopping at the first non-None result), then
    wrappers are torn down in reverse order (post-yield code runs, result or
    exception is passed back via send/throw).

    Wrappers must be sync generators — they cannot perform async I/O. The
    async I/O happens in non-wrapper hookimpls, which this function awaits.

    Args:
        hook_name: Name of the hook, used in error messages.
        hookimpls: All hookimpls for this hook, in pluggy storage order.
                   This function iterates them in reverse (execution order).
        caller_kwargs: Keyword arguments passed by the caller.
        firstresult: If True, stop after the first non-None result from
                     non-wrappers; aggregate is a single value. If False,
                     collect all results into a list.
    """
    # teardowns holds sync generators for each wrapper hookimpl, in the order
    # they were set up (outermost first). Teardown iterates this in reverse so
    # innermost teardown runs first.
    teardowns: list[Any] = []

    # results accumulates return values from non-wrapper hookimpls.
    results: list[Any] = []

    # pending holds broadcast (firstresult=False) non-wrappers, as function and
    # kwargs, to be gathered concurrently once every wrapper is set up.
    pending: list[tuple[Callable[..., Any], dict[str, Any]]] = []

    # exception captures any exception raised during setup or invocation, so
    # teardown wrappers can receive it via throw().
    exception: BaseException | None = None

    try:
        # -----------------------------------------------------------------------
        # Setup phase: iterate hookimpls in pluggy execution order (reversed).
        # Wrappers run their pre-yield code; non-wrappers are awaited.
        # -----------------------------------------------------------------------
        for hook_impl in reversed(hookimpls):
            # Build kwargs by selecting only the args this hookimpl declares.
            # pluggy strips 'self' from argnames, so we never need to pass it.
            impl_kwargs = _impl_kwargs(hook_impl, caller_kwargs)

            if hook_impl.hookwrapper:
                # Old-style hookwrapper=True is not supported in this dispatch path.
                raise NotImplementedError(
                    f"{hook_name}: hookwrapper=True hookimpls are not supported "
                    "by _async_multicall; use wrapper=True instead"
                )

            if hook_impl.wrapper:
                # Wrapper hookimpl: call function to get a sync generator,
                # then advance to its first yield (running pre-yield code).
                gen = hook_impl.function(**impl_kwargs)

                try:
                    # Advance the generator to its yield point.
                    # If it raises StopIteration, it never yielded — programming error.
                    next(gen)
                except StopIteration:
                    raise RuntimeError(
                        f"{hook_name}: wrapper {hook_impl.function!r} did not yield"
                    )

                # Push the generator onto the teardown stack for post-yield processing.
                teardowns.append(gen)

            elif not firstresult:
                # Broadcast non-wrapper: defer it so all of them can be gathered
                # concurrently below, as on the no-wrapper path. The coroutine is
                # not created yet, so a later wrapper's setup failure leaves no
                # never-awaited coroutine behind.
                pending.append((hook_impl.function, impl_kwargs))

            else:
                # firstresult non-wrapper: await in order and stop at the first
                # non-None result, so these must stay sequential.
                result = await hook_impl.function(**impl_kwargs)
                results.append(result)
                if result is not None:
                    break

        # -----------------------------------------------------------------------
        # Broadcast invocation: run every deferred non-wrapper concurrently.
        # pluggy stores wrappers after non-wrappers, so every wrapper's pre-yield
        # code has already run by now, matching the sequential order. Each one
        # runs to completion (return_exceptions=True): a raising implementation
        # neither cancels nor orphans its siblings. The first exception in
        # execution order is then thrown into the wrappers.
        # -----------------------------------------------------------------------
        if pending:
            outcomes = await asyncio.gather(
                *(func(**kwargs) for func, kwargs in pending),
                return_exceptions=True,
            )
            exception = next(
                (o for o in outcomes if isinstance(o, BaseException)), None
            )
            if exception is None:
                results = list(outcomes)

    except BaseException as e:
        # Any exception during setup or invocation (from a wrapper's pre-yield
        # code or from a non-wrapper) is captured here. Teardown will throw it
        # into any already-pushed wrappers.
        exception = e

    # -----------------------------------------------------------------------
    # Compute aggregate result from non-wrapper invocations.
    # -----------------------------------------------------------------------
    if firstresult:
        # Return the single first non-None result, or None if all returned None.
        aggregate: Any = next((r for r in results if r is not None), None)
    else:
        # Return all collected results as a list.
        aggregate = results

    # -----------------------------------------------------------------------
    # Teardown phase: drive wrappers in reverse setup order (innermost first).
    # Send the aggregate result (or throw any exception) into each wrapper's
    # generator to run its post-yield code.
    # -----------------------------------------------------------------------
    for gen in reversed(teardowns):
        try:
            if exception is not None:
                # An exception is in flight — throw it into the wrapper generator.
                # The wrapper can catch it, recover, and return a new result, or
                # re-raise (or raise a new exception), which replaces 'exception'.
                try:
                    gen.throw(exception)
                except RuntimeError as re:
                    # Python 3.7+: if the thrown exception was a StopIteration,
                    # Python wraps it in RuntimeError inside the generator body.
                    # Match pluggy's handling at _callers.py:140-148.
                    if isinstance(exception, StopIteration) and re.__cause__ is exception:
                        # Close the generator cleanly and continue to the next wrapper.
                        gen.close()
                        continue
                    else:
                        # A genuine RuntimeError from the wrapper — let it replace
                        # the current exception in the outer except BaseException block.
                        raise
            else:
                # No exception: send the current aggregate result into the wrapper.
                gen.send(aggregate)

            # If we reach here without an exception, the generator yielded again,
            # which is illegal (wrappers must yield exactly once).
            gen.close()
            raise RuntimeError(f"{hook_name}: wrapper yielded twice")

        except StopIteration as si:
            # Normal success path: the wrapper returned a value via 'return'.
            # Capture the returned value as the new aggregate result.
            aggregate = si.value
            exception = None  # wrapper handled any in-flight exception

        except BaseException as e:
            # Wrapper raised during teardown (re-raised or raised a new exception).
            # This replaces the current exception; continue teardown to the next wrapper.
            exception = e
            continue

    # -----------------------------------------------------------------------
    # Final result: raise any surviving exception, or return the aggregate.
    # -----------------------------------------------------------------------
    if exception is not None:
        raise exception

    return aggregate


class _Escaped:
    """A KeyboardInterrupt or SystemExit caught inside one implementation.

    asyncio's task step re-raises these two straight out of the event loop
    instead of storing them on the task, so gather(return_exceptions=True)
    would never see them and the siblings would be abandoned. Boxing them as a
    plain value lets every sibling finish before _call_isolated re-raises.
    """

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


async def _call_isolated(
    hook: HookCaller,
    caller_kwargs: dict[str, Any],
    timeout: float | None,
) -> list[tuple[str, Result]]:
    """Run every hookimpl to completion and report each one's outcome.

    Returns one (plugin_name, pluggy.Result) pair per implementation, in
    pluggy's execution order (tryfirst first, trylast last). Each Result holds
    the implementation's return value or the Exception it raised; with a
    timeout, an implementation that overruns is cancelled and its Result holds
    asyncio.TimeoutError. A BaseException that is not an Exception (e.g.
    CancelledError, KeyboardInterrupt) is raised once every sibling has
    finished, the first in hook order if several.

    Like the firstresult and wrapper paths, this skips pluggy's _hookexec, so
    hook-call monitoring and tracing don't see these calls.

    Raises:
        HookCallError: the hook is firstresult=True, has a wrapper=True or
            hookwrapper=True implementation, or the call omits an argument an
            implementation needs. Nothing has run when this is raised.
    """
    hookimpls: list[HookImpl] = hook.get_hookimpls()

    # Refuse hooks this mode can't drive faithfully, before anything runs:
    # firstresult stops at the first value, and a wrapper expects an exception
    # thrown into it, which this mode reports instead of raising.
    if hook.spec and hook.spec.opts.get("firstresult", False):
        raise HookCallError(
            f"{hook.name}: return_exceptions=True can't run a firstresult=True "
            "hook; call it through pm.ahook"
        )
    for hook_impl in hookimpls:
        if hook_impl.wrapper or hook_impl.hookwrapper:
            option = "wrapper" if hook_impl.wrapper else "hookwrapper"
            raise HookCallError(
                f"{hook.name}: return_exceptions=True can't run the {option}=True "
                f"implementation in plugin {hook_impl.plugin_name!r}; call the "
                "hook through pm.ahook"
            )

    # Warn about omitted spec args as HookCaller.__call__ would, then bind
    # every implementation's kwargs eagerly, so a missing argument raises
    # before any implementation is called.
    hook._verify_all_args_are_provided(caller_kwargs)
    calls = [
        (hook_impl, _impl_kwargs(hook_impl, caller_kwargs))
        for hook_impl in reversed(hookimpls)
    ]

    async def invoke(hook_impl: "HookImpl", impl_kwargs: dict[str, Any]) -> Any:
        # The innermost coroutine: call the implementation here, so a sync
        # implementation's raise or non-awaitable return is its own outcome,
        # and box KeyboardInterrupt/SystemExit here, inside any Task that
        # wait_for creates, so they can't escape the loop.
        try:
            return await hook_impl.function(**impl_kwargs)
        except (KeyboardInterrupt, SystemExit) as e:
            return _Escaped(e)

    async def run(hook_impl: "HookImpl", impl_kwargs: dict[str, Any]) -> Any:
        # Apply the per-implementation time limit, if any.
        if timeout is None:
            return await invoke(hook_impl, impl_kwargs)
        return await asyncio.wait_for(invoke(hook_impl, impl_kwargs), timeout)

    # Run all implementations concurrently; return_exceptions lets each one
    # finish whatever its siblings do.
    outcomes = await asyncio.gather(
        *(run(hook_impl, impl_kwargs) for hook_impl, impl_kwargs in calls),
        return_exceptions=True,
    )

    # Every sibling has finished: re-raise the first BaseException that is not
    # an Exception, in hook order.
    for outcome in outcomes:
        if isinstance(outcome, _Escaped):
            raise outcome.exc
        if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
            raise outcome

    # Pair each outcome with the plugin whose implementation produced it.
    return [
        (
            hook_impl.plugin_name,
            Result(None, outcome)
            if isinstance(outcome, BaseException)
            else Result(outcome, None),
        )
        for (hook_impl, _), outcome in zip(calls, outcomes)
    ]


class AHook:
    """Call async hooks: ``await pm.ahook.<hook>(**kwargs)``.

    With ``return_exceptions=True`` (``pm.ahook_isolated``), a call runs every
    implementation to completion and returns ``(plugin_name, pluggy.Result)``
    pairs in hook order instead of raising the first exception. Unlike
    asyncio's flag of the same name, the results are attributed Result
    objects, not bare exceptions. ``timeout`` (seconds) then limits each
    implementation separately. See _call_isolated.
    """

    def __init__(
        self,
        pm: PluginManager_,
        return_exceptions: bool = False,
        timeout: float | None = None,
    ) -> None:
        # A timeout is only applied per implementation in isolated mode; on
        # the default path it would silently do nothing.
        if timeout is not None and not return_exceptions:
            raise ValueError(
                "AHook: timeout requires return_exceptions=True; pass both, or "
                "drop timeout"
            )
        self.pm = pm
        # Underscored so they never shadow a hook of the same name.
        self._return_exceptions = return_exceptions
        self._timeout = timeout

    def __getattr__(self, name: str) -> Callable[..., Coroutine[Any, Any, Any]]:
        if self._return_exceptions:

            async def call_isolated(**kwargs: Any) -> list[tuple[str, Result]]:
                # Keyword arguments only, as pluggy's HookCaller requires.
                hook: HookCaller = getattr(self.pm.hook, name)
                return await _call_isolated(hook, kwargs, self._timeout)

            return call_isolated

        async def call(*args: Any, **kwargs: Any) -> Any:
            # Resolve the named HookCaller from pluggy's hook namespace.
            hook: HookCaller = getattr(self.pm.hook, name)

            # Check whether this hookspec uses firstresult semantics.
            # A missing spec means no hookspec was registered; treat as False.
            firstresult: bool = bool(
                hook.spec and hook.spec.opts.get("firstresult", False)
            )

            # Check whether any hookimpl is a wrapper (wrapper=True or hookwrapper=True).
            # When wrappers are present, we must use _async_multicall to drive the
            # sync generator teardown protocol.
            hookimpls = hook.get_hookimpls()
            has_wrappers = any(hi.wrapper or hi.hookwrapper for hi in hookimpls)

            if has_wrappers:
                # Wrapper path: drive setup, invocation, and teardown manually.
                # Handles both firstresult=True and firstresult=False.
                # Skips HookCaller.__call__, so warn about omitted spec args here.
                hook._verify_all_args_are_provided(kwargs)
                return await _async_multicall(name, hookimpls, kwargs, firstresult)
            elif firstresult:
                # firstresult=True, no wrappers: bypass _multicall; iterate impls
                # sequentially. _multicall returns a single coroutine for these
                # hooks, which is not iterable and would crash asyncio.gather.
                # Positional args are intentionally not forwarded here — pluggy
                # hookspecs use keyword arguments exclusively.
                # Skips HookCaller.__call__, so warn about omitted spec args here.
                hook._verify_all_args_are_provided(kwargs)
                return await _call_firstresult(hook, kwargs)
            else:
                # firstresult=False, no wrappers: collect unawaited coroutines from
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
