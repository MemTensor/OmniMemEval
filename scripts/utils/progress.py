"""
Unified progress tracking using rich.progress.
Provides drop-in replacements for tqdm across the project.
"""

import asyncio
import time

from concurrent.futures import as_completed

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _fmt_duration(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def create_progress(**kwargs):
    """Create a configured rich Progress instance for manual use.

    Usage (async patterns or multi-task scenarios):
        with create_progress() as progress:
            task_id = progress.add_task("Working...", total=100)
            for coro in asyncio.as_completed(tasks):
                result = await coro
                progress.advance(task_id)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("→"),
        TimeRemainingColumn(),
        **kwargs,
    )


def track(iterable, *, description="Processing", total=None, transient=False):
    """Drop-in replacement for tqdm(iterable, desc=..., total=...).

    Usage:
        for item in track(items, description="Loading"):
            process(item)

        # With generators (must provide total):
        for future in track(as_completed(futures), description="Search", total=len(futures)):
            result = future.result()
    """
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)
    with create_progress(transient=transient) as progress:
        task_id = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task_id)


def track_concurrent(futures, total, desc="Processing", unit="item"):
    """Wrap ThreadPoolExecutor futures with a rich progress bar.

    Yields (success: bool, result_or_exception) for each completed future.
    After iteration, prints a summary line with timing and error count.

    Usage:
        with ThreadPoolExecutor(...) as executor:
            futures = [executor.submit(fn, arg) for arg in args]
            for ok, result in track_concurrent(futures, len(args), desc="Search"):
                if ok:
                    handle(result)
    """
    start = time.time()
    errors = 0

    with create_progress() as progress:
        task_id = progress.add_task(desc, total=total)
        for future in as_completed(futures):
            try:
                result = future.result()
                yield True, result
            except Exception as e:
                errors += 1
                yield False, e
            progress.advance(task_id)

    elapsed = time.time() - start
    rate = total / elapsed if elapsed > 0 else 0
    parts = [
        f"  ✓ {desc}: {total - errors}/{total}",
        f"in {_fmt_duration(elapsed)}",
        f"({rate:.1f} {unit}/s)",
    ]
    if errors:
        parts.append(f"| {errors} errors")
    print("  ".join(parts))


async def track_async(coros, total, desc="Processing", unit="item"):
    """Wrap asyncio tasks with a rich progress bar.

    Yields (success: bool, result_or_exception) for each completed task.

    Usage:
        async for ok, result in track_async(tasks, len(tasks), desc="Eval"):
            if ok:
                handle(result)
    """
    start = time.time()
    errors = 0

    with create_progress() as progress:
        task_id = progress.add_task(desc, total=total)
        for coro in asyncio.as_completed(coros):
            try:
                result = await coro
                yield True, result
            except Exception as e:
                errors += 1
                yield False, e
            progress.advance(task_id)

    elapsed = time.time() - start
    rate = total / elapsed if elapsed > 0 else 0
    parts = [
        f"  ✓ {desc}: {total - errors}/{total}",
        f"in {_fmt_duration(elapsed)}",
        f"({rate:.1f} {unit}/s)",
    ]
    if errors:
        parts.append(f"| {errors} errors")
    print("  ".join(parts))
