"""Pure conversation packing that never separates tool calls from their replies."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


@dataclass(frozen=True)
class PackedHistory:
    """The retained history and complete request additions, with packing counts."""

    history: list[BaseMessage]
    messages: list[BaseMessage]
    removed_messages: int
    removed_exchanges: int
    fits: bool


def _exchanges(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """Group assistants with all their contiguous tool replies; reject bad history.

    This runs only at an invocation boundary, when dispatch has replied to every
    advertised call. Recovered malformed calls use their history-safe arguments
    but retain the original IDs, so they receive the same pairing guarantees.
    """
    groups: list[list[BaseMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, ToolMessage):
            raise ValueError('Conversation contains an orphan tool reply')
        group = [message]
        index += 1
        if isinstance(message, AIMessage):
            calls = [*message.tool_calls, *message.invalid_tool_calls]
            expected = [call.get('id') for call in calls]
            if len(set(expected)) != len(expected):
                raise ValueError('Conversation contains duplicate tool-call IDs')
            received = []
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                reply = messages[index]
                received.append(reply.tool_call_id)
                group.append(reply)
                index += 1
            if len(received) != len(expected) or set(received) != set(expected):
                raise ValueError('Conversation tool calls and replies do not match')
        groups.append(group)
    return groups


def pack_history(
    messages: Sequence[BaseMessage],
    *,
    max_messages: Optional[int] = None,
    fits: Optional[Callable[[list[BaseMessage]], bool]] = None,
    pending_messages: Sequence[BaseMessage] = (),
    force_minimal: bool = False,
) -> PackedHistory:
    """Keep the newest fitting suffix of complete exchanges without mutation.

    The opening exchange and newest assistant exchange (plus any following
    messages) are mandatory. ``max_messages`` is a secondary limit on retained
    history, not on mandatory request additions; mandatory exchanges may exceed
    it. ``pending_messages`` supplies current state or active advice on every
    request and remains mandatory even when history is removed.

    ``fits`` must account for the entire candidate request, including any system
    message or tool definitions that the caller adds. It should be monotonic as
    old exchanges are removed. Binary search limits expensive provider counts;
    the final callback always corresponds to the returned candidate. If even the
    mandatory request is oversized, return it with ``fits=False`` for the caller
    to reject explicitly. ``force_minimal`` supports one bounded overflow retry.

    Returned ``history`` excludes ephemeral additions and can be persisted by the
    caller. Message objects are retained by identity and are never modified.
    """
    if max_messages is not None and max_messages < 0:
        raise ValueError('max_messages must be nonnegative or None')

    groups = _exchanges(messages)
    # Pin the opening plus the most recent assistant and all messages after it.
    latest = max((index for index, group in enumerate(groups)
                  if isinstance(group[0], AIMessage)), default=len(groups) - 1)
    mandatory_start = max(1, latest)
    optional = groups[1:mandatory_start]
    opening = groups[:1]
    tail = groups[mandatory_start:]

    removed_sizes = [0]
    for group in optional:
        removed_sizes.append(removed_sizes[-1] + len(group))
    remove = 0
    if max_messages is not None:
        while (remove < len(optional)
               and len(messages) - removed_sizes[remove] > max_messages):
            remove += 1
    if force_minimal:
        remove = len(optional)

    def candidate(n_removed):
        retained = [message for group in (*opening, *optional[n_removed:], *tail)
                    for message in group]
        return retained, [*retained, *pending_messages]

    last_removed = None
    last_fits = True

    def check(n_removed):
        nonlocal last_removed, last_fits
        last_removed = n_removed
        last_fits = bool(fits(candidate(n_removed)[1])) if fits else True
        return last_fits

    if not check(remove):
        maximum = len(optional)
        if remove < maximum and check(maximum):
            low, high = remove + 1, maximum
            while low < high:
                middle = (low + high) // 2
                if check(middle):
                    high = middle
                else:
                    low = middle + 1
            remove = low
        else:
            remove = maximum
    if last_removed != remove:
        check(remove)

    history, request = candidate(remove)
    return PackedHistory(
        history=history,
        messages=request,
        removed_messages=removed_sizes[remove],
        removed_exchanges=remove,
        fits=last_fits,
    )
