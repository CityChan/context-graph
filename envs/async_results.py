"""Deliver search results from worker threads to the owning asyncio loop."""


def deliver_result(future, result):
    def resolve():
        # Cancellation may happen after scheduling and before this callback.
        if not future.done():
            future.set_result(result)

    future.get_loop().call_soon_threadsafe(resolve)
