"""Expose a loopback-only host proxy to containers on the Docker bridge."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging


LOGGER = logging.getLogger("proxy_bridge")


async def copy_stream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(ConnectionError, RuntimeError):
            writer.write_eof()


async def relay(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        target_reader, target_writer = await asyncio.open_connection(
            target_host,
            target_port,
        )
    except OSError as exc:
        LOGGER.warning("Cannot connect to proxy target for %s: %s", peer, exc)
        client_writer.close()
        await client_writer.wait_closed()
        return

    try:
        await asyncio.gather(
            copy_stream(client_reader, target_writer),
            copy_stream(target_reader, client_writer),
        )
    finally:
        target_writer.close()
        client_writer.close()
        await asyncio.gather(
            target_writer.wait_closed(),
            client_writer.wait_closed(),
            return_exceptions=True,
        )


async def serve(args: argparse.Namespace) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: relay(
            reader,
            writer,
            args.target_host,
            args.target_port,
        ),
        args.listen_host,
        args.listen_port,
    )
    LOGGER.info(
        "Forwarding %s:%s to %s:%s",
        args.listen_host,
        args.listen_port,
        args.target_host,
        args.target_port,
    )
    async with server:
        await server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", required=True, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        asyncio.run(serve(parse_args()))
    except KeyboardInterrupt:
        pass
