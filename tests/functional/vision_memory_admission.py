#!/usr/bin/env python3
"""Owner-run Flash-Next vision admission qualification; requires local GPU/weights.

Owns a separate loopback server and a new disposable cache under --output.
Never downloads models, changes host settings or stops an existing process.
Not part of CPU CI. Numerical image oracles remain separate qualification.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import re
import socket
import struct
import threading
import time
import zlib

from run import option, provenance, server, write_json


def png_url(side, rgb):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))
    pixels = (b"\0" + bytes(rgb) * side) * side
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def request(port, path, body=None, timeout=14400):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    started = time.monotonic()
    try:
        connection.request("POST" if body is not None else "GET", path,
                           None if body is None else json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read().decode()
        value = json.loads(raw) if "json" in response.getheader("Content-Type", "") else raw
        return {"status": response.status, "elapsed_ms": 1000 * (time.monotonic() - started),
                "headers": dict(response.getheaders()), "response": value}
    finally:
        connection.close()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def process_memory(pid):
    values = {}
    for label, path in (("host", Path("/proc/meminfo")),
                        ("process", Path(f"/proc/{pid}/status"))):
        if path.is_file():
            values[label] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                             for line in path.read_text().splitlines()
                             if line.startswith(("MemAvailable:", "SwapTotal:", "SwapFree:",
                                                 "VmRSS:", "VmHWM:", "VmSwap:"))}
    values["drm"] = {str(path): path.read_text().strip()
                     for path in Path("/sys/class/drm").glob("card*/device/mem_info_*_used")}
    return values


class Sampler:
    def __init__(self, port, pid, path):
        self.port, self.pid, self.path = port, pid, path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open("w") as output:
            while not self.stop.is_set():
                sample = {"time": time.time()}
                try:
                    sample.update(process_memory(self.pid))
                    sample["slots"] = request(self.port, "/slots", timeout=10)["response"]
                    sample["metrics"] = request(self.port, "/metrics", timeout=10)["response"]
                except (OSError, ValueError, http.client.HTTPException) as error:
                    sample["poll_error"] = str(error)
                output.write(json.dumps(sample) + "\n")
                output.flush()
                self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=25)

    def records(self, since=0):
        with self.path.open() as source:
            for line in source:
                if not line.endswith("\n"):
                    break  # a concurrent sampler write has not finished yet
                sample = json.loads(line)
                if sample["time"] >= since:
                    yield sample


def concurrent_progress(samples):
    """Require all four admitted tasks to advance while all four are held."""
    first, advanced = {}, set()
    for sample in samples:
        slots = sample.get("slots", [])
        if not isinstance(slots, list) or len(slots) != 4:
            continue
        if not all(slot.get("is_processing") for slot in slots):
            continue
        tasks = tuple(sorted(slot["id_task"] for slot in slots))
        if len(set(tasks)) != 4:
            continue
        for slot in slots:
            key = tasks, slot["id_task"]
            progress = (slot["n_prompt_tokens_processed"], slot["next_token"][0]["n_decoded"])
            if key not in first:
                first[key] = progress
            elif progress > first[key]:
                advanced.add(key)
        if all((tasks, task) in advanced for task in tasks):
            return {"task_ids": tasks, "sample_time": sample["time"]}
    raise RuntimeError("no evidence of four execution sessions advancing concurrently; queued requests do not qualify")


def parallel(function, values):
    workers = ThreadPoolExecutor(max_workers=4)
    try:
        return list(workers.map(function, values))
    finally:
        # On interruption, return to the owned server's cleanup immediately;
        # its shutdown closes in-flight requests instead of waiting hours here.
        workers.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18092)
    parser.add_argument("--prompt-tokens", type=int, default=229000)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 3 or command[1:3] != ["serve", "llm"]:
        parser.error("pass a production gufo serve llm command after --")
    for flag in ("--host", "--port", "--cache-disk", "--served-model-name"):
        if option(command, flag) is not None:
            parser.error(f"this runner owns {flag}")
    expected = {"--sessions": "4", "--context": "262144", "--speculative": "mtp",
                "--draft-tokens": "7", "--cache-ram-bytes": "4294967296",
                "--cache-disk-bytes": "53687091200", "--vision-memory-bytes": "4294967296"}
    for flag, value in expected.items():
        if option(command, flag) != value:
            parser.error(f"qualification requires {flag} {value}")
    if not 0 < args.prompt_tokens <= 262144 - args.max_tokens or args.max_tokens <= 0:
        parser.error("prompt plus output must fit the full context")
    for flag in ("--model", "--mtp-model", "--mmproj"):
        if not Path(option(command, flag, "")).is_file():
            parser.error(f"{flag} must name an existing local file")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))  # refuse an occupied port
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command += ["--host", "127.0.0.1", "--port", str(args.port),
                "--cache-disk", str(output / "disposable-cache"),
                "--served-model-name", "vision-admission-validation"]
    report = {**provenance(), "command": command, "status": "running",
              "hardware_capacity": "unverified", "numerical_image_correctness": "unverified",
              "requests": {}, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output / "report.json", report)

    def chat(label, body, expected_status=200):
        response = request(args.port, "/v1/chat/completions", body, args.timeout)
        write_json(output / f"{label}.json", response)
        require(response["status"] == expected_status, f"{label}: HTTP {response['status']}, expected {expected_status}")
        report["requests"][label] = {"elapsed_ms": response["elapsed_ms"],
                                      "usage": response["response"].get("usage"),
                                      "timings": response["response"].get("timings")}
        return response["response"]

    def text_body(tag, lines, max_tokens):
        return {"model": "vision-admission-validation", "temperature": 0,
                "max_tokens": max_tokens, "cache_prompt": True,
                "messages": [{"role": "system", "content": f"Independent conversation {tag}.\n"},
                             {"role": "user", "content":
                              (f"Archive {tag}: marker nominal; voltage steady; no faults.\n" * lines)
                              + "List numbered operational checks, with detailed explanations, until the output budget is exhausted."}]}

    def count(body):
        response = request(args.port, "/v1/chat/completions/input_tokens", body, args.timeout)
        require(response["status"] == 200, "prompt token count failed")
        return response["response"]["input_tokens"]

    def verify_load(log):
        props = request(args.port, "/props")["response"]
        require(props["total_slots"] == 4 and props["default_generation_settings"]["n_ctx"] == 262144,
                "server changed session count or context")
        require(props["modalities"]["vision"], "vision projector is not available")
        text = log.read_text()
        require("event=vision_cache_grant policy=request-aware granted_bytes=4294967296" in text,
                "effective RAM checkpoint cache is not 4 GiB")
        backing = re.findall(r"event=backing_prepared committed_bytes=(\d+) required_payload_bytes=(\d+)", text)
        require(len(backing) == 1 and int(backing[0][1]) > 0
                and int(backing[0][0]) >= int(backing[0][1]),
                "the full granted checkpoint payload was not committed before readiness")
        report.setdefault("cache_backing", {})[log.name] = {
            "committed_bytes": int(backing[0][0]), "required_payload_bytes": int(backing[0][1])}
        require(re.search(r"event=load_completed[^\n]*speculative=mtp", text), "loaded mode is not MTP")
        return props

    image_body = None
    try:
        log = output / "server.log"
        with server(command, log, 1800) as process, Sampler(args.port, process.pid, output / "samples.jsonl") as samples:
            report["props"] = verify_load(log)
            image_body = text_body("image-history", 512, 128)
            image_body["messages"][-1]["content"] = [
                {"type": "text", "text": image_body["messages"][-1]["content"] + " Name the image's dominant color."},
                {"type": "image_url", "image_url": {"url": png_url(32, (255, 0, 0)), "detail": "high"}}]
            seed = chat("image-seed", image_body)
            checkpoint_ids = seed["usage"]["gufo"]["component_cache"]["captured_checkpoint_ids"]
            require(checkpoint_ids, "image request captured no checkpoint")
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                published = re.findall(r"event=published tokens=(\d+) checkpoint_id=(\d+)", log.read_text())
                if any(int(identity) in checkpoint_ids and int(tokens) >= seed["usage"]["prompt_tokens"] - 16
                       for tokens, identity in published):
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("image checkpoint was not published to the disposable disk cache")
            # Occupy every execution slot with new histories, forcing the image
            # continuation to use a retained checkpoint instead of its live slot.
            before_churn = time.time()
            parallel(lambda i: chat(f"churn-{i}", text_body(f"churn-{i}", 512, 2048)), range(4))
            report["churn_concurrency"] = concurrent_progress(samples.records(before_churn))
            warm = chat("image-ram", image_body)
            require(warm["usage"]["gufo"]["component_cache"]["reuse"] == "ram", "image checkpoint was not reused from RAM")
            require(warm["usage"]["cached_tokens"] >= seed["usage"]["prompt_tokens"] - 16,
                    "RAM checkpoint did not include the image-bearing prefix")
            # Count the actual prepared prompt; leave the complete output allowance.
            low, high = 1, args.prompt_tokens
            while low + 1 < high:
                middle = (low + high) // 2
                if count(text_body("0", middle, args.max_tokens)) <= args.prompt_tokens:
                    low = middle
                else:
                    high = middle
            bodies = [text_body(str(i), low, args.max_tokens) for i in range(4)]
            counts = [count(body) for body in bodies]
            require(all(args.prompt_tokens - 128 <= n <= 262144 - args.max_tokens for n in counts),
                    f"actual prompt sizes do not match requested depth: {counts}")
            report["concurrent_prompt_tokens"] = counts
            before_text = time.time()
            results = parallel(lambda item: chat(f"text-{item[0]}", item[1]), enumerate(bodies))
            report["concurrency"] = concurrent_progress(samples.records(before_text))
            require(sum(result["usage"].get("draft_tokens", 0) for result in results) > 0,
                    "MTP was configured but no actual drafts executed")

            # Five maximum-sized images fit context/body limits but exceed the
            # 4 GiB pool (history embeddings + RGB + encoder work), even cold.
            oversized = {**image_body, "stream": True, "max_tokens": 16,
                         "messages": [{"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": png_url(4096, (i * 40, 0, 0)), "detail": "high"}}
                             for i in range(5)]}]}
            refusal = chat("oversized-image", oversized, 503)
            require(refusal["error"]["code"] == "resource_exhausted", "oversized image did not produce an explicit resource refusal")
            chat("text-after-refusal", text_body("recovery", 1, 32))
        # The context manager stops only the child it started and drains disk
        # publication. Restart with the same owned cache, model and parameters.
        restarted_log = output / "server-restarted.log"
        with server(command, restarted_log, 1800) as process, Sampler(args.port, process.pid, output / "restart-samples.jsonl"):
            report["restart_props"] = verify_load(restarted_log)
            restored = chat("image-disk", image_body)
            require(restored["usage"]["gufo"]["cache_disk_hit"], "image continuation was not restored from disk")
            require(restored["usage"]["cached_tokens"] >= seed["usage"]["prompt_tokens"] - 16,
                    "disk checkpoint did not include the image-bearing prefix")
            require(restored["choices"][0]["message"] == warm["choices"][0]["message"], "RAM and disk image continuation differ")
            chat("text-after-restart", text_body("restart-recovery", 1, 32))
        report["status"] = "passed"
        report["hardware_capacity"] = "observed for this workload; inspect recorded peak memory and timings"
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = str(error)
        raise
    finally:
        write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
