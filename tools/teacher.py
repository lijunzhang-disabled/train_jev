#!/usr/bin/env python3
"""Talk to the teacher model (GLM3-Flash) through its OpenAI-compatible API. Stdlib only.

    python3 -I tools/teacher.py --base-url http://HOST:PORT/v1 --model MODEL check
    python3 -I tools/teacher.py --base-url http://HOST:PORT/v1 --model MODEL bench --concurrency 4 --requests 16
    python3 -I tools/teacher.py --mock check        # no server: tests this script itself

The API key comes from --api-key or $TEACHER_API_KEY (local servers often need none).
The endpoint is shared with colleagues, so keep --concurrency low: a few requests at a time, never the full capacity.
"""
import argparse
import concurrent.futures
import http.client
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request

BENCH_PROMPT = """Write three SystemVerilog assertions (SVA) for this module. Reply with only the assertions.

module counter (input logic clk, input logic rst, output logic [3:0] cnt);
  always_ff @(posedge clk) begin
    if (rst)              cnt <= 4'd0;
    else if (cnt == 4'd9) cnt <= 4'd0;
    else                  cnt <= cnt + 4'd1;
  end
endmodule"""


class Client:
    def __init__(self, base_url, model, api_key=None, timeout=600, mock=False):
        self.base_url, self.model, self.api_key = (base_url or "").rstrip("/"), model, api_key
        self.timeout, self.mock = timeout, mock

    def _request(self, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers,
                                     method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} from {self.base_url + path}: {e.read()[:500]!r}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as e:
            reason = getattr(e, "reason", e)
            raise RuntimeError(f"cannot reach {self.base_url + path}: {reason!r}") from None

    def models(self):
        if self.mock:
            return ["mock-model"]
        return [m["id"] for m in self._request("/models").get("data", [])]

    def chat(self, messages, max_tokens=512, temperature=0.7, n=1):
        """One chat completion request; returns the raw response dict (choices, usage)."""
        if self.mock:
            time.sleep(random.uniform(0.05, 0.2))
            k = random.randint(50, max_tokens)
            return {"choices": [{"message": {"role": "assistant", "content": "OK"}} for _ in range(n)],
                    "usage": {"prompt_tokens": 120, "completion_tokens": k * n}}
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "n": n}
        return self._request("/chat/completions", payload)


def check(client):
    print(f"endpoint: {client.base_url or '(mock)'}")
    try:
        names = client.models()
        print(f"[PASS] models: {names}")
        if client.model and names and client.model not in names:
            print(f"[WARN] --model {client.model!r} is not in that list")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] GET /models failed ({e}); trying a chat request anyway")
    t0 = time.time()
    try:
        resp = client.chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=1024, temperature=0)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] chat: {e}")
        print("       check --base-url (it usually ends in /v1), --model, the API key, and the firewall")
        return False
    msg = resp["choices"][0]["message"]
    print(f"[PASS] chat: {msg.get('content')!r} in {time.time() - t0:.1f}s, usage {resp.get('usage')}")
    if msg.get("reasoning_content"):
        print(f"       (reasoning model: {len(msg['reasoning_content'])} chars of reasoning returned)")
    return True


def bench(client, concurrency, requests, max_tokens):
    messages = [{"role": "user", "content": BENCH_PROMPT}]
    latencies, completion, prompt = [], 0, 0

    def one(_):
        t0 = time.time()
        resp = client.chat(messages, max_tokens=max_tokens)
        return time.time() - t0, resp.get("usage", {})

    t_start = time.time()
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(one, i) for i in range(requests)]):
            try:
                seconds, usage = fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"[WARN] request failed: {e}")
                continue
            latencies.append(seconds)
            completion += usage.get("completion_tokens", 0)
            prompt += usage.get("prompt_tokens", 0)
    wall = time.time() - t_start
    if not latencies:
        print("[FAIL] every request failed")
        return False
    latencies.sort()
    print(f"[PASS] {len(latencies)} requests ({errors} failed) at concurrency {concurrency} in {wall:.1f}s")
    print(f"       output tokens/s (all requests together): {completion / wall:,.0f}")
    print(f"       prompt tokens/s: {prompt / wall:,.0f}")
    print(f"       latency p50 {statistics.median(latencies):.1f}s  p90 {latencies[int(0.9 * len(latencies))]:.1f}s")
    print(f"       mean output per request: {completion / len(latencies):,.0f} tokens")
    return errors == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL"), help="e.g. http://10.0.0.5:8000/v1")
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL"))
    ap.add_argument("--api-key", default=os.environ.get("TEACHER_API_KEY"))
    ap.add_argument("--mock", action="store_true", help="fake responses, for testing this script")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="list models and send one request")
    b = sub.add_parser("bench", help="measure throughput with concurrent requests")
    b.add_argument("--concurrency", type=int, default=4, help="requests in flight; the server is shared, so keep it low")
    b.add_argument("--requests", type=int, default=16)
    b.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()
    if not args.mock and not args.base_url:
        ap.error("--base-url (or $TEACHER_BASE_URL) is required unless --mock")
    client = Client(args.base_url, args.model, args.api_key, mock=args.mock)
    if args.cmd == "check":
        ok = check(client)
    else:
        ok = bench(client, args.concurrency, args.requests, args.max_tokens)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
