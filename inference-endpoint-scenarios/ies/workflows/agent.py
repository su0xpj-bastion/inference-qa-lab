"""Agent: a multi-turn tool-calling loop, timed end to end.

Agent users wait on the whole loop (plan -> tool calls -> tool results -> ... -> answer),
not on one request, so per-request TTFT understates what fast inference buys or costs.
The task needs two tool rounds with parallel calls:

  get_weather(Paris), get_weather(Tokyo)  ->  convert_c_to_f(18), convert_c_to_f(22)
  ->  "Paris=64.4F, Tokyo=71.6F"

  agent_task_success   final answer has both correct Fahrenheit values, every rep
  agent_tool_args      every tool call's arguments parse as JSON and match the schema
  agent_loop_latency   loop wall-clock p50 within budget; per-turn TTFT and turn count reported
  agent_tool_use       the model used the conversion tool instead of doing the math itself
                       (reported: a correct answer without the tool is a different behaviour)
"""
from __future__ import annotations

import json
import re

import jsonschema

from ..core import Check, Ctx, guarded, pct

W = "agent"
WEATHER = {"Paris": (18, "cloudy"), "Tokyo": (22, "sunny"), "Berlin": (15, "rain"), "Austin": (31, "sunny")}
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current temperature in Celsius and conditions for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "convert_c_to_f", "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {"type": "object", "properties": {"celsius": {"type": "number"}},
                       "required": ["celsius"], "additionalProperties": False}}},
]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
SYSTEM = ("You are a travel assistant. Use the tools for every fact and every conversion; do not compute "
          "conversions yourself. Call independent tools in parallel.")
TASK = ("Get the current weather in Paris and in Tokyo, convert both temperatures to Fahrenheit with the "
        "conversion tool, then reply with exactly one line in this format: Paris=<F>F, Tokyo=<F>F")
EXPECT = {"Paris": "64.4", "Tokyo": "71.6"}


def execute(name: str, args: dict) -> dict:
    if name == "get_weather":
        c, cond = WEATHER.get(args["city"], (20, "unknown"))
        return {"city": args["city"], "temp_c": c, "condition": cond}
    if name == "convert_c_to_f":
        return {"celsius": args["celsius"], "fahrenheit": round(args["celsius"] * 9 / 5 + 32, 1)}
    return {"error": f"unknown tool {name}"}


def run_loop(ctx: Ctx) -> dict:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    turns, ttfts, bad_args, used = [], [], [], []
    wall = 0.0
    final = ""
    for turn in range(ctx.slo["agent_max_turns"]):
        c = ctx.chat(msgs, stream=True, tools=TOOLS, temperature=0, max_completion_tokens=256)
        wall += c.wall
        turns.append({"turn": turn, "wall": c.wall, "ttft": c.ttft, "calls": len(c.tool_calls),
                      "finish": c.finish_reason})
        if c.ttft is not None:
            ttfts.append(c.ttft)
        if not c.tool_calls:
            final = c.content.strip()
            break
        ids = [tc.get("id") or f"call_{turn}_{i}" for i, tc in enumerate(c.tool_calls)]
        msgs.append({"role": "assistant", "content": c.content or None, "tool_calls": [
            {"id": i, "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for i, tc in zip(ids, c.tool_calls)]})
        for call_id, tc in zip(ids, c.tool_calls):
            used.append(tc["name"])
            try:
                args = json.loads(tc["arguments"] or "{}")
                jsonschema.validate(args, SCHEMAS[tc["name"]])
                result = execute(tc["name"], args)
            except (json.JSONDecodeError, jsonschema.ValidationError, KeyError) as exc:
                bad_args.append(f"{tc['name']}({tc['arguments']!r}): {type(exc).__name__}")
                result = {"error": "invalid arguments; call again with valid JSON matching the schema"}
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
    got = dict(re.findall(r"(Paris|Tokyo)\s*=\s*(-?\d+(?:\.\d+)?)", final))
    return {"final": final, "correct": all(got.get(k) == v for k, v in EXPECT.items()), "turns": turns,
            "wall": wall, "ttfts": ttfts, "bad_args": bad_args, "used": used, "finished": bool(final)}


def _runs(ctx: Ctx) -> list[dict]:
    if "_agent" not in ctx.__dict__:
        ctx.__dict__["_agent"] = [run_loop(ctx) for _ in range(ctx.slo["agent_reps"])]
    return ctx.__dict__["_agent"]


def agent_task_success(ctx: Ctx) -> Check:
    runs = _runs(ctx)
    ok = sum(r["correct"] for r in runs)
    stalled = sum(not r["finished"] for r in runs)
    detail = f"{ok}/{len(runs)} loops answered correctly"
    if stalled:
        detail += f"; {stalled} hit the {ctx.slo['agent_max_turns']}-turn cap without answering"
    wrong = [r["final"] for r in runs if r["finished"] and not r["correct"]]
    if wrong:
        detail += f"; wrong answers {wrong[:2]}"
    return Check(W, "agent_task_success", "pass" if ok == len(runs) else "fail", detail)


def agent_tool_args(ctx: Ctx) -> Check:
    bad = [b for r in _runs(ctx) for b in r["bad_args"]]
    calls = sum(len(r["used"]) for r in _runs(ctx))
    return Check(W, "agent_tool_args", "fail" if bad else "pass",
                 f"{len(bad)}/{calls} tool calls had invalid arguments {bad[:2] or ''}".strip())


def agent_loop_latency(ctx: Ctx) -> Check:
    runs = [r for r in _runs(ctx) if r["finished"]]
    if not runs:
        return Check(W, "agent_loop_latency", "fail", "no loop finished")
    p50 = pct([r["wall"] for r in runs], 50)
    budget = ctx.slo["agent_loop_p50_max_s"]
    turns = [len(r["turns"]) for r in runs]
    ttft = pct([t for r in runs for t in r["ttfts"]], 50)
    return Check(W, "agent_loop_latency", "pass" if p50 <= budget else "fail",
                 f"loop p50 {p50 * 1e3:.0f} ms (budget {budget * 1e3:.0f}) over {turns} turns; "
                 f"per-turn TTFT p50 {ttft * 1e3:.0f} ms",
                 metrics={"loop_p50": p50, "turns": turns, "turn_ttft_p50": ttft})


def agent_tool_use(ctx: Ctx) -> Check:
    runs = _runs(ctx)
    no_convert = sum("convert_c_to_f" not in r["used"] for r in runs)
    parallel = sum(any(t["calls"] > 1 for t in r["turns"]) for r in runs)
    detail = f"conversion tool used in {len(runs) - no_convert}/{len(runs)} loops; parallel calls in {parallel}"
    return Check(W, "agent_tool_use", "warn" if no_convert else "pass", detail, gated=False)


CHECKS = (agent_task_success, agent_tool_args, agent_loop_latency, agent_tool_use)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
