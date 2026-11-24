"""Strata #4663 ported: Anthropic's tool search on /v1/messages - `defer_loading` tools and `tool_reference` blocks.

Claude Code (ENABLE_TOOL_SEARCH) declares ToolSearch and a `defer_loading` DeferredToolPlaceholder, and names its MCP
tools only in the conversation. ToolSearch's result is a tool_result of tool_reference blocks; from then on the found
tool is in `tools` with `defer_loading: true`. A deferred tool stays out of the prompt's tool list and its definition
is put where the reference stands, so loading a tool leaves the prompt's start (and the prompt cache) unchanged.

    python -m unittest serve.test_tool_search -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import OutputParser, anthropic_to_messages, anthropic_tools  # noqa: E402
from serve.test_dsv41 import ApiBase, call  # noqa: E402

MCP = "mcp__plugin_ctx__ctx_search"
SEARCH = {"name": "ToolSearch", "description": "Load deferred tools.",
          "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}
PLACEHOLDER = {"name": "DeferredToolPlaceholder", "input_schema": {"type": "object"}, "defer_loading": True}
LOADED = {"name": MCP, "description": "Search the index.", "defer_loading": True,
          "input_schema": {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}},
                                                            "source": {"type": "string"}}}}
CALL = f"<tool_call>\n<function={MCP}>\n<parameter=queries>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>"
FIRST = [{"role": "user", "content": "find it"}]
REFERENCE = FIRST + [
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch",
                                       "input": {"query": "select:" + MCP}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                  "content": [{"type": "tool_reference", "tool_name": MCP}]}]}]


def events(text, tools, deferred, thinking=False):
    parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True, declared_only=True, deferred=deferred)
    return parser.feed(text) + parser.finish("stop")


def result_text(messages) -> str:
    content = messages[-1]["content"]
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


class Conversion(unittest.TestCase):
    def test_deferred_tools_stay_out_of_the_tool_list(self):
        _, tools, _ = anthropic_to_messages({"tools": [SEARCH, PLACEHOLDER, LOADED], "messages": REFERENCE})
        self.assertEqual([t["name"] for t in tools], ["ToolSearch"])

    def test_a_reference_loads_the_definition_where_it_is(self):
        messages, _, _ = anthropic_to_messages({"tools": [SEARCH, LOADED], "messages": REFERENCE})
        self.assertEqual(messages[-1]["role"], "tool")
        line = next(x for x in result_text(messages).splitlines() if x.startswith("{"))
        self.assertEqual(json.loads(line),
                         {"name": MCP, "description": "Search the index.", "parameters": LOADED["input_schema"]})

    def test_a_reference_to_a_listed_tool_adds_nothing(self):
        req = {"tools": [SEARCH, {**LOADED, "defer_loading": False}], "messages": REFERENCE}
        self.assertEqual(result_text(anthropic_to_messages(req)[0]), "")

    def test_a_reference_to_an_undeclared_tool_is_left_out(self):
        self.assertEqual(result_text(anthropic_to_messages({"tools": [SEARCH], "messages": REFERENCE})[0]), "")

    def test_the_parser_gets_every_declared_tool(self):
        tools, deferred = anthropic_tools({"tools": [SEARCH, PLACEHOLDER, LOADED]})
        self.assertEqual([t["name"] for t in tools], ["ToolSearch", "DeferredToolPlaceholder", MCP])
        self.assertTrue(deferred)
        self.assertFalse(anthropic_tools({"tools": [SEARCH]})[1])
        self.assertEqual(anthropic_tools({}), (None, False))


class Calls(unittest.TestCase):
    def call(self, evs):
        return [e.call for e in evs if e.kind == "tool_call"]

    def test_an_undeclared_call_is_a_call_with_tool_search(self):
        calls = self.call(events(CALL, [{"name": "ToolSearch"}], deferred=True))
        self.assertEqual([(c.name, c.arguments) for c in calls], [(MCP, {"queries": ["a", "b"]})])

    def test_without_tool_search_it_stays_text(self):
        evs = events(CALL, [{"name": "ToolSearch"}], deferred=False)
        self.assertEqual(self.call(evs), [])
        self.assertIn("<tool_call>", "".join(e.text for e in evs if e.kind == "content"))

    def test_a_loaded_deferred_tool_keeps_its_schema(self):
        tools, deferred = anthropic_tools({"tools": [SEARCH, LOADED]})
        got = self.call(events(CALL.replace("queries>\n[\"a\", \"b\"]", "source>\n42"), tools, deferred))[0]
        self.assertEqual(got.arguments, {"source": "42"})

    def test_a_call_in_the_reasoning_is_rescued_too(self):
        calls = self.call(events("Searching.\n" + CALL, [{"name": "ToolSearch"}], deferred=True, thinking=True))
        self.assertEqual([c.name for c in calls], [MCP])


class Route(ApiBase):
    SCRIPT = "</think>" + call((MCP, [("queries", '["a", "b"]', False)]))

    def messages(self, tools, messages):
        return self.post("/v1/messages", {"model": "x", "max_tokens": 400, "tools": tools, "messages": messages})

    def test_claude_code_tool_search(self):
        self.start(self.SCRIPT)
        status, r = self.messages([SEARCH, PLACEHOLDER], FIRST)
        self.assertEqual(status, 200, r)
        self.assertEqual(r["stop_reason"], "tool_use")
        self.assertEqual([(u["name"], u["input"]) for u in r["content"] if u["type"] == "tool_use"],
                         [(MCP, {"queries": ["a", "b"]})])
        self.assertNotIn("DeferredToolPlaceholder", self.prompt())

    def test_the_loaded_tool_is_in_the_tool_result_not_the_tool_list(self):
        self.start(self.SCRIPT)
        status, r = self.messages([SEARCH, PLACEHOLDER, LOADED], REFERENCE)
        self.assertEqual(status, 200, r)
        prompt = self.prompt()
        self.assertEqual(prompt.count(f'"name": "{MCP}"'), 1)
        self.assertGreater(prompt.index(f'"name": "{MCP}"'), prompt.index("find it"))

    def test_loading_a_tool_keeps_the_prompt_start(self):
        self.start(self.SCRIPT)
        self.messages([SEARCH, PLACEHOLDER], FIRST)
        before = self.prompt()
        self.messages([SEARCH, PLACEHOLDER, LOADED], REFERENCE)
        after = self.prompt()
        head = before[:before.index("find it")]
        self.assertTrue(after.startswith(head))


if __name__ == "__main__":
    unittest.main()
