import { beforeEach, describe, expect, it, vi } from "vitest"
import OpenAI from "openai"

import { OpenAiCodexHandler } from "../openai-codex"
import { NativeToolCallParser } from "../../../core/assistant-message/NativeToolCallParser"
import { openAiCodexOAuthManager } from "../../../integrations/openai-codex/oauth"
import type { ApiStreamChunk } from "../../transform/stream"
import { collectStream } from "../../../test-utils/stream"
import trace from "./fixtures/codex-native-tool-events.json"

vi.mock("@roo-code/telemetry", () => ({
	TelemetryService: { instance: { captureException: vi.fn() } },
}))

vi.mock("../../../i18n", () => ({
	t: (key: string, options?: { message?: string }) => `${key}: ${options?.message ?? ""}`,
}))

function item(index: number, name = "read_file", args = JSON.stringify({ path: `fixture-${index}.txt` })) {
	return { type: "function_call", id: `fc_${index}`, call_id: `call_${index}`, name, arguments: args }
}

function added(index: number) {
	return { type: "response.output_item.added", output_index: index, item: item(index, "read_file", "") }
}

function done(index: number, args?: string) {
	return { type: "response.output_item.done", output_index: index, item: item(index, "read_file", args) }
}

// Exercise the real parser contract used by Task, without invoking filesystem tools or UI.
function finalize(chunks: ApiStreamChunk[]) {
	const scope = NativeToolCallParser.createScope()
	const consume = (events: ReturnType<typeof NativeToolCallParser.processRawChunk>) => {
		const results = []
		for (const event of events) {
			if (event.type === "tool_call_start")
				NativeToolCallParser.startStreamingToolCall(event.id, event.name, scope)
			if (event.type === "tool_call_delta")
				NativeToolCallParser.processStreamingChunk(event.id, event.delta, scope)
			if (event.type === "tool_call_end") {
				const block = NativeToolCallParser.finalizeStreamingToolCall(event.id, scope)
				results.push(block ? { ...block, id: event.id } : null)
			}
		}
		return results
	}
	const results = []
	for (const chunk of chunks) {
		if (chunk.type === "tool_call_partial")
			results.push(...consume(NativeToolCallParser.processRawChunk(chunk, scope)))
	}
	results.push(...consume(NativeToolCallParser.finalizeRawChunks(scope)))
	return results
}

describe("Codex native argument assembly", () => {
	let handler: OpenAiCodexHandler

	beforeEach(() => {
		vi.restoreAllMocks()
		handler = new OpenAiCodexHandler({ apiModelId: "gpt-6-sol" })
		vi.spyOn(openAiCodexOAuthManager, "getAccessToken").mockResolvedValue("fixture-token")
		vi.spyOn(openAiCodexOAuthManager, "getAccountId").mockResolvedValue("fixture-account")
	})

	async function replay(events: unknown[]) {
		const chunks: ApiStreamChunk[] = []
		for (const event of events)
			chunks.push(...(await collectStream(handler["processEvent"](event, handler.getModel()))))
		return chunks
	}

	it("preserves all three calls from the anonymized real trace", async () => {
		const chunks = await replay(trace)
		expect(chunks).toHaveLength(3)
		expect(new Set(chunks.filter((c) => c.type === "tool_call_partial").map((c) => c.index)).size).toBe(3)
		expect(finalize(chunks)).toMatchObject([
			{ id: "call_1", name: "read_file", nativeArgs: { path: "fixture.txt" } },
			{ id: "call_2", name: "list_files", nativeArgs: { path: "." } },
			{ id: "call_3", name: "codebase_search", nativeArgs: { query: "fixture marker", path: "." } },
		])
	})

	it("keeps 21 sequential calls separate even without legacy index", async () => {
		const events = Array.from({ length: 21 }, (_, index) => [
			added(index),
			{
				type: "response.function_call_arguments.delta",
				output_index: index,
				item_id: `fc_${index}`,
				delta: item(index).arguments,
			},
			done(index),
		]).flat()
		const chunks = await replay(events)
		expect(chunks).toHaveLength(21)
		expect(finalize(chunks)).toEqual(
			Array.from({ length: 21 }, (_, index) =>
				expect.objectContaining({
					id: `call_${index}`,
					nativeArgs: { path: `fixture-${index}.txt` },
				}),
			),
		)
	})

	it("resolves interleaved events by item_id, call_id and output_index", async () => {
		const chunks = await replay([
			added(2),
			added(7),
			{ type: "response.function_call_arguments.delta", item_id: "fc_2", delta: '{"path":' },
			{ type: "response.function_call_arguments.delta", call_id: "call_7", delta: '{"path":"seven.txt"}' },
			{ type: "response.function_call_arguments.delta", output_index: 2, delta: '"two.txt"}' },
			{ type: "response.function_call_arguments.done", item_id: "fc_7", arguments: '{"path":"seven.txt"}' },
			{ type: "response.output_item.done", item: { ...item(7), arguments: undefined } },
			{ type: "response.output_item.done", item: { ...item(2), arguments: undefined } },
		])
		expect(finalize(chunks)).toMatchObject([
			{ id: "call_7", nativeArgs: { path: "seven.txt" } },
			{ id: "call_2", nativeArgs: { path: "two.txt" } },
		])
	})

	it.each(["", '{"path":', '{"path":"wrong.txt"}'])("uses terminal arguments instead of delta %j", async (delta) => {
		const chunks = await replay([
			added(0),
			{ type: "response.function_call_arguments.delta", output_index: 0, delta },
			done(0),
		])
		expect(finalize(chunks)).toMatchObject([{ nativeArgs: { path: "fixture-0.txt" } }])
	})

	it.each(["", '{"path":'])("does not repair invalid terminal JSON %j with valid deltas", async (args) => {
		const chunks = await replay([
			added(0),
			{ type: "response.function_call_arguments.delta", output_index: 0, delta: item(0).arguments },
			done(0, args),
		])
		expect(finalize(chunks)).toEqual([null])
	})

	it("emits each call once across repeated terminal events and response.completed", async () => {
		const chunks = await replay([
			added(0),
			done(0),
			done(0),
			{ type: "response.function_call_arguments.delta", item_id: "fc_0", delta: "late duplicate" },
			{ type: "response.completed", response: { output: [item(0)] } },
			{ type: "response.done", response: { output: [item(0)] } },
		])
		expect(chunks).toHaveLength(1)
		expect(finalize(chunks)).toMatchObject([{ id: "call_0", nativeArgs: { path: "fixture-0.txt" } }])
	})

	it("recovers completed-only calls independently of assistant text", async () => {
		const chunks = await replay([
			{ type: "response.output_text.delta", delta: "hello" },
			{ type: "response.completed", response: { output: [item(0), item(1)] } },
		])
		expect(finalize(chunks)).toMatchObject([{ id: "call_0" }, { id: "call_1" }])
	})

	it("accepts tool_call aliases and object arguments", async () => {
		const chunks = await replay([
			{
				type: "response.output_item.added",
				index: 5,
				item: { type: "tool_call", id: "tc", tool_call_id: "call_tc", function: { name: "read_file" } },
			},
			{ type: "response.tool_call_arguments.done", item_id: "tc", arguments: { path: "fixture.txt" } },
			{ type: "response.completed", response: { output: [] } },
		])
		expect(finalize(chunks)).toMatchObject([{ id: "call_tc", nativeArgs: { path: "fixture.txt" } }])
	})

	it.each(["sdk", "sse"])("resets assembly between requests through %s transport", async (transport) => {
		if (transport === "sse") {
			handler["client"] = new OpenAI({ apiKey: "fixture-token" })
			vi.spyOn(handler["client"].responses, "create").mockRejectedValue(new Error("SDK unavailable"))
		}
		const fetchMock = vi.spyOn(globalThis, "fetch")
		for (const path of ["first.txt", "second.txt"]) {
			const events = [
				added(0),
				done(0, JSON.stringify({ path })),
				{ type: "response.completed", response: { output: [] } },
			]
			fetchMock.mockResolvedValueOnce(
				new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
					headers: { "Content-Type": "text/event-stream" },
				}),
			)
			const chunks = await collectStream(handler.createMessage("fixture", [{ role: "user", content: "fixture" }]))
			expect(finalize(chunks)).toMatchObject([{ id: "call_0", nativeArgs: { path } }])
			expect(chunks.filter((c) => c.type === "tool_call_partial")).toHaveLength(1)
		}
		expect(fetchMock).toHaveBeenCalledTimes(2)
	})

	it.each(["error", "response.error", "response.failed"])("preserves %s provider failures", async (type) => {
		await replay([
			added(0),
			{ type: "response.function_call_arguments.delta", output_index: 0, delta: item(0).arguments },
		])
		await expect(
			replay([
				{
					type,
					response: { error: { message: "fixture provider failure" } },
					error: { message: "fixture provider failure" },
				},
			]),
		).rejects.toThrow("fixture provider failure")
	})
	it.each(["sdk", "sse"])(
		"propagates provider errors through %s without replaying or emitting buffered calls",
		async (transport) => {
			if (transport === "sse") {
				handler["client"] = new OpenAI({ apiKey: "fixture-token" })
				vi.spyOn(handler["client"].responses, "create").mockRejectedValue(new Error("SDK unavailable"))
			}
			const events = [
				added(0),
				{ type: "response.function_call_arguments.delta", output_index: 0, delta: item(0).arguments },
				{ type: "response.failed", response: { error: { message: "fixture provider failure" } } },
			]
			const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
				new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""), {
					headers: { "Content-Type": "text/event-stream" },
				}),
			)
			const chunks: ApiStreamChunk[] = []
			const consume = async () => {
				for await (const chunk of handler.createMessage("fixture", [{ role: "user", content: "fixture" }]))
					chunks.push(chunk)
			}
			await expect(consume()).rejects.toThrow("fixture provider failure")
			expect(chunks).toEqual([])
			expect(fetchMock).toHaveBeenCalledTimes(1)
		},
	)
})
