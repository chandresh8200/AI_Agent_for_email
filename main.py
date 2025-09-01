import os
from dotenv import load_dotenv
from typing import List
from langgraph.graph import StateGraph, END
import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from speech_processing import transcribe_audio, text_to_speech
# IMPORTANT: Make sure you are importing from the right tools file
# For this version, we need the Gmail API tools, not the RAG tools.
from email_tools import AVAILABLE_TOOLS
from agent_state import AgentState, ToolCall

# (The top part of your file remains the same)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in .env file")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)


# --- Nodes: transcribe and correct are unchanged ---
def transcribe_node(state: AgentState) -> AgentState:
    if state.initialUserInput:
        text_input = state.initialUserInput
    else:
        text_input = transcribe_audio()
    state.userInput = text_input
    return state


def correct_transcription_node(state: AgentState) -> AgentState:
    if not state.userInput.strip():
        state.correctedInput = ""
        return state
    prompt = f"Correct the following transcribed text into a clean command. RAW: '{state.userInput}' CORRECTED:"
    response = llm.invoke(prompt)
    state.correctedInput = response.content.strip()
    return state


# --- NEW, SMARTER PLANNER NODE ---
def planner_node(state: AgentState) -> AgentState:
    """Node 3: Creates a robust, step-by-step plan based on the user's command."""
    print("--- 🧠 Planning... ---")

    tool_signatures = "\n".join([f"- {name}: {func.__doc__.strip()}" for name, func in AVAILABLE_TOOLS.items()])

    prompt = f"""You are an expert planner for an email assistant. Your job is to create a step-by-step plan to accomplish the user's goal.
You have the following tools available:
{tool_signatures}

**CRITICAL RULES:**
1.  **ID Placeholder:** When a tool needs a `message_id` from a previous `search_emails` step, you MUST use the exact placeholder string: `"<ID from previous step>"`.
2.  **Content Placeholder:** When a tool needs to use the text content extracted by `get_pdf_attachment_text` or `get_email_details`, you MUST use the exact placeholder string: `"<Content from previous step>"`.
3.  **PDF/Invoice Logic:** If the user asks a question about an invoice, report, or any attached document, the plan MUST be:
    a. `search_emails` to find the email ID.
    b. `get_pdf_attachment_text` using the ID.
    c. (Optional) `summarize_email_content` using the extracted text.
4.  **Latest Email Logic:** If the user asks for the "last" or "latest" email, you MUST set `max_results=1` in your `search_emails` tool call.

User's Request: "{state.correctedInput}"

Return ONLY the JSON list of tool calls.
"""
    response = llm.invoke(prompt)
    try:
        plan_json_str = response.content.strip().replace("```json", "").replace("```", "")
        plan_list = json.loads(plan_json_str)
        state.plan = [ToolCall(**call) for call in plan_list]
        print(f"--- 📋 Plan Created: ---")
        for i, step in enumerate(state.plan):
            print(f"{i + 1}. {step.tool_name}({step.tool_kwargs})")
    except (json.JSONDecodeError, TypeError) as e:
        print(f"--- ❌ Error generating plan: {e} ---")
        state.error = "I had trouble creating a plan for that request. Could you rephrase it?"
    return state


# --- NEW, SMARTER EXECUTOR NODE ---
def execute_tool_node(state: AgentState) -> AgentState:
    """Node 4: Executes the next tool and handles handoffs of data between steps."""
    if not state.plan:
        return state

    current_step = state.plan.pop(0)
    tool_name = current_step.tool_name
    tool_kwargs = current_step.tool_kwargs

    print(f"--- 🛠️ Executing: {tool_name}({tool_kwargs}) ---")

    if tool_name not in AVAILABLE_TOOLS:
        result = f"Error: Tool '{tool_name}' not found."
    else:
        # Handle handoffs by replacing placeholders with actual data
        for key, value in tool_kwargs.items():
            if isinstance(value, str):
                # Handoff for Message ID
                if "<ID from previous step>" in value:
                    all_previous_results = " ".join([str(res.get('result', '')) for res in state.tool_results])
                    found_ids = re.findall(r"ID: ([a-zA-Z0-9]+)", all_previous_results)
                    if found_ids:
                        tool_kwargs[key] = found_ids[-1]
                    else:
                        state.error = "Could not find a message ID from a previous step."
                        state.plan = []
                        return state

                # Handoff for Content (from PDF or email body)
                if "<Content from previous step>" in value:
                    if state.tool_results:
                        # Get the result of the absolute last tool call
                        tool_kwargs[key] = state.tool_results[-1].get('result', '')
                    else:
                        state.error = "Could not find content from a previous step."
                        state.plan = []
                        return state

        # Inject the LLM instance if needed
        if "llm" in AVAILABLE_TOOLS[tool_name].__code__.co_varnames:
            tool_kwargs['llm'] = llm

        try:
            tool_function = AVAILABLE_TOOLS[tool_name]
            result = tool_function(**tool_kwargs)
        except Exception as e:
            result = f"Error executing tool: {e}"

    print(f"--- Result: {result} ---")
    state.tool_results.append({"tool": tool_name, "result": result})
    return state


# (The rest of main.py: synthesize_response_node, final_response_node, router, create_workflow, and the main loop are UNCHANGED)
def synthesize_response_node(state: AgentState) -> AgentState:
    print("--- 💬 Synthesizing Final Response... ---")
    if state.error:
        state.final_response = state.error
        return state
    prompt = f"""You are an email assistant. Based on the user's original request and the results of the tools you used, provide a clear, concise, and friendly final response.

Original Request: "{state.userInput}"
Tool Results:
{json.dumps(state.tool_results, indent=2)}

Final Response:
"""
    response = llm.invoke(prompt)
    state.final_response = response.content.strip()
    return state


def final_response_node(state: AgentState) -> AgentState:
    print(f"--- Final Answer: {state.final_response} ---")
    text_to_speech(state.final_response)
    return state


def should_continue_router(state: AgentState):
    if state.error or not state.plan:
        return "synthesize_response"
    return "execute_tool"


def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("transcribe", transcribe_node)
    workflow.add_node("correct_transcription", correct_transcription_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("synthesize_response", synthesize_response_node)
    workflow.add_node("final_response", final_response_node)
    workflow.set_entry_point("transcribe")
    workflow.add_edge("transcribe", "correct_transcription")
    workflow.add_edge("correct_transcription", "planner")
    workflow.add_conditional_edges("planner",
                                   lambda x: "execute_tool" if x.plan and not x.error else "synthesize_response",
                                   {"execute_tool": "execute_tool", "synthesize_response": "synthesize_response"})
    workflow.add_conditional_edges("execute_tool", should_continue_router,
                                   {"execute_tool": "execute_tool", "synthesize_response": "synthesize_response"})
    workflow.add_edge("synthesize_response", "final_response")
    workflow.add_edge("final_response", END)
    return workflow.compile()


if __name__ == "__main__":
    print("🤖 Plan-and-Execute Email Agent is running!")
    print("=" * 50)
    app = create_workflow()
    while True:
        try:
            print("\n--- 🚀 Starting new conversation cycle ---")
            choice = input("Choose input method:\n[1] Voice Command 🎤\n[2] Text Command ⌨️\n> ")
            initial_state = AgentState()
            if choice == '2':
                initial_state.initialUserInput = input("Please type your command: ")
            final_state = app.invoke(initial_state)
            if final_state.correctedInput.lower() in ["exit", "quit", "stop", "goodbye"]:
                text_to_speech("Goodbye!")
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"--- 🚨 Critical error in main loop: {e} ---")
            continue

