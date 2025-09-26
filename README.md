

-----

# AI-Powered Voice and Text Email Agent

This project is a sophisticated AI agent that can manage your Gmail inbox using natural language commands, either through voice or text. It leverages a "plan-and-execute" architecture built with LangGraph and Google's Gemini Pro model to understand user requests, create a step-by-step plan, and execute a series of tools to accomplish the task.

##Features

  - **Dual Input Modes**: Interact with the agent via **voice commands**  or **text input** .
  - **Intelligent Planning**: Dynamically creates and executes multi-step plans to handle complex requests.
  - **Gmail Integration**: Natively integrates with the Gmail API to perform a variety of email tasks.
  - **Voice Transcription**: Utilizes a local Whisper model for fast and accurate speech-to-text transcription.
  - **Text-to-Speech Response**: Provides a natural, spoken response to the user after completing a task.
  - **Robust Error Handling**: Gracefully manages errors during planning or execution and informs the user.

##  Architecture Overview

The agent is built on a **plan-and-execute** model using a state machine managed by **LangGraph**. This approach allows for more reliable and complex task completion than a simple ReAct agent.

The workflow is orchestrated through a series of nodes, each responsible for a specific part of the process:

1.  **Transcribe Node**: Captures user input, either from the microphone (using Whisper) or direct text input.
2.  **Correct Transcription Node**: A Gemini model cleans up the raw transcription into a clear, actionable command.
3.  **Planner Node**: The core of the agent. This node takes the corrected command and, based on the available tools, generates a JSON list of steps to achieve the user's goal. It's specifically designed to handle dependencies, like using an email ID from a search step in a later summarization step.
4.  **Execute Tool Node**: Executes the steps in the plan one by one. It handles the "handoff" of data between steps, such as passing a message ID or extracted content to the next tool in the sequence.
5.  **Synthesize Response Node**: Once all tools have been executed (or if an error occurs), this node generates a final, user-friendly summary of the results.
6.  **Final Response Node**: The final answer is converted to speech and played back to the user.

This cyclical process (`execute_tool` -\> `execute_tool` -\> ...) continues until the plan is complete, at which point a final response is generated.

## Available Tools

The agent has access to the following tools to manage your Gmail account:

  - **`search_emails(query: str, max_results: int = 5)`**: Searches the inbox with a Gmail query string and returns a list of matching emails with their IDs, senders, and subjects.
  - **`summarize_email(llm, message_id: str)`**: Provides a concise summary of a specific email's content using its ID.
  - **`draft_email(recipient: str, subject: str, body: str)`**: Creates a draft email (does not send it).
  - **`delete_email(message_id: str)`**: Moves a specific email to the trash.

##  Setup and Installation

Follow these steps to get the agent running on your local machine.

### 1\. Prerequisites

  - Python 3.8+
  - Access to a microphone for voice commands.

### 2\. Clone the Repository

```bash
git clone https://github.com/your-username/ai_agent_for_email.git
cd ai_agent_for_email
```

### 3\. Install Dependencies

Install all the required Python packages using the `Requirements.txt` file.

```bash
pip install -r Requirements.txt
```

### 4\. Google API Configuration

#### a) Set up Gemini API Key

1.  Go to the [Google AI Studio](https://aistudio.google.com/app/apikey) to get your Gemini API key.
2.  Create a `.env` file in the root of the project directory.
3.  Add your API key to the `.env` file:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

#### b) Set up Gmail API Credentials

1.  Follow Google's [Python Quickstart guide](https://developers.google.com/gmail/api/quickstart/python) to enable the Gmail API and download your `credentials.json` file.
2.  Place the downloaded `credentials.json` file in the root of your project directory.
3.  **First Run**: The first time you run the agent, you will be prompted to authorize access to your Gmail account through a browser window. This will generate a `token.json` file, which will be used for authentication on subsequent runs.

## How to Run the Agent

Once the setup is complete, run the main script from your terminal:

```bash
python main.py
```

You will be prompted to choose your input method:

1.  **Voice Command **: The agent will listen for your command through your microphone.
2.  **Text Command ⌨️**: You can type your command directly into the terminal.

### Example Commands

  - "Find the latest email from newsletter@example.com"
  - "Search for emails about my recent order and then summarize the latest one"
  - "Draft an email to john.doe@email.com with the subject 'Meeting Follow-up' and the body 'Hi John, just following up on our meeting.'"
  - "Find the email with the subject 'Your Invoice' and then delete it"

To stop the agent, press `Ctrl+C` or say/type "goodbye".
