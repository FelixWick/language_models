import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3"  # Change to your installed model (e.g., 'mistral', 'gemma', etc.)

def stream_chat(messages):
    """Send chat history to Ollama and stream the model's reply."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True
    }

    with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "message" in data and "content" in data["message"]:
                chunk = data["message"]["content"]
                print(chunk, end="", flush=True)
                full_response += chunk
            if data.get("done"):
                break

        print()  # new line after stream ends
        return full_response.strip()


def main():
    print("🤖 Ollama Chatbot — type 'exit' to quit\n")
    messages = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        print("Assistant:", end=" ", flush=True)

        reply = stream_chat(messages)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()