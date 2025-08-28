import json
import os
from datetime import datetime
from collections import defaultdict, deque
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# -------------------------
# Load TinyLlama model
# -------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model... this may take a minute ⏳")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"   # accelerate handles placement
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# -------------------------
# NPC logic
# -------------------------
player_states = defaultdict(lambda: deque(maxlen=3))
npc_moods = defaultdict(lambda: "neutral")

def update_mood(player_id, message):
    """Simple rules for mood change"""
    msg_lower = message.lower()
    if any(word in msg_lower for word in ["help", "please", "thanks"]):
        npc_moods[player_id] = "friendly"
    elif any(word in msg_lower for word in ["stupid", "hate", "idiot"]):
        npc_moods[player_id] = "angry"
    # else keep same mood

def generate_reply(player_id, message):
    """Generate NPC reply based on last 3 msgs + mood"""
    update_mood(player_id, message)

    history = "\n".join(player_states[player_id])
    mood = npc_moods[player_id]

    prompt = f"""
You are an NPC in a game. 
Current mood: {mood}.
Recent conversation:
{history}
Player says: {message}
NPC reply (stay in mood, short, natural):
"""

    outputs = generator(
        prompt,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    reply = outputs[0]["generated_text"].split("NPC reply")[-1].strip()
    player_states[player_id].append(f"Player: {message}")
    player_states[player_id].append(f"NPC: {reply}")
    return reply

# -------------------------
# Process input messages
# -------------------------
def main():
    with open("players.json", "r") as f:
        data = json.load(f)

    # sort by timestamp
    data.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]))

    with open("chatlog.txt", "w", encoding="utf-8") as log:
        for msg in data:
            player_id = msg["player_id"]
            message = msg.get("text", "").strip()   # <-- use "text"
            ts = msg["timestamp"]

            reply = generate_reply(player_id, message)
            mood = npc_moods[player_id]
            state = list(player_states[player_id])[-6:]  # last 3 exchanges

            log_entry = f"""
[{ts}]
Player {player_id}: {message}
NPC reply: {reply}
Mood: {mood}
State: {state}
"""
            print(log_entry)
            log.write(log_entry + "\n")

if __name__ == "__main__":
    main()
