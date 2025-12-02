import os
import time
from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY_HERE")

# Get file path

def get_memory_database_path():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    memory_db = os.path.join(desktop, "MemoryDatabase")
    return memory_db


# Send info to GPT

def load_known_user_info(person_name: str):
    """
    Searches Desktop/MemoryDatabase for a file named '<person_name>.txt'
    and returns its content.
    """

    memory_db = get_memory_database_path()
    file_path = os.path.join(memory_db, f"{person_name}.txt")

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading known user file: {e}")
            return None
    else:
        print(f"[Warning] No memory file found for {person_name}: {file_path}")
        return None


# Create unknown

def create_unknown_user_file():
    """
    Creates a new file for an unknown user using a timestamp.
    Returns path to the file.
    """

    memory_db = get_memory_database_path()
    timestamp = int(time.time())
    filename = f"UnknownUser_{timestamp}.txt"
    file_path = os.path.join(memory_db, filename)

    try:
        with open(file_path, "w") as f:
            f.write("Name: \n")
            f.write("Hobbies: \n")
            f.write("Likes: \n")
            f.write("Age: \n")
            f.write("Occupation: \n")
        print(f"Created new user memory file: {file_path}")
    except Exception as e:
        print(f"Error creating unknown user file: {e}")
        return None

    return file_path


# Chat

def user_interaction_chat(faceKnown: bool, person_name: str = None):
    """
    If faceKnown = True, person_name MUST be provided.
    """

    model_name = "gpt-4o-mini"
    memory_file_path = None

    # known
    if faceKnown:
        if not person_name:
            raise ValueError("faceKnown=True requires a person_name.")

        print(f"Face Recognized: Loading profile for {person_name}...")

        user_info = load_known_user_info(person_name)
        if not user_info:
            user_info = "No stored profile information available."

        system_prompt = (
            "You are a helpful assistant. The user is a known contact, and here is their "
            f"profile information:\n---\n{user_info}\n---\n"
            "Greet the user appropriately, referencing a subject in their profile. "
            "Keep responses concise and friendly."
        )

    # unknown
    else:
        print("Face Not Recognized.")

        memory_file_path = create_unknown_user_file()

        system_prompt = (
            "You are a friendly and polite assistant. Your goal is to learn the user's "
            "name, hobbies/interests/likes, age, and occupation—but through natural conversation.\n"
            "Start by introducing yourself and casually asking for their name.\n"
            "Do NOT ask for all details at once. Keep the flow natural."
        )

    # prep convo
    conversation_history = [{"role": "system", "content": system_prompt}]

    print("-" * 40)
    print(
        f"AI Chat Session Started "
        f"({'KNOWN' if faceKnown else 'UNKNOWN'}). Type 'quit' to end.\n"
    )

    try:
        initial_response = client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature=0.7,
            max_tokens=150
        ).choices[0].message.content

        print(f"AI: {initial_response}\n")
        conversation_history.append({"role": "assistant", "content": initial_response})

    except Exception as e:
        print(f"Error during GPT initialization: {e}")
        return

    # main convo
    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Goodbye! 👋")
            break

        conversation_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                temperature=0.7,
                max_tokens=200
            )
            ai_response = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": ai_response})

            print(f"AI: {ai_response}\n")

            # add info to file
            if not faceKnown and memory_file_path:
                with open(memory_file_path, "a") as f:
                    f.write(f"\nUser said: {user_input}")
                    f.write(f"\nAI inferred: {ai_response}\n")

        except Exception as e:
            print(f"GPT error: {e}\n")

###########################################################################
# TESTING ONLY

if __name__ == "__main__":

    # supply a name here
    user_interaction_chat(faceKnown=True, person_name="Robert G")

    print("\n" + "=" * 50 + "\n")

    # unnknown
    user_interaction_chat(faceKnown=False)
