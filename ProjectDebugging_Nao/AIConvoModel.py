# AIConvoModel.py
import os
import json
import time
from openai import OpenAI

client = OpenAI(api_key="KEYHERE")

def get_memory_database_path():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    memory_db = os.path.join(desktop, "MemoryDatabase")
    if not os.path.exists(memory_db):
        os.makedirs(memory_db)
    return memory_db

def load_known_user_info(person_name: str):
    file_path = os.path.join(get_memory_database_path(), f"{person_name}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read().strip()
    return None

def create_unknown_user_file():
    timestamp = int(time.time())
    filename = f"UnknownUser_{timestamp}.txt"
    file_path = os.path.join(get_memory_database_path(), filename)
    with open(file_path, "w") as f:
        f.write("Name: \nHobbies: \nLikes: \nAge: \nOccupation: \n")
    return file_path

def load_profile_as_dict(file_path):
    profile = {"Name": "", "Hobbies": "", "Likes": "", "Age": "", "Occupation": ""}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    if key in profile:
                        profile[key] = value.strip()
    except Exception:
        pass
    return profile

def save_profile_from_dict(file_path, profile_dict):
    try:
        with open(file_path, "w") as f:
            for key, value in profile_dict.items():
                f.write(f"{key}: {value}\n")
    except Exception:
        pass

def split_ai_response(ai_response):
    json_start = ai_response.find("{")
    if json_start == -1:
        return ai_response.strip(), None
    chat_text = ai_response[:json_start].strip()
    json_text = ai_response[json_start:].strip()
    return chat_text, json_text

def extract_profile_updates(ai_response_json):
    try:
        return json.loads(ai_response_json)
    except Exception:
        return {}

class AIConvoModel:
    def __init__(self, face_known=False, person_name=None):
        self.client = OpenAI(api_key="KEYHERE")
        self.face_known = face_known
        self.person_name = person_name
        self.memory_file_path = None
        self.model_name = "gpt-4o-mini"
        self.conversation_history = []
        self._prepare_system_prompt()

    def get_initial_greeting(self):
        # If conversation_history only contains system prompt, generate the first assistant message
        if len(self.conversation_history) == 1:
            response = self.ask_gpt("Start the conversation by greeting the user using their profile.")
            return response
        else:
            # If already has assistant message
            return self.conversation_history[-1]["content"]


    def _prepare_system_prompt(self):
        if self.face_known:
            user_info = load_known_user_info(self.person_name) or "No stored info available."
            system_prompt = (
                "You are a friendly and polite AI chatbot. The user is a known contact, and here is their "
                f"profile information:\n---\n{user_info}\n---\n"
                "Greet the user by their name, referencing a subject in their profile."
                "Keep responses concise and friendly."
            )
        else:
            self.memory_file_path = create_unknown_user_file()
            system_prompt = (
                "You are a friendly and polite AI chatbot. Your goal is to naturally learn "
                "the user's name, hobbies, likes, age, and occupation.\n"
                "After your normal conversational message, ALWAYS output a JSON block but do not print it into the conversation:\n"
                "{\n"
                '  "name": null or string,\n'
                '  "hobbies": null or string,\n'
                '  "likes": null or string,\n'
                '  "age": null or string,\n'
                '  "occupation": null or string\n'
                "}\n"
                "Start by introducing yourself and asking their name."
            )
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def ask_gpt(self, user_text):
        self.conversation_history.append({"role": "user", "content": user_text})
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                temperature=0.7,
                max_tokens=200
            )
            full_response = response.choices[0].message.content
            chat_text, json_text = split_ai_response(full_response)
            self.conversation_history.append({"role": "assistant", "content": full_response})

            # If unknown user, try to update profile
            if not self.face_known and self.memory_file_path and json_text:
                profile = load_profile_as_dict(self.memory_file_path)
                updates = extract_profile_updates(json_text)
                for key, value in updates.items():
                    field = key.capitalize()
                    if value and profile.get(field, "") == "":
                        profile[field] = value
                save_profile_from_dict(self.memory_file_path, profile)
                # Rename file if name is known now
                if profile.get("Name") and "UnknownUser" in self.memory_file_path:
                    new_file = os.path.join(get_memory_database_path(), f"{profile['Name']}.txt")
                    os.rename(self.memory_file_path, new_file)
                    self.memory_file_path = new_file

            return chat_text

        except Exception as e:
            print(f"GPT error: {e}")
            return "Sorry, I had trouble processing that."


