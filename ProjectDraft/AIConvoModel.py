import os
import socket
import threading
import sys
import queue
import sounddevice as sd
import numpy as np
import wave
from tempfile import NamedTemporaryFile
import time
import json
from openai import OpenAI

client = OpenAI(api_key='ADD IN THE KEY HERE')

# Get file path #########################################################
def get_memory_database_path():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    memory_db = os.path.join(desktop, "MemoryDatabase")
    return memory_db


# Send info to GPT #########################################################
def load_known_user_info(person_name: str):
    # searches for '<person_name>.txt and returns

    memory_db = get_memory_database_path()
    file_path = os.path.join(memory_db, f"{person_name}.txt")

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                print(f"File is read")
                return f.read().strip()
        except Exception as e:
            print(f"Error with known file: {e}")
            return None
    else:
        print(f"No file found for {person_name}: {file_path}")
        return None


# Create unknown #########################################################
def create_unknown_user_file():
    # make new file and return

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
        print(f"Created new memory file: {file_path}")
    except Exception as e:
        print(f"Error creating unknown user file: {e}")
        return None

    return file_path

# Convert profile text file → dict #########################################################
def load_profile_as_dict(file_path):
    profile = {"Name": "", "Hobbies": "", "Likes": "", "Age": "", "Occupation": ""}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    if key.strip() in profile:
                        profile[key.strip()] = value.strip()
    except:
        pass
    return profile


# Save dict → back into the profile file #########################################################
def save_profile_from_dict(file_path, profile_dict):
    with open(file_path, "w") as f:
        for key, value in profile_dict.items():
            f.write(f"{key}: {value}\n")


# Extract JSON block from the AI response #########################################################
def extract_profile_updates(ai_response):
    try:
        start = ai_response.find("{")
        end = ai_response.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(ai_response[start:end+1])
    except:
        return {}
    
# Separate text from JSON in AI response #############################
def split_ai_response(ai_response):
    json_start = ai_response.find("{")
    if json_start == -1:
        return ai_response.strip(), None  # no JSON found
    
    chat_text = ai_response[:json_start].strip()
    json_text = ai_response[json_start:].strip()
    
    return chat_text, json_text

#########################################################
# MAIN CHAT #########################################################
#########################################################
def user_interaction_chat(faceKnown: bool, person_name: str = None):
    model_name = "gpt-4o-mini"
    memory_file_path = None

    # known
    if faceKnown:
        if not person_name:
            raise ValueError("faceKnown=True requires a person_name.")

        print(f"Face Recognized. Getting profile for {person_name}...")

        user_info = load_known_user_info(person_name)
        if not user_info:
            user_info = "No stored info available."

        system_prompt = (
            "You are a friendly and polite AI chatbot. The user is a known contact, and here is their "
            f"profile information:\n---\n{user_info}\n---\n"
            "Greet the user by their name, referencing a subject in their profile."
            "Keep responses concise and friendly."
        )

    # unknown 
    else:
        print("Face Not Recognized.")

        memory_file_path = create_unknown_user_file()

        system_prompt = (
            "You are a friendly and polite AI chatbot. Your goal is to naturally learn "
            "the user's name, hobbies, likes, age, and occupation.\n"
            "After your normal conversational message, ALWAYS output a JSON block:\n"
            "{\n"
            '  "name": null or string,\n'
            '  "hobbies": null or string,\n'
            '  "likes": null or string,\n'
            '  "age": null or string,\n'
            '  "occupation": null or string\n'
            "}\n"
            "Start by introducing yourself and asking their name."
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

    # main convo loop
    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Bye!")
            break

        conversation_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                temperature=0.7,
                max_tokens=200
            )
            full_response = response.choices[0].message.content
            chat_text, json_text = split_ai_response(full_response)

            # Show only chat portion
            print(f"AI: {chat_text}\n")

            conversation_history.append({"role": "assistant", "content": full_response})

            if not faceKnown and memory_file_path:
                profile = load_profile_as_dict(memory_file_path)

                updates = extract_profile_updates(json_text if json_text else "{}")

                for key, value in updates.items():
                    field = key.capitalize()
                    if value and profile[field] == "":
                        profile[field] = value

                save_profile_from_dict(memory_file_path, profile)

                if profile["Name"] and "UnknownUser" in memory_file_path:
                    new_file = os.path.join(
                        get_memory_database_path(),
                        f"{profile['Name']}.txt"
                    )
                    os.rename(memory_file_path, new_file)
                    memory_file_path = new_file
                    print(f"[File renamed → {new_file}]")

        except Exception as e:

            print(f"GPT error: {e}\n")
