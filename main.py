from src.chatbot import get_response


print("Chatbot is running (type 'quit' to exit)")
while True:
    user = input("You: ")
    if user.lower() == "quit":
        break
    print("Bot:", get_response(user))
