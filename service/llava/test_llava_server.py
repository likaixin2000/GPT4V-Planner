from PIL import Image

from api import create_conversation, chat
from utils import construct_html_page_from_chat

def test_llava_server():
    file_path = '/home/kaixin/vision_feedback/src/owl_vit/images/scenic_design.jpg'
    image = Image.open(file_path)


    chat_history = create_conversation(model_name="llava-v1.5-13b")

    text_response, chat_history = chat(chat_history, "Describe the design in the demonstration figure in details.", image, 
        temperature=1.0, 
        top_p=10, 
        max_new_tokens=1000, 
        model_name="llava-v1.5-13b", 
        server_addr="http://0.0.0.0:21002"
    )
    

    print("_"*100)
    print(text_response)
    print("Whole history:\n", chat_history)

    construct_html_page_from_chat(chat_history, './')

# Run the test function
if __name__ == "__main__":
    test_llava_server()
