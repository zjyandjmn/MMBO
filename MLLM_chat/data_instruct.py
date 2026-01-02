import base64
import requests
import os
import re
import io
import random  
import numpy as np
import threading
from util import *
from PIL import Image
import time
import csv

class GPT4VSystem():
    def __init__(self, key, save_folder_path, save_file_name):
        # API Key
        self.api_key = key
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        self.save_folder_path = save_folder_path
        self.save_file_name = save_file_name

    def dialogue(self, messages: list, image_names: list=None, mos_list: list=None):
        text_prompt = messages[-1]["content"][0]["text"]
        payload = {"model": "gpt-4o", "messages": messages, "max_tokens": 2000}
        response = requests.post("https://4.0.wokaai.com/v1/chat/completions", headers=self.headers, json=payload)

        print(response.json())
        # Get GPT4V role
        re_role = response.json()['choices'][0]['message']['role']

        # get GPT4V answer
        re_content = response.json()['choices'][0]['message']['content']
        all_info_respond = response.json()
        token_num = all_info_respond['usage']['total_tokens']

        print("Your input prompt:\n" + text_prompt)
        print("Your input image name: {}\n".format(image_names))
        print("GPT4V:\n" + re_content)

        if mos_list is not None:
            print("Mos: {}\n".format(mos_list))

        return re_role, re_content, token_num
    
class MessageModule():
    def __init__(self):
        pass

    def define_text_dict(self, input_prompt):
        text_dict = {
            "type": "text",
            "text": input_prompt
        }
        return text_dict

    def define_image_dict(self, encoded_image):
        image_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        }
        return image_dict

    def encode_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.resize((1536, 768))
            img = img.convert('RGB')
            with io.BytesIO() as byte_stream:
                img.save(byte_stream, format='png')
                byte_stream.seek(0)
                encoded_string = base64.b64encode(byte_stream.read()).decode('utf-8')
                return encoded_string
    
    def define_one_content(self, text_prompt: str="", image_path_list: list=[]):
        input_text_dict = self.define_text_dict(text_prompt)
        image_num = 0

        if image_path_list != []:
            input_image_dicts = []
            for i in range(len(image_path_list)):
                img_path = image_path_list[i]
                encoded_image = self.encode_image(img_path)
                cur_image_dict = self.define_image_dict(encoded_image)
                input_image_dicts.append(cur_image_dict)

            image_num = len(input_image_dicts)

        # Add image dicts
        content = [input_text_dict]
        if image_num != 0:
            for i in range(image_num):
                content.append(input_image_dicts[i])
        
        return content

    def define_one_message(self, role: str="user", content: list=None):
        message = [
            {
                "role": role,
                "content": content
            }
        ]
        return message

    def add_message(
            self,
            original_message: list=None,
            role: str="user", # assistant
            text_prompt: str="",
            image_path_list: list=[]
        ):

        new_content = self.define_one_content(text_prompt=text_prompt, image_path_list=image_path_list)
        new_message = self.define_one_message(role=role, content=new_content)
        if original_message is None:
            print("创建会话")
            return new_message
        else:
            original_message.append(new_message[0])

        return original_message

#保存与gpt4对话的输出
def log_respone(
        folder_path,
        file_name,
        text_prompt,
        input_content,
        input_image_names,
        csv_name,
        csv_folder_path,
        token_num
    ):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    file_path = os.path.join(folder_path, file_name)
    csv_path = os.path.join(csv_folder_path,csv_name)
    result=input_content.replace('\n', ' ')
    with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([input_image_names, result])
    f = open(file_path, 'a')
    f.write("Your input prompt:\n{}\n".format(text_prompt))
    f.write("Your input image names: {}\n".format(input_image_names))
    try:
        f.write("GPT4V:\n{}\n".format(input_content))
    except:
        f.write("GPT4V:\n{}\n".format("Something wrong"))


    f.write("Used token number: {}\n\n\n".format(token_num))
    f.close()

def extract_floats_from_string(s, str_following: str="Score:"):
    score_str = s.split(str_following)[1]
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", score_str)
    return [float(num) for num in numbers]


class CommandResult:
    def __init__(self):
        self.re_role = None
        self.re_content = None
        self.token_num = None


def execute_with_timeout(timeout_seconds, GPT4v, message, image_name_list):
    result_container = CommandResult()
    thread = threading.Thread(
        target=execute_command, args=(result_container, GPT4v, message, image_name_list))
    thread.daemon=True
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print("Time Out...........")
        thread.join()
        time.sleep(1)
        return execute_with_timeout(timeout_seconds)
    else:
        return result_container.re_role, result_container.re_content, result_container.token_num

def execute_command(result_container, GPT4v, message, image_name_list):
    result_container.re_role, result_container.re_content, result_container.token_num = GPT4v.dialogue(
        messages=message, image_names=image_name_list
    )
    
def split_dataset_name(csv_path,directory):
    image_names=[]
    image_paths=[]
    with open(csv_path, 'r') as file:  # Replace 'file.csv' with the actual file path
         data = csv.reader(file)
        #  next(data)
         for row in data:
            image_names.append(row[0])
            image_paths.append(os.path.join(directory,row[0]))
    return image_paths,image_names


def get_image_paths_and_names(directory):
    image_extensions = ['.jpg', '.png', '.jpeg', '.tiff', '.bmp', '.gif']  # 添加或者删除需要的图片格式
    image_paths = []
    image_names = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
                image_names.append(file)

    return image_paths, image_names

#确保与模型的对话顺利进行，并且在出现问题时进行适当的处理。
def dialogue_with_check(
        message,
        GPT4v,
        MesModule,
        text_prompt,
        image_name_list,
        image_path_list,
    ):
    MesModule=MessageModule()
    error_times = 0
    while True:
        # Check whether sucessfully dialogue
        try:
            message = MesModule.add_message(
                original_message=message,
                role="user", text_prompt=text_prompt, image_path_list=image_path_list
            )

            # Add thread to control time
            re_role, re_content, token_num = execute_with_timeout(
                20, GPT4v, message, image_name_list)

                    
            break
        except:
            print("Meet error......{}".format(error_times))
            error_times += 1
            message = None
        
            if error_times >= 3:
                re_role = ""

                re_content = "This is a High quality image"
                token_num = 0
                break
    
    return re_role, re_content, token_num

def GPT4_dataset_instruct(
        key: str,
        dataset_name: str,
        image_path_list: list,
        image_name_list: list,
        text_prompt: str,
        save_folder_path: str,
        save_file_name: str,
        csv_name: str,
        csv_folder_path: str,
        ic_path: str
    ):

    GPT4v = GPT4VSystem(
        key=key, save_folder_path=save_folder_path, save_file_name=save_file_name
    )
    MesModule = MessageModule()
    print("path length {}, name length {}".format(
            len(image_path_list), len(image_name_list)
        )
    )

    # process input list
    chunked_image_path_list = chunk_list(original_list=image_path_list, num=1)
    chunked_image_name_list = chunk_list(original_list=image_name_list, num=1)

    # # saving the score results
    # result_file_name = "{}_NR_num-{}_gpt4v.txt".format(
    #     dataset_name, len(image_path_list))    
    # result_save_path = os.path.join(save_folder_path, result_file_name)
    # f2 = open(result_save_path, 'a')

    for i in range(len(chunked_image_path_list)):
        # need to send single elemnet
        print("Current {} image names {}".format(i + 1, chunked_image_name_list[i]))
        print(chunked_image_path_list[i])
        cur_image_path = ic_path + chunked_image_path_list[i]
        re_role, re_content,token_num = dialogue_with_check(
            message=None,
            GPT4v=GPT4v,
            MesModule=MesModule,
            text_prompt=text_prompt,
            image_name_list=chunked_image_name_list[i],
            image_path_list=cur_image_path,
        )

        # if error happend 10 times
        log_respone(
            folder_path=save_folder_path,
            file_name=save_file_name,
            text_prompt=text_prompt,
            input_content=re_content,
            input_image_names=chunked_image_name_list[i],
            csv_name=csv_name,
            csv_folder_path=csv_folder_path,
            token_num=token_num
        )
    
    # f2.close()

if __name__ == "__main__":
    directory = '/mnt/10T/wkc/Database/OIQ-10K/OIQ-10K_image'  # 将这里替换为你的目录路径
    csv_path='/mnt/10T/zjy/database/oiq_10k_wz.csv'
    image_paths,image_names = split_dataset_name(csv_path,directory)
    key="sk-T4e25lpZr6PsN9djF7107dB585D74881Ba4cAe14CaF6614a"
    dataset_name="OIQ_10k"
    image_path_list=image_paths
    image_name_list=image_names
    text_prompt="Please generate image captions for this omnidirectional image in terms of quality"
    save_folder_path="/mnt/10t/zjy/OIQA-database/result"
    csv_folder_path="/mnt/10t/zjy/OIQA-database/result/csv_result"
    csv_name="caption_test_OIQ_10k.csv"
    save_file_name="Gemini_OQA_10k.txt"
    ic_path=""
    GPT4_dataset_instruct(
            key=key,
            dataset_name=dataset_name,
            image_path_list=image_path_list,
            image_name_list=image_name_list,
            text_prompt=text_prompt,
            save_folder_path=save_folder_path,
            save_file_name=save_file_name,
            csv_name=csv_name,
            csv_folder_path=csv_folder_path,
            ic_path=[] 
        )
    