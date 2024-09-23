# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import json
#
# # 加载模型和分词器
# model_id = "D:/text_fusion/CrossFuse-main/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )
#
# # 确保模型在GPU上
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# # 读取JSON文件
# with open("F:/MSRS-main/test/cropped_ir_captions.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # 新建一个字典来保存实体识别结果
# entities_data = {}
#
# # 遍历JSON文件中的键值对
# for key, value in data.items():
#     # 使用模型进行实体识别
#     prompt = f"Identify and list the entities in the sentence:: '{value}'"
#     inputs = tokenizer(prompt, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     outputs = model.generate(
#         inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         max_length=50,
#         pad_token_id=model.config.pad_token_id
#     )
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # 将识别的实体保存到新的字典中
#     entities_data[key] = generated_text
#     # 打印每一步生成的新的键值对
#     # print(f"{key}: {generated_text}")
#
# # 将新的字典写回JSON文件
# with open("F:/MSRS-main/test/cropped_ir_captions_entities.json", "w", encoding="utf-8") as f:
#     json.dump(entities_data, f, ensure_ascii=False, indent=4)
#
# print("Entities have been extracted and saved to 'cropped_ir_captions_entities.json'")

# import json
# import spacy
#
# # 加载spaCy的英语模型
# nlp = spacy.load("en_core_web_sm")
#
# def extract_entities(description):
#     """提取文本中的命名实体"""
#     doc = nlp(description)
#     entities = [ent.text for ent in doc.ents]
#     return entities
#
# def create_entity_dict(json_data):
#     """创建一个新的字典，包含图像名和对应的命名实体"""
#     entity_dict = {}
#     for image_name, description in json_data.items():
#         entities = extract_entities(description)
#         entity_dict[image_name] = entities
#     return entity_dict
#
# # 读取JSON文件
# with open('F:/MSRS-main/test/cropped_ir_captions.json', 'r', encoding='utf-8') as file:
#     json_data = json.load(file)
#
# # 创建包含命名实体的新字典
# new_dict = create_entity_dict(json_data)
#
# # 打印结果或保存到文件
# print(new_dict)
# # 如果需要保存到文件，可以使用以下代码
# with open('F:/MSRS-main/test/cropped_ir_captions_entities.json', 'w', encoding='utf-8') as file:
#     json.dump(new_dict, file, ensure_ascii=False, indent=4)





import json
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

# 创建SceneGraphParser实例，指定使用CUDA（如果可用）
parser = SceneGraphParser('D:/text_fusion/CrossFuse-main/lizhuang144flan-t5-base-VG-factual-sg', device='cuda')

# 读取JSON文件中的字典
with open('F:/MSRS-main/train/cropped_vi_captions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 创建一个空字典来存储新的键值对
new_data = {}

# 遍历字典中的每个键值对
for key, text in data.items():
    # 使用SceneGraphParser解析文本
    graph_obj = parser.parse([text], beam_size=5, return_text=False, max_output_len=128)

    # 提取只包含'head'部分的entities列表
    heads_only = [entity['head'] for entity in graph_obj[0]['entities']]

    # 将包含'head'部分的列表与原始键组成新的键值对
    new_data[key] = heads_only


# 打印新的键值对
for key, heads in new_data.items():
    print(f"{key}: {heads}")  # 打印键和实体的'head'部分

# 将新字典保存为新的JSON文件
with open('F:/MSRS-main/train/cropped_vi_captions_new.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)