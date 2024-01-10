from llava.eval.run_llava_functions import eval_model_from_PIL
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from PIL import Image

class DictDotNotation(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self 

model_path = "liuhaotian/llava-v1.5-13b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_4bit=True
)

image_path = "/home/ace15208mc/object_state/LLaVA_easy_inference/test_image.png"
image_data = Image.open(image_path).convert("RGB")


args = DictDotNotation({
        "model": model,
        "model_name": model_name,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "image_data": [image_data],
        "query": "Describe this image", 
        "conv_mode": None,
        "seq": ",",
        "temperature": 0.2,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "device": 0
    })

print(eval_model_from_PIL(args))